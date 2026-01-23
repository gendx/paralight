// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{ExactParallelSink, FromExactParallelSink};
use std::collections::VecDeque;
use std::mem::ManuallyDrop;
use std::sync::Mutex;

impl<T: Send> FromExactParallelSink for Vec<T> {
    type Item = T;
    type Sink = VecParallelSink<T>;

    unsafe fn finalize(sink: Self::Sink) -> Self {
        debug_assert!(sink.skipped.into_inner().unwrap().is_empty());

        let base_ptr: *mut T = sink.ptr.get();
        let len = sink.len;
        let capacity = sink.capacity;

        // SAFETY:
        // - The `base_ptr` has been allocated with the global allocator, as it is
        //   derived from the vector created in `VecParallelSink::new`.
        // - `T` has the same alignement as what `base_ptr` was allocated with, because
        //   `base_ptr` derives from a vector of `T`s.
        // - `T * capacity` is the size of what `base_ptr` was allocated with, because
        //   that's the capacity of the vector created in `VecParallelSink::new`.
        // - `len <= capacity` because the vector was created using `Vec::with_capacity`
        //   with `len` as the requested minimal capacity.
        // - The first `len` values are properly initialized values of type `T`, as the
        //   safety pre-conditions of `finalize()` require that each index in `0..len`
        //   was passed once and only once to `VecParallelSink::push_item`, and that no
        //   call to `VecParallelSink::skip_item_range` was made.
        // - The allocated size in bytes isn't larger than `isize::MAX`, because that's
        //   derived from the vector created in `VecParallelSink::new`.
        unsafe { Vec::from_raw_parts(base_ptr, len, capacity) }
    }
}

impl<T: Send> FromExactParallelSink for Box<[T]> {
    type Item = T;
    type Sink = VecParallelSink<T>;

    unsafe fn finalize(sink: Self::Sink) -> Self {
        // SAFETY: Safety is guaranteed by induction, as the pre-requisites of
        // `Self::finalize` are forwarded to `<FromExactParallelSink for
        // Vec<T>>::finalize`.
        let vec: Vec<T> = unsafe { FromExactParallelSink::finalize(sink) };
        vec.into_boxed_slice()
    }
}

impl<T: Send> FromExactParallelSink for VecDeque<T> {
    type Item = T;
    type Sink = VecParallelSink<T>;

    unsafe fn finalize(sink: Self::Sink) -> Self {
        // SAFETY: Safety is guaranteed by induction, as the pre-requisites of
        // `Self::finalize` are forwarded to `<FromExactParallelSink for
        // Vec<T>>::finalize`.
        let vec: Vec<T> = unsafe { FromExactParallelSink::finalize(sink) };
        vec.into()
    }
}

/// A parallel sink towards a [`Vec`]. This struct is consumed by the
/// [`finalize()`](FromExactParallelSink::finalize) method on
/// [`FromExactParallelSink`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSink`] trait, but it is nonetheless public as
/// it is an associated type of [`FromExactParallelSink`] for [`Vec`] and
/// similar collections.
#[must_use = "iterator adaptors are lazy"]
pub struct VecParallelSink<T: Send> {
    ptr: MutPtrWrapper<T>,
    len: usize,
    capacity: usize,
    skipped: Mutex<Vec<std::ops::Range<usize>>>,
}

impl<T: Send> ExactParallelSink for VecParallelSink<T> {
    type Item = T;
    const NEEDS_CLEANUP: bool = std::mem::needs_drop::<T>();

    fn new(len: usize) -> Self {
        let vec: Vec<T> = Vec::with_capacity(len);

        // TODO(MSRV >= 1.93.0): Use Vec::into_raw_parts().
        let mut vec = ManuallyDrop::new(vec);
        let mut_ptr = vec.as_mut_ptr();
        let capacity = vec.capacity();
        debug_assert_eq!(vec.len(), 0);

        Self {
            ptr: MutPtrWrapper(mut_ptr),
            len,
            capacity,
            skipped: Mutex::new(Vec::new()),
        }
    }

    unsafe fn push_item(&self, index: usize, item: Self::Item) {
        debug_assert!(index < self.len);
        let base_ptr: *mut T = self.ptr.get();
        // SAFETY:
        // - The offset in bytes `index * size_of::<T>()` fits in an `isize`, because
        //   the index is smaller than the length of the (well-formed) output vector.
        //   This is ensured by the safety pre-conditions of the `push_item()` function
        //   (the `index` must be in the range `0..self.len`), and further confirmed by
        //   the assertion.
        // - The `base_ptr` is derived from an allocated object (the output vector), and
        //   the entire range between `base_ptr` and the resulting `item_ptr` is in
        //   bounds of that allocated object. This is because the index is smaller than
        //   the length of the output vector.
        let item_ptr: *mut T = unsafe { base_ptr.add(index) };
        // SAFETY:
        // - The `item_ptr` is properly aligned, as it is constructed by calling `add()`
        //   on the aligned `base_ptr`.
        // - The `item_ptr` points to a not yet initialized value of type `T`, the
        //   element from the input vector at position `index`.
        // - The `item_ptr` is valid for writes. This is ensured by the safety
        //   pre-conditions of the `push_item()` function (each index must be passed at
        //   most once), i.e. this item hasn't yet been written to. Additionally, there
        //   are no concurrent reads nor writes to this slot in the vector.
        unsafe { std::ptr::write(item_ptr, item) };
    }

    unsafe fn skip_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= self.len);
            debug_assert!(range.end <= self.len);
            self.skipped.lock().unwrap().push(range);
        }
    }

    unsafe fn cancel(self) {
        let base_ptr: *mut T = self.ptr.get();

        if Self::NEEDS_CLEANUP {
            // Drop all items, except those that were skipped.
            let mut skipped = self.skipped.into_inner().unwrap();
            skipped.sort_unstable_by(|a, b| a.start.cmp(&b.start));

            let mut prev = 0..0;
            for range in skipped.into_iter() {
                Self::cleanup_item_range(base_ptr, self.len, prev.end..range.start);
                prev = range.clone();
            }

            Self::cleanup_item_range(base_ptr, self.len, prev.end..self.len);
        }

        // Deallocate the vector.
        //
        // SAFETY:
        // - The `base_ptr` has been allocated with the global allocator, as it is
        //   derived from the vector created in `VecParallelSink::new`.
        // - `T` has the same alignement as what `base_ptr` was allocated with, because
        //   `base_ptr` derives from a vector of `T`s.
        // - `T * capacity` is the size of what `base_ptr` was allocated with, because
        //   that's the capacity of the vector created in `VecParallelSink::new`.
        // - `length <= capacity` because the `length` is set to zero here.
        // - The first `length` values are properly initialized values of type `T`
        //   because the `length` is set to zero.
        // - The allocated size in bytes isn't larger than `isize::MAX`, because that's
        //   derived from the vector created in `VecParallelSink::new`.
        let vec: Vec<T> = unsafe { Vec::from_raw_parts(base_ptr, 0, self.capacity) };
        drop(vec);
    }
}

impl<T: Send> VecParallelSink<T> {
    fn cleanup_item_range(base_ptr: *mut T, len: usize, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= len);
            debug_assert!(range.end <= len);
            // SAFETY:
            // - The offset in bytes `range.start * size_of::<T>()` fits in an `isize`,
            //   because the range is included in the length of the (well-formed) output
            //   vector. This is ensured by the safety pre-conditions of the
            //   `ExactParallelSink::skip_item_range()` function (the `range` must be
            //   included in `0..self.len`).
            // - The `base_ptr` is derived from an allocated object (the output vector), and
            //   the entire range between `base_ptr` and the resulting `start_ptr` is in
            //   bounds of that allocated object. This is because the range start is smaller
            //   than the length of the input vector.
            let start_ptr: *mut T = unsafe { base_ptr.add(range.start) };
            let slice: *mut [T] =
                std::ptr::slice_from_raw_parts_mut(start_ptr, range.end - range.start);
            // SAFETY:
            // - The `slice` is properly aligned, as it is constructed by calling `add()` on
            //   the aligned `base_ptr`.
            // - The `slice` isn't null, as it is constructed by calling `add()` on the
            //   non-null `base_ptr`. Indeed, `Vec` guarantees that its pointer is never
            //   null: https://doc.rust-lang.org/stable/std/vec/struct.Vec.html#guarantees.
            // - The `slice` is valid for reads and writes. This is ensured by the safety
            //   pre-conditions of the `ExactParallelSink::cancel()` function (each index
            //   appears once in calls to `push_item()` and `skip_item_range()`), i.e. the
            //   items in this slice have all been pushed.
            // - The `slice` is valid for dropping, as it is a part of the output vector
            //   that nothing else accesses.
            // - Nothing else is accessing the `slice` while `drop_in_place` is executing.
            //
            // The above properties (aligned, non-null, etc.) still hold if the slice is
            // empty.
            unsafe { std::ptr::drop_in_place(slice) };
        }
    }
}

/// A helper struct for the implementation of [`VecParallelSink`], that wraps a
/// [`*mut T`](pointer). This enables sending [`&mut [T]`](slice) to
/// other threads.
struct MutPtrWrapper<T>(*mut T);

impl<T> MutPtrWrapper<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

/// SAFETY:
///
/// A [`MutPtrWrapper`] is meant to be shared among threads as a way to send
/// items of type [`&mut [T]`](slice) to other threads. Therefore we make it
/// [`Sync`] if and only if [`&mut [T]`](slice) is [`Send`], which is when `T`
/// is [`Send`].
unsafe impl<T: Send> Sync for MutPtrWrapper<T> {}
