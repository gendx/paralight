// Copyright 2025-2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{IntoParallelSource, ParallelSource, SourceCleanup, SourceDescriptor};
use std::mem::ManuallyDrop;

/// A parallel source over a [`Vec`]. This struct is created by the
/// [`into_par_iter()`](IntoParallelSource::into_par_iter) method on
/// [`IntoParallelSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
///
/// See also [`SliceParallelSource`](super::slice::SliceParallelSource) and
/// [`MutSliceParallelSource`](super::slice::MutSliceParallelSource).
///
/// ```
/// # use paralight::iter::VecParallelSource;
/// # use paralight::prelude::*;
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let input = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let iter: VecParallelSource<_> = input.into_par_iter();
/// let sum = iter.with_thread_pool(&mut thread_pool).sum::<i32>();
/// assert_eq!(sum, 5 * 11);
/// ```
#[must_use = "iterator adaptors are lazy"]
pub struct VecParallelSource<T> {
    vec: Vec<T>,
}

impl<T: Send> IntoParallelSource for Vec<T> {
    type Item = T;
    type Source = VecParallelSource<T>;

    fn into_par_iter(self) -> Self::Source {
        VecParallelSource { vec: self }
    }
}

impl<T: Send> IntoParallelSource for Box<[T]> {
    type Item = T;
    type Source = VecParallelSource<T>;

    fn into_par_iter(self) -> Self::Source {
        // There's no Box::<[T]>::from_raw_parts(), so we just piggy back on Vec.
        VecParallelSource {
            vec: self.into_vec(),
        }
    }
}

impl<T: Send> ParallelSource for VecParallelSource<T> {
    type Item = T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let mut vec = ManuallyDrop::new(self.vec);
        let mut_ptr = vec.as_mut_ptr();
        let len = vec.len();
        let capacity = vec.capacity();

        VecSourceDescriptor {
            ptr: MutPtrWrapper(mut_ptr),
            len,
            capacity,
        }
    }
}

struct VecSourceDescriptor<T> {
    ptr: MutPtrWrapper<T>,
    len: usize,
    capacity: usize,
}

impl<T: Send> SourceCleanup for VecSourceDescriptor<T> {
    const NEEDS_CLEANUP: bool = std::mem::needs_drop::<T>();

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= self.len);
            debug_assert!(range.end <= self.len);
            let base_ptr: *mut T = self.ptr.get();
            // SAFETY:
            // - The offset in bytes `range.start * size_of::<T>()` fits in an `isize`,
            //   because the range is included in the length of the (well-formed) input
            //   vector. This is ensured by the safety pre-conditions of the
            //   `cleanup_item_range()` function (the `range` must be included in
            //   `0..self.len`).
            // - The `base_ptr` is derived from an allocated object (the input vector), and
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
            //   pre-conditions of the `cleanup_item_range()` function (each index appears
            //   at most once in calls to `fetch_item()` and `cleanup_item_range()`), i.e.
            //   the range of items in this slice isn't accessed by anything else.
            // - The `slice` is valid for dropping, as it is a part of the input vector that
            //   nothing else accesses.
            // - Nothing else is accessing the `slice` while `drop_in_place` is executing.
            //
            // The above properties (aligned, non-null, etc.) still hold if the slice is
            // empty.
            unsafe { std::ptr::drop_in_place(slice) };
        }
    }
}

impl<T: Send> SourceDescriptor for VecSourceDescriptor<T> {
    type Item = T;

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        let base_ptr: *const T = self.ptr.get();
        // SAFETY:
        // - The offset in bytes `index * size_of::<T>()` fits in an `isize`, because
        //   the index is smaller than the length of the (well-formed) input vector.
        //   This is ensured by the safety pre-conditions of the `fetch_item()` function
        //   (the `index` must be in the range `0..self.len`), and further confirmed by
        //   the assertion.
        // - The `base_ptr` is derived from an allocated object (the input vector), and
        //   the entire range between `base_ptr` and the resulting `item_ptr` is in
        //   bounds of that allocated object. This is because the index is smaller than
        //   the length of the input vector.
        let item_ptr: *const T = unsafe { base_ptr.add(index) };
        // SAFETY:
        // - The `item_ptr` is properly aligned, as it is constructed by calling `add()`
        //   on the aligned `base_ptr`.
        // - The `item_ptr` points to a properly initialized value of type `T`, the
        //   element from the input vector at position `index`.
        // - The `item_ptr` is valid for reads. This is ensured by the safety
        //   pre-conditions of the `fetch_item()` function (each index must be passed at
        //   most once), i.e. this item hasn't been read (and moved out of the vector)
        //   yet. Additionally, there are no concurrent writes to this slot in the
        //   vector.
        let item: T = unsafe { std::ptr::read(item_ptr) };
        item
    }
}

impl<T> Drop for VecSourceDescriptor<T> {
    fn drop(&mut self) {
        let base_ptr: *mut T = self.ptr.get();
        // SAFETY:
        // - The `base_ptr` has been allocated with the global allocator, as it is
        //   derived from the source vector.
        // - `T` has the same alignement as what `base_ptr` was allocated with, because
        //   `base_ptr` derives from a vector of `T`s.
        // - `T * capacity` is the size of what `base_ptr` was allocated with, because
        //   that's the capacity of the source vector.
        // - `length <= capacity` because the `length` is set to zero here.
        // - The first `length` values are properly initialized values of type `T`
        //   because the `length` is set to zero.
        // - The allocated size in bytes isn't larger than `isize::MAX`, because that's
        //   derived from the source vector.
        let vec: Vec<T> = unsafe { Vec::from_raw_parts(base_ptr, 0, self.capacity) };
        drop(vec);
    }
}

/// A helper struct for the implementation of [`VecSourceDescriptor`], that
/// wraps a [`*mut T`](pointer). This enables sending [`&mut [T]`](slice) to
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
/// items of type [`&mut [T]`](slice) to other threads (see the safety
/// comments in [`VecSourceDescriptor::fetch_item`] and
/// [`VecSourceDescriptor::cleanup_item_range`]). Therefore we make it [`Sync`]
/// if and only if [`&mut [T]`](slice) is [`Send`], which is when `T` is
/// [`Send`].
unsafe impl<T: Send> Sync for MutPtrWrapper<T> {}
