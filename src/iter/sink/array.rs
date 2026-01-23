// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{ExactParallelSink, FromExactParallelSink};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Mutex;

impl<T: Send, const N: usize> FromExactParallelSink for [T; N] {
    type Item = T;
    type Sink = ArrayParallelSink<T, N>;

    unsafe fn finalize(sink: Self::Sink) -> Self {
        debug_assert!(sink.skipped.into_inner().unwrap().is_empty());

        // SAFETY:
        //
        // All the values are properly initialized, as the safety pre-conditions of
        // `finalize()` require that each index in `0..N` was passed once and only once
        // to `ArrayParallelSink::push_item`, and that no call to
        // `ArrayParallelSink::skip_item_range` was made.
        unsafe { sink.array.into_array() }
    }
}

/// A parallel sink towards an [array](array). This struct is consumed by the
/// [`finalize()`](FromExactParallelSink::finalize) method on
/// [`FromExactParallelSink`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSink`] trait, but it is nonetheless public as
/// it is an associated type of [`FromExactParallelSink`] for [array](array) and
/// similar collections.
#[must_use = "iterator adaptors are lazy"]
pub struct ArrayParallelSink<T: Send, const N: usize> {
    array: ArrayWrapper<T, N>,
    skipped: Mutex<Vec<std::ops::Range<usize>>>,
}

impl<T: Send, const N: usize> ExactParallelSink for ArrayParallelSink<T, N> {
    type Item = T;
    const NEEDS_CLEANUP: bool = std::mem::needs_drop::<T>();

    fn new(len: usize) -> Self {
        assert_eq!(
            len, N,
            "tried to collect an iterator into an array of the wrong length"
        );

        Self {
            array: ArrayWrapper::new(),
            skipped: Mutex::new(Vec::new()),
        }
    }

    unsafe fn push_item(&self, index: usize, item: Self::Item) {
        debug_assert!(index < N);
        let base_ptr: *mut T = self.array.start();
        // SAFETY:
        // - The offset in bytes `index * size_of::<T>()` fits in an `isize`, because
        //   the index is smaller than the length of the (well-formed) wrapped array.
        //   This is ensured by the safety pre-conditions of the `push_item()` function
        //   (the `index` must be in the range `0..N`), and further confirmed by the
        //   assertion.
        // - The `base_ptr` is derived from an allocated object (the wrapped array), and
        //   the entire range between `base_ptr` and the resulting `item_ptr` is in
        //   bounds of that allocated object. This is because the index is smaller than
        //   the length of the wrapped array.
        let item_ptr: *mut T = unsafe { base_ptr.add(index) };
        // SAFETY:
        // - The `item_ptr` is properly aligned, as it is constructed by calling `add()`
        //   on the aligned `base_ptr`.
        // - The `item_ptr` points to a not yet initialized value of type `T`, the
        //   element from the wrapped array at position `index`.
        // - The `item_ptr` is valid for writes. This is ensured by the safety
        //   pre-conditions of the `push_item()` function (each index must be passed at
        //   most once), i.e. this item hasn't yet been written to. Additionally, there
        //   are no concurrent reads nor writes to this slot in the array.
        unsafe { std::ptr::write(item_ptr, item) };
    }

    unsafe fn skip_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= N);
            debug_assert!(range.end <= N);
            self.skipped.lock().unwrap().push(range);
        }
    }

    unsafe fn cancel(self) {
        let base_ptr: *mut T = self.array.start();

        if Self::NEEDS_CLEANUP {
            // Drop all items, except those that were skipped.
            let mut skipped = self.skipped.into_inner().unwrap();
            skipped.sort_unstable_by(|a, b| a.start.cmp(&b.start));

            let mut prev = 0..0;
            for range in skipped.into_iter() {
                Self::cleanup_item_range(base_ptr, prev.end..range.start);
                prev = range.clone();
            }

            Self::cleanup_item_range(base_ptr, prev.end..N);
        }

        // We can just forget the ArrayWrapper.
    }
}

impl<T: Send, const N: usize> ArrayParallelSink<T, N> {
    fn cleanup_item_range(base_ptr: *mut T, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= N);
            debug_assert!(range.end <= N);
            // SAFETY:
            // - The offset in bytes `range.start * size_of::<T>()` fits in an `isize`,
            //   because the range is included in the length of the (well-formed) wrapped
            //   array. This is ensured by the safety pre-conditions of the
            //   `ExactParallelSink::skip_item_range()` function (the `range` must be
            //   included in `0..N`).
            // - The `base_ptr` is derived from an allocated object (the wrapped array), and
            //   the entire range between `base_ptr` and the resulting `start_ptr` is in
            //   bounds of that allocated object. This is because the range start is smaller
            //   than the length of the wrapped array.
            let start_ptr: *mut T = unsafe { base_ptr.add(range.start) };
            let slice: *mut [T] =
                std::ptr::slice_from_raw_parts_mut(start_ptr, range.end - range.start);
            // SAFETY:
            // - The `slice` is properly aligned, as it is constructed by calling `add()` on
            //   the aligned `base_ptr`.
            // - The `slice` isn't null, as it is constructed by calling `add()` on the
            //   non-null `base_ptr`.
            // - The `slice` is valid for reads and writes. This is ensured by the safety
            //   pre-conditions of the `ExactParallelSink::cancel()` function (each index
            //   appears once in calls to `push_item()` and `skip_item_range()`), i.e. the
            //   items in this slice have all been pushed.
            // - The `slice` is valid for dropping, as it is a part of the wrapped array
            //   that nothing else accesses.
            // - Nothing else is accessing the `slice` while `drop_in_place` is executing.
            //
            // The above properties (aligned, non-null, etc.) still hold if the slice is
            // empty.
            unsafe { std::ptr::drop_in_place(slice) };
        }
    }
}

/// A helper struct for the implementation of [`ArrayParallelSink`], that wraps
/// a [`[T; N]`](array) and enables sending `T`s and [`&mut [T]`](slice)
/// to other threads.
///
/// This contains two layers of wrapping.
/// - Each item is wrapped in a [`MaybeUninit`], which inhibits default dropping
///   and ensures items are only accessed manually (preventing not yet
///   initialized items from being accessed).
/// - The whole array is wrapped in an [`UnsafeCell`], which allows obtaining
///   mutable references to the items in order to write to them them in the
///   [`ArrayParallelSink::push_item`] function.
///
/// Note: Contrary to the implementation of e.g. `VecParallelSink`, wrapping
/// a pointer to the start of the array isn't enough. The wrapper needs to
/// somehow contain the array, otherwise items are accessed by worker threads
/// after they go out of scope.
struct ArrayWrapper<T, const N: usize>(UnsafeCell<[MaybeUninit<T>; N]>);

impl<T, const N: usize> ArrayWrapper<T, N> {
    fn new() -> Self {
        ArrayWrapper(UnsafeCell::new([const { MaybeUninit::uninit() }; N]))
    }

    /// # SAFETY
    ///
    /// This can only be called after all items in the array have been
    /// initialized.
    unsafe fn into_array(self) -> [T; N] {
        let array: [MaybeUninit<T>; N] = self.0.into_inner();
        let array: MaybeUninit<[T; N]> = array.transpose();
        // SAFETY: As ensured by the caller, the whole array is now in an initialized
        // state.
        unsafe { array.assume_init() }
    }

    fn start(&self) -> *mut T {
        let array_ptr: *mut [MaybeUninit<T>; N] = self.0.get();
        let start_ptr: *mut MaybeUninit<T> = array_ptr.as_mut_ptr();
        start_ptr as *mut T
    }
}

/// SAFETY:
///
/// An [`ArrayWrapper`] is meant to be shared among threads as a way to send
/// items of type [`&mut [T]`](slice) to other threads. Therefore we make it
/// [`Sync`] if and only if [`&mut [T]`](slice) is [`Send`], which is when `T`
/// is [`Send`].
unsafe impl<T: Send, const N: usize> Sync for ArrayWrapper<T, N> {}
