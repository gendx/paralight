// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{IntoParallelSource, ParallelSource, SourceCleanup, SourceDescriptor};
use std::cell::UnsafeCell;
use std::mem::MaybeUninit;

/// A parallel source over an [array](array). This struct is created by the
/// [`into_par_iter()`](IntoParallelSource::into_par_iter) method on
/// [`IntoParallelSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
///
/// See also [`SliceParallelSource`](super::slice::SliceParallelSource) and
/// [`MutSliceParallelSource`](super::slice::MutSliceParallelSource).
///
/// ### Stability blockers
///
/// This struct is currently only available on Rust nightly with the `nightly`
/// feature of Paralight enabled. This is because the implementation depends on
/// the following nightly Rust features:
/// - [`array_ptr_get`](https://github.com/rust-lang/rust/issues/119834),
/// - [`maybe_uninit_uninit_array_transpose`](https://github.com/rust-lang/rust/issues/96097).
///
/// ```
/// # use paralight::iter::{
/// #     ArrayParallelSource, IntoParallelSource, ParallelIteratorExt, ParallelSourceExt,
/// # };
/// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let iter: ArrayParallelSource<_, 10> = input.into_par_iter();
/// let sum = iter.with_thread_pool(&mut thread_pool).sum::<i32>();
/// assert_eq!(sum, 5 * 11);
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "nightly")))]
#[must_use = "iterator adaptors are lazy"]
pub struct ArrayParallelSource<T, const N: usize> {
    array: [T; N],
}

impl<T: Send, const N: usize> IntoParallelSource for [T; N] {
    type Item = T;
    type Source = ArrayParallelSource<T, N>;

    fn into_par_iter(self) -> Self::Source {
        ArrayParallelSource { array: self }
    }
}

impl<T: Send, const N: usize> ParallelSource for ArrayParallelSource<T, N> {
    type Item = T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        ArraySourceDescriptor {
            array: ArrayWrapper::new(self.array),
        }
    }
}

struct ArraySourceDescriptor<T, const N: usize> {
    array: ArrayWrapper<T, N>,
}

impl<T: Send, const N: usize> SourceCleanup for ArraySourceDescriptor<T, N> {
    const NEEDS_CLEANUP: bool = std::mem::needs_drop::<T>();

    fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            let base_ptr: *mut T = self.array.start();
            // SAFETY:
            // - The offset in bytes `range.start * size_of::<T>()` fits in an `isize`,
            //   because the range is included in the length of the (well-formed) wrapped
            //   array. This is ensured by the thread pool's `pipeline()` function (which
            //   only yields in-bound ranges for cleanup).
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
            // - The `slice` is valid for reads and writes. This is ensured by the thread
            //   pool's `pipeline()` function, which yields non-overlapping indices and
            //   cleanup ranges. I.e. the range of items in this slice isn't accessed by
            //   anything else.
            // - The `slice` is valid for dropping, as it is a part of the wrapped array
            //   that nothing else accesses.
            // - Nothing else is accessing the `slice` while `drop_in_place` is executing.
            //
            // The above properties (aligned, non-null, etc.) still hold if the `slice` is
            // empty.
            unsafe { std::ptr::drop_in_place(slice) };
        }
    }
}

impl<T: Send, const N: usize> SourceDescriptor for ArraySourceDescriptor<T, N> {
    type Item = T;

    fn len(&self) -> usize {
        N
    }

    fn fetch_item(&self, index: usize) -> Self::Item {
        assert!(index < N);
        let base_ptr: *const T = self.array.start();
        // SAFETY:
        // - The offset in bytes `index * size_of::<T>()` fits in an `isize`, because
        //   the index is smaller than the length of the (well-formed) wrapped array.
        //   This is ensured by the thread pool's `pipeline()` function (which yields
        //   indices in the range `0..len`), and further confirmed by the assertion.
        // - The `base_ptr` is derived from an allocated object (the wrapped array), and
        //   the entire range between `base_ptr` and the resulting `item_ptr` is in
        //   bounds of that allocated object. This is because the index is smaller than
        //   the length of the wrapped array.
        let item_ptr: *const T = unsafe { base_ptr.add(index) };
        // SAFETY:
        // - The `item_ptr` is properly aligned, as it is constructed by calling `add()`
        //   on the aligned `base_ptr`.
        // - The `item_ptr` points to a properly initialized value of type `T`, the
        //   element from the wrapped array at position `index`.
        // - The `item_ptr` is valid for reads. This is ensured by the thread pool's
        //   `pipeline()` function (which yields distinct indices in the range
        //   `0..len`), i.e. this item hasn't been read (and moved out of the array)
        //   yet. Additionally, there are no concurrent writes to this slot in the
        //   array.
        let item: T = unsafe { std::ptr::read(item_ptr) };
        item
    }
}

/// A helper struct for the implementation of [`ArraySourceDescriptor`], that
/// wraps a [`[T; N]`](array) and enables sending `T`s and [`&mut [T]`](slice)
/// to other threads.
///
/// This contains two layers of wrapping.
/// - Each item is wrapped in a [`MaybeUninit`], which inhibits default dropping
///   and ensures items are only accessed manually (preventing already moved
///   items from being accessed again).
/// - The whole array is wrapped in an [`UnsafeCell`], which allows obtaining
///   mutable references to the items in order to drop them in the
///   [`ArraySourceDescriptor::cleanup_item_range`] function.
///
/// Note: Contrary to the implementation of e.g. `VecSourceDescriptor`, wrapping
/// a pointer to the start of the array isn't enough. The wrapper needs to
/// somehow contain the array, otherwise items are accessed by worker threads
/// after they go out of scope.
struct ArrayWrapper<T, const N: usize>(UnsafeCell<[MaybeUninit<T>; N]>);

impl<T, const N: usize> ArrayWrapper<T, N> {
    fn new(array: [T; N]) -> Self {
        let array: [MaybeUninit<T>; N] = MaybeUninit::new(array).transpose();
        ArrayWrapper(UnsafeCell::new(array))
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
/// items of type [`&mut [T]`](slice) to other threads (see the safety
/// comments in [`ArraySourceDescriptor::fetch_item`] and
/// [`ArraySourceDescriptor::cleanup_item_range`]). Therefore we make it
/// [`Sync`] if and only if [`&mut [T]`](slice) is [`Send`], which is when `T`
/// is [`Send`].
unsafe impl<T: Send, const N: usize> Sync for ArrayWrapper<T, N> {}
