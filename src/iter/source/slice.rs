// Copyright 2024-2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{
    ExactParallelSource, ExactSourceDescriptor, IntoExactParallelRefMutSource,
    IntoExactParallelRefSource, SourceCleanup,
};
use std::marker::PhantomData;

impl<'data, T: Sync + 'data> IntoExactParallelRefSource<'data> for [T] {
    type Item = &'data T;

    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn par_iter(&'data self) -> impl ExactParallelSource<Item = Self::Item> {
        SliceParallelSource { slice: self }
    }
}

struct SliceParallelSource<'data, T> {
    slice: &'data [T],
}

impl<'data, T: Sync> ExactParallelSource for SliceParallelSource<'data, T> {
    type Item = &'data T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        SliceSourceDescriptor { slice: self.slice }
    }
}

struct SliceSourceDescriptor<'data, T: Sync> {
    slice: &'data [T],
}

impl<T: Sync> SourceCleanup for SliceSourceDescriptor<'_, T> {
    const NEEDS_CLEANUP: bool = false;

    fn len(&self) -> usize {
        self.slice.len()
    }

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {
        // Nothing to cleanup
    }
}

impl<'data, T: Sync> ExactSourceDescriptor for SliceSourceDescriptor<'data, T> {
    type Item = &'data T;

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.slice.len());
        // SAFETY: The index is smaller than the length of the input slice, due to the
        // safety pre-conditions of the `exact_fetch_item()` function.
        unsafe { self.slice.get_unchecked(index) }
    }
}

impl<'data, T: Send + 'data> IntoExactParallelRefMutSource<'data> for [T] {
    type Item = &'data mut T;

    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let mut values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// values
    ///     .par_iter_mut()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each(|x| *x *= 2);
    /// assert_eq!(values, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    /// ```
    fn par_iter_mut(&'data mut self) -> impl ExactParallelSource<Item = Self::Item> {
        MutSliceParallelSource { slice: self }
    }
}

struct MutSliceParallelSource<'data, T> {
    slice: &'data mut [T],
}

impl<'data, T: Send> ExactParallelSource for MutSliceParallelSource<'data, T> {
    type Item = &'data mut T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let len = self.slice.len();
        let ptr = MutPtrWrapper(self.slice.as_mut_ptr());
        MutSliceSourceDescriptor {
            len,
            ptr,
            _phantom: PhantomData,
        }
    }
}

struct MutSliceSourceDescriptor<'data, T: Send + 'data> {
    len: usize,
    ptr: MutPtrWrapper<T>,
    _phantom: PhantomData<&'data ()>,
}

impl<'data, T: Send + 'data> SourceCleanup for MutSliceSourceDescriptor<'data, T> {
    const NEEDS_CLEANUP: bool = false;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {
        // Nothing to cleanup
    }
}

impl<'data, T: Send + 'data> ExactSourceDescriptor for MutSliceSourceDescriptor<'data, T> {
    type Item = &'data mut T;

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        let base_ptr: *mut T = self.ptr.get();
        // SAFETY:
        // - The offset in bytes `index * size_of::<T>()` fits in an `isize`, because
        //   the index is smaller than the length of the (well-formed) input slice. This
        //   is ensured by the safety pre-conditions of the `exact_fetch_item()`
        //   function (the `index` must be in the range `0..self.len`), and further
        //   confirmed by the assertion.
        // - The `base_ptr` is derived from an allocated object (the input slice), and
        //   the entire range between `base_ptr` and the resulting `item_ptr` is in
        //   bounds of that allocated object. This is because the index is smaller than
        //   the length of the input slice.
        let item_ptr: *mut T = unsafe { base_ptr.add(index) };
        // SAFETY:
        //
        // From https://doc.rust-lang.org/std/ptr/index.html#pointer-to-reference-conversion:
        // - The `item_ptr` is properly aligned, as it is constructed by calling `add()`
        //   on the aligned `base_ptr`.
        // - The `item_ptr` is not null, as the `base_ptr` isn't null (obtained from a
        //   well-formed non-empty slice) and the `index` is in bounds of the slice (no
        //   wrap around).
        // - The `item_ptr` is dereferenceable, as the whole memory range of length
        //   `size_of::<T>()` starting from it is within bounds of a single allocated
        //   object (the input slice).
        // - The `item_ptr` points to a valid value of type `T`, the element from the
        //   input slice at position `index`.
        // - The `item_ptr` follows the aliasing rules: while this reference exists
        //   (within this scope and in particular during the call to `process_item()`),
        //   the memory it points to isn't accessed through any other pointer or
        //   reference. This is ensured by the safety pre-conditions of the
        //   `exact_fetch_item()` function (each index must be passed at most once), and
        //   because the slice is exclusively owned during the scope of this
        //   `ParallelIterator::pipeline()` function.
        //
        // Lastly, materializing this mutable reference on any thread is sound given
        // that `T` is `Send`. Indeed, this amounts to sending a `&mut T` across
        // threads, and `&mut T` is `Send` if and only if `T` is `Send`. This is the
        // rationale for why `MutPtrWrapper` implements `Sync` when `T` is `Send`.
        let item: &mut T = unsafe { &mut *item_ptr };
        item
    }
}

/// A helper struct for the implementation of [`MutSliceParallelSource`], that
/// wraps a [`*mut T`](pointer). This enables sending [`&mut T`](reference)
/// derived from a [`&mut [T]`](slice) to other threads.
struct MutPtrWrapper<T>(*mut T);
impl<T> MutPtrWrapper<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

/// SAFETY:
///
/// A [`MutPtrWrapper`] is meant to be shared among threads as a way to send
/// items of type [`&mut T`](reference) to other threads (see the safety
/// comments in [`MutSliceSourceDescriptor::exact_fetch_item`]). Therefore we
/// make it [`Sync`] if and only if [`&mut T`](reference) is [`Send`], which is
/// when `T` is [`Send`].
unsafe impl<T: Send> Sync for MutPtrWrapper<T> {}
