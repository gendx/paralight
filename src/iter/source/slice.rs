// Copyright 2024-2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{
    IntoParallelRefMutSource, IntoParallelRefSource, ParallelSource, SourceCleanup,
    SourceDescriptor,
};
use std::marker::PhantomData;

/// A parallel source over a [slice](slice). This struct is created by the
/// [`par_iter()`](IntoParallelRefSource::par_iter) method on
/// [`IntoParallelRefSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
///
/// See also [`MutSliceParallelSource`].
///
/// ```
/// # use paralight::iter::SliceParallelSource;
/// # use paralight::prelude::*;
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let iter: SliceParallelSource<_> = input.par_iter();
/// let sum = iter.with_thread_pool(&mut thread_pool).sum::<i32>();
/// assert_eq!(sum, 5 * 11);
/// ```
#[must_use = "iterator adaptors are lazy"]
pub struct SliceParallelSource<'data, T> {
    slice: &'data [T],
}

impl<'data, T: Sync + 'data> IntoParallelRefSource<'data> for [T] {
    type Item = &'data T;
    type Source = SliceParallelSource<'data, T>;

    fn par_iter(&'data self) -> Self::Source {
        SliceParallelSource { slice: self }
    }
}

impl<'data, T: Sync> ParallelSource for SliceParallelSource<'data, T> {
    type Item = &'data T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        SliceSourceDescriptor { slice: self.slice }
    }
}

struct SliceSourceDescriptor<'data, T: Sync> {
    slice: &'data [T],
}

impl<T: Sync> SourceCleanup for SliceSourceDescriptor<'_, T> {
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {
        // Nothing to cleanup
    }
}

impl<'data, T: Sync> SourceDescriptor for SliceSourceDescriptor<'data, T> {
    type Item = &'data T;

    fn len(&self) -> usize {
        self.slice.len()
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.slice.len());
        // SAFETY: The index is smaller than the length of the input slice, due to the
        // safety pre-conditions of the `fetch_item()` function.
        unsafe { self.slice.get_unchecked(index) }
    }
}

/// A parallel source over a [mutable slice](slice). This struct is created by
/// the [`par_iter_mut()`](IntoParallelRefMutSource::par_iter_mut) method on
/// [`IntoParallelRefMutSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
///
/// See also [`SliceParallelSource`].
///
/// ```
/// # use paralight::iter::MutSliceParallelSource;
/// # use paralight::prelude::*;
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let mut values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let iter: MutSliceParallelSource<_> = values.par_iter_mut();
/// iter.with_thread_pool(&mut thread_pool)
///     .for_each(|x| *x *= 2);
/// assert_eq!(values, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
/// ```
#[must_use = "iterator adaptors are lazy"]
pub struct MutSliceParallelSource<'data, T> {
    slice: &'data mut [T],
}

impl<'data, T: Send + 'data> IntoParallelRefMutSource<'data> for [T] {
    type Item = &'data mut T;
    type Source = MutSliceParallelSource<'data, T>;

    fn par_iter_mut(&'data mut self) -> Self::Source {
        MutSliceParallelSource { slice: self }
    }
}

impl<'data, T: Send> ParallelSource for MutSliceParallelSource<'data, T> {
    type Item = &'data mut T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
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

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {
        // Nothing to cleanup
    }
}

impl<'data, T: Send + 'data> SourceDescriptor for MutSliceSourceDescriptor<'data, T> {
    type Item = &'data mut T;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        let base_ptr: *mut T = self.ptr.get();
        // SAFETY:
        // - The offset in bytes `index * size_of::<T>()` fits in an `isize`, because
        //   the index is smaller than the length of the (well-formed) input slice. This
        //   is ensured by the safety pre-conditions of the `fetch_item()` function (the
        //   `index` must be in the range `0..self.len`), and further confirmed by the
        //   assertion.
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
        //   `fetch_item()` function (each index must be passed at most once), and
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
/// comments in [`MutSliceSourceDescriptor::fetch_item`]). Therefore we make it
/// [`Sync`] if and only if [`&mut T`](reference) is [`Send`], which is when `T`
/// is [`Send`].
unsafe impl<T: Send> Sync for MutPtrWrapper<T> {}
