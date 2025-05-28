// Copyright 2024-2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{
    IntoParallelRefArrayChunks, IntoParallelRefMutArrayChunks, IntoParallelRefMutSource,
    IntoParallelRefSource, ParallelSource, SourceCleanup, SourceDescriptor,
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
/// # use paralight::iter::{
/// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, SliceParallelSource,
/// # };
/// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
        &self.slice[index]
    }
}

/// A parallel source over a [slice](slice) producing array chunks. This struct
/// is created by the
/// [`par_array_chunks_exact()`](IntoParallelRefArrayChunks::par_array_chunks_exact)
/// method on [`IntoParallelRefArrayChunks`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
///
/// See also [`MutSliceArrayChunkParallelSource`].
///
/// ```
/// # use paralight::iter::{
/// #     IntoParallelRefArrayChunks, ParallelIteratorExt, ParallelSourceExt,
/// #     SliceArrayChunkParallelSource,
/// # };
/// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let iter: SliceArrayChunkParallelSource<_, 2> = input.par_array_chunks_exact::<2>();
/// let [sum_odd, sum_even] = iter
///     .with_thread_pool(&mut thread_pool)
///     .copied()
///     .reduce(|| [0, 0], |[a0, a1], [b0, b1]| [a0 + b0, a1 + b1]);
/// assert_eq!(sum_odd, 5 * 5);
/// assert_eq!(sum_even, 5 * 6);
/// ```
#[must_use = "iterator adaptors are lazy"]
pub struct SliceArrayChunkParallelSource<'data, T, const N: usize> {
    slice: &'data [T],
    _phantom: PhantomData<[(); N]>,
}

impl<'data, T: Sync + 'data> IntoParallelRefArrayChunks<'data> for [T] {
    type ArrayChunk<const N: usize> = &'data [T; N];
    type Source<const N: usize> = SliceArrayChunkParallelSource<'data, T, N>;

    fn par_array_chunks_exact<const N: usize>(&'data self) -> Self::Source<N> {
        assert_ne!(
            N, 0,
            "called par_array_chunks_exact() with a chunk size of zero"
        );
        assert_eq!(
            self.len() % N, 0,
            "called par_array_chunks_exact() with a chunk size that doesn't divide the slice length"
        );
        SliceArrayChunkParallelSource {
            slice: self,
            _phantom: PhantomData,
        }
    }
}

impl<'data, T: Sync, const N: usize> ParallelSource for SliceArrayChunkParallelSource<'data, T, N> {
    type Item = &'data [T; N];

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        SliceArrayChunkSourceDescriptor {
            slice: self.slice,
            _phantom: PhantomData,
        }
    }
}

struct SliceArrayChunkSourceDescriptor<'data, T: Sync, const N: usize> {
    slice: &'data [T],
    _phantom: PhantomData<[(); N]>,
}

impl<T: Sync, const N: usize> SourceCleanup for SliceArrayChunkSourceDescriptor<'_, T, N> {
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {
        // Nothing to cleanup
    }
}

impl<'data, T: Sync, const N: usize> SourceDescriptor
    for SliceArrayChunkSourceDescriptor<'data, T, N>
{
    type Item = &'data [T; N];

    fn len(&self) -> usize {
        self.slice.len() / N
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        let slice: &'data [T] = &self.slice[index * N..(index + 1) * N];
        let item_ptr = slice.as_ptr() as *const [T; N];
        // SAFETY:
        //
        // From https://doc.rust-lang.org/std/ptr/index.html#pointer-to-reference-conversion:
        // - The `item_ptr` is properly aligned for `[T; N]` because it is aligned for
        //   `T`. In turn, this is because it is constructed by taking the pointer to a
        //   slice of `T`s and casting it to `*const [T; N]`.
        // - The `item_ptr` is not null, as it is obtained from a well-formed non-empty
        //   slice of length `N`.
        // - The `item_ptr` is dereferenceable, as the whole memory range of length `N *
        //   size_of::<T>()` starting from it is within bounds of a single allocated
        //   object (the input slice).
        // - The `item_ptr` points to a valid value of type `[T; N]`, the aggregate of
        //   `N` elements from the input slice starting at position `index`.
        // - The `item_ptr` follows the aliasing rules: while this reference exists
        //   (within this scope and in particular during the call to `process_item()`),
        //   the memory it points to isn't mutated. This is ensured by the safety
        //   pre-conditions of the `fetch_item()` function (each index must be passed at
        //   most once), and because the slice is a shared reference during the scope of
        //   this `ParallelIterator::pipeline()` function.
        unsafe { &*item_ptr }
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
/// # use paralight::iter::{
/// #     IntoParallelRefMutSource, MutSliceParallelSource, ParallelIteratorExt, ParallelSourceExt,
/// # };
/// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
        assert!(index < self.len);
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

/// A parallel source over a [mutable slice](slice) producing array chunks. This
/// struct is created by the
/// [`par_array_chunks_exact_mut()`](IntoParallelRefMutArrayChunks::par_array_chunks_exact_mut)
/// method on [`IntoParallelRefMutArrayChunks`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
///
/// See also [`SliceArrayChunkParallelSource`].
///
/// ```
/// # use paralight::iter::{
/// #     IntoParallelRefMutArrayChunks, MutSliceArrayChunkParallelSource, ParallelIteratorExt,
/// #     ParallelSourceExt,
/// # };
/// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let mut values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let iter: MutSliceArrayChunkParallelSource<_, 2> = values.par_array_chunks_exact_mut::<2>();
/// iter.with_thread_pool(&mut thread_pool).for_each(|[x, y]| {
///     *x *= 2;
///     *y *= 2;
/// });
/// assert_eq!(values, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
/// ```
#[must_use = "iterator adaptors are lazy"]
pub struct MutSliceArrayChunkParallelSource<'data, T, const N: usize> {
    slice: &'data mut [T],
    _phantom: PhantomData<[(); N]>,
}

impl<'data, T: Send + 'data> IntoParallelRefMutArrayChunks<'data> for [T] {
    type ArrayChunk<const N: usize> = &'data mut [T; N];
    type Source<const N: usize> = MutSliceArrayChunkParallelSource<'data, T, N>;

    fn par_array_chunks_exact_mut<const N: usize>(&'data mut self) -> Self::Source<N> {
        assert_ne!(
            N, 0,
            "called par_array_chunks_exact_mut() with a chunk size of zero"
        );
        assert_eq!(
            self.len() % N, 0,
            "called par_array_chunks_exact_mut() with a chunk size that doesn't divide the slice length"
        );
        MutSliceArrayChunkParallelSource {
            slice: self,
            _phantom: PhantomData,
        }
    }
}

impl<'data, T: Send, const N: usize> ParallelSource
    for MutSliceArrayChunkParallelSource<'data, T, N>
{
    type Item = &'data mut [T; N];

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let len = self.slice.len();
        let ptr = MutPtrWrapper(self.slice.as_mut_ptr());
        MutSliceArrayChunkSourceDescriptor {
            len,
            ptr,
            _phantom: PhantomData,
        }
    }
}

struct MutSliceArrayChunkSourceDescriptor<'data, T: Send + 'data, const N: usize> {
    len: usize,
    ptr: MutPtrWrapper<T>,
    _phantom: PhantomData<&'data [(); N]>,
}

impl<'data, T: Send + 'data, const N: usize> SourceCleanup
    for MutSliceArrayChunkSourceDescriptor<'data, T, N>
{
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {
        // Nothing to cleanup
    }
}

impl<'data, T: Send + 'data, const N: usize> SourceDescriptor
    for MutSliceArrayChunkSourceDescriptor<'data, T, N>
{
    type Item = &'data mut [T; N];

    fn len(&self) -> usize {
        self.len / N
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        assert!(index < self.len / N);
        let base_ptr: *mut T = self.ptr.get();
        // SAFETY:
        // - The offset in bytes `index * N * size_of::<T>()` fits in an `isize`,
        //   because the index is smaller than the length of the (well-formed) input
        //   slice divided by `N`. This is ensured by the safety pre-conditions of the
        //   `fetch_item()` function (the `index` must be in the range `0..self.len()`),
        //   and further confirmed by the assertion.
        // - The `base_ptr` is derived from an allocated object (the input slice), and
        //   the entire range between `base_ptr` and the resulting `array_start_ptr` is
        //   in bounds of that allocated object. This is because the index is smaller
        //   than the length of the input slice divided by `N`.
        let array_start_ptr: *mut T = unsafe { base_ptr.add(index * N) };
        let item_ptr: *mut [T; N] = array_start_ptr as _;
        // SAFETY:
        //
        // From https://doc.rust-lang.org/std/ptr/index.html#pointer-to-reference-conversion:
        // - The `item_ptr` is properly aligned for `[T; M]` because it is obtained from
        //   `array_start_ptr` which is properly aligned for `T`. In turn,
        //   `array_start_ptr` is properly aligned, as it is constructed by calling
        //   `add()` on the aligned `base_ptr`.
        // - The `item_ptr` is not null, as the `base_ptr` isn't null (obtained from a
        //   well-formed non-empty slice) and the `index` is in bounds of the slice (no
        //   wrap around).
        // - The `item_ptr` is dereferenceable, as the whole memory range of length `N *
        //   size_of::<T>()` starting from it is within bounds of a single allocated
        //   object (the input slice).
        // - The `item_ptr` points to a valid value of type `[T; N]`, the aggregate of
        //   `N` elements from the input slice starting at position `index * M`.
        // - The `item_ptr` follows the aliasing rules: while this reference exists
        //   (within this scope and in particular during the call to `process_item()`),
        //   the memory it points to isn't accessed through any other pointer or
        //   reference. This is ensured by the safety pre-conditions of the
        //   `fetch_item()` function (each index must be passed at most once), and
        //   because the slice is exclusively owned during the scope of this
        //   `ParallelIterator::pipeline()` function.
        //
        // Lastly, materializing this mutable reference on any thread is sound given
        // that `T` is `Send`. Indeed, this amounts to sending a `&mut [T; N]` across
        // threads, and `&mut [T; N]` is `Send` if and only if `[T; N]` is `Send`, i.e.
        // when `T` is `Send`. This is the rationale for why `MutPtrWrapper`
        // implements `Sync` when `T` is `Send`.
        let item: &mut [T; N] = unsafe { &mut *item_ptr };
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
