// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Multi-dimensional parallel sources.
//!
//! ```
//! # use paralight::iter::{
//! #     MultiDimParallelSourceExt, MultiDimZipableSource, MutSlice2dParallelSource,
//! #     ParallelIteratorExt, ParallelSourceExt, Slice2dParallelSource,
//! # };
//! # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
//! # let mut thread_pool = ThreadPoolBuilder {
//! #     num_threads: ThreadCount::AvailableParallelism,
//! #     range_strategy: RangeStrategy::WorkStealing,
//! #     cpu_pinning: CpuPinningPolicy::No,
//! # }
//! # .build();
//! use std::array::from_fn;
//! let x: [[usize; 10]; 20] = from_fn(|i| from_fn(|j| i + j));
//! let y: [[usize; 10]; 20] = from_fn(|i| from_fn(|j| i * j));
//! let mut z: [[usize; 10]; 20] = [[0; 10]; 20];
//!
//! let parx = Slice2dParallelSource { slice: &x };
//! let pary = Slice2dParallelSource { slice: &y };
//! let parz = MutSlice2dParallelSource { slice: &mut z };
//! (parx, pary, parz)
//!     .ndim_zip_eq()
//!     .flatten()
//!     .with_thread_pool(&mut thread_pool)
//!     .for_each(|(x, y, z)| *z = *x + *y);
//!
//! assert_eq!(z, from_fn(|i| from_fn(|j| i + j + i * j)));
//! ```

mod slice;
mod zip;

use super::{ParallelSource, SourceCleanup, SourceDescriptor};
pub use slice::{
    MutSlice1dParallelSource, MutSlice2dParallelSource, MutSlice3dParallelSource,
    Slice1dParallelSource, Slice2dParallelSource, Slice3dParallelSource,
};
use std::marker::PhantomData;
pub use zip::{MultiDimZipEq, MultiDimZipableSource};

/// TODO
#[allow(clippy::len_without_is_empty)]
pub trait MultiDimSourceDescriptor<const DIM: usize>: SourceCleanup {
    /// The type of items that this parallel source produces.
    type Item: Send;

    /// Returns the number of items that this source produces in each dimension.
    fn len(&self) -> [usize; DIM];

    /// Fetch the item at the given multi-dimensional index.
    ///
    /// # Safety
    ///
    /// TODO
    unsafe fn fetch_item(&self, index: [usize; DIM]) -> Self::Item;
}

/// TODO
pub trait MultiDimParallelSource<const DIM: usize>: Sized {
    /// TODO
    type Item: Send;

    /// TODO
    fn descriptor(self) -> impl MultiDimSourceDescriptor<DIM, Item = Self::Item> + Sync;
}

/// TODO
pub trait MultiDimParallelSourceExt<const DIM: usize>: MultiDimParallelSource<DIM> {
    /// TODO
    fn flatten(self) -> Flattened<DIM, Self> {
        Flattened {
            inner: self,
            _phantom: PhantomData,
        }
    }

    /// TODO
    ///
    /// ```
    /// # use paralight::iter::{
    /// #     MultiDimParallelSourceExt, MultiDimZipableSource, MutSlice2dParallelSource,
    /// #     ParallelIteratorExt, ParallelSourceExt, Slice2dParallelSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// use std::array::from_fn;
    /// let x: [[usize; 5]; 10] = from_fn(|i| from_fn(|j| i + j));
    /// let y: [[usize; 10]; 5] = from_fn(|i| from_fn(|j| i + j));
    /// let mut z: [[usize; 5]; 10] = [[0; 5]; 10];
    ///
    /// let parx = Slice2dParallelSource { slice: &x };
    /// let pary = Slice2dParallelSource { slice: &y };
    /// let parz = MutSlice2dParallelSource { slice: &mut z };
    /// (parx, pary.permute([1, 0]), parz)
    ///     .ndim_zip_eq()
    ///     .flatten()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each(|(x, y, z)| *z = *x + *y);
    ///
    /// assert_eq!(z, from_fn(|i| from_fn(|j| 2 * (i + j))));
    /// ```
    ///
    /// ```
    /// # use paralight::iter::{
    /// #     MultiDimParallelSourceExt, MultiDimZipableSource, MutSlice3dParallelSource,
    /// #     ParallelIteratorExt, ParallelSourceExt, Slice3dParallelSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// use std::array::from_fn;
    /// let x: [[[usize; 5]; 10]; 15] = from_fn(|i| from_fn(|j| from_fn(|k| i + j + k)));
    /// let y: [[[usize; 10]; 15]; 5] = from_fn(|i| from_fn(|j| from_fn(|k| i + j + k)));
    /// let mut z: [[[usize; 5]; 10]; 15] = [[[0; 5]; 10]; 15];
    ///
    /// // multi-dimensional length = [15, 10, 5]
    /// let parx = Slice3dParallelSource { slice: &x };
    /// // multi-dimensional length = [5, 15, 10]
    /// let pary = Slice3dParallelSource { slice: &y };
    /// // multi-dimensional length = [15, 10, 5]
    /// let parz = MutSlice3dParallelSource { slice: &mut z };
    /// (parx, pary.permute([1, 2, 0]), parz)
    ///     .ndim_zip_eq()
    ///     .flatten()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each(|(x, y, z)| *z = *x + *y);
    ///
    /// assert_eq!(z, from_fn(|i| from_fn(|j| from_fn(|k| 2 * (i + j + k)))));
    /// ```
    fn permute(self, permutation: [usize; DIM]) -> Permuted<DIM, Self> {
        let mut used = [false; DIM];
        for i in permutation {
            if used[i] {
                panic!(
                    "called permute() with an argument that isn't a permutation ({permutation:?})"
                );
            }
            used[i] = true;
        }
        let mut inv_permutation = [0; DIM];
        for i in 0..DIM {
            inv_permutation[permutation[i]] = i;
        }
        Permuted {
            inner: self,
            permutation,
            inv_permutation,
        }
    }
}

impl<const DIM: usize, T: MultiDimParallelSource<DIM>> MultiDimParallelSourceExt<DIM> for T {}

#[must_use = "iterator adaptors are lazy"]
pub struct Flattened<const DIM: usize, T> {
    inner: T,
    _phantom: PhantomData<[(); DIM]>,
}

impl<const DIM: usize, T: MultiDimParallelSource<DIM>> ParallelSource for Flattened<DIM, T> {
    type Item = T::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.descriptor();
        let ndim_len = descriptor.len();
        let len = ndim_len.iter().product();
        FlattenedSourceDescriptor {
            inner: descriptor,
            len,
            ndim_len,
        }
    }
}

struct FlattenedSourceDescriptor<const DIM: usize, Inner: MultiDimSourceDescriptor<DIM>> {
    inner: Inner,
    len: usize,
    ndim_len: [usize; DIM],
}

impl<const DIM: usize, Inner: MultiDimSourceDescriptor<DIM>> SourceCleanup
    for FlattenedSourceDescriptor<DIM, Inner>
{
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // SAFETY: TODO
            unsafe {
                self.inner.cleanup_item_range(range);
            }
        }
    }
}

impl<const DIM: usize, Inner: MultiDimSourceDescriptor<DIM>> SourceDescriptor
    for FlattenedSourceDescriptor<DIM, Inner>
{
    type Item = Inner::Item;

    fn len(&self) -> usize {
        self.len
    }

    // TODO: Add a more efficient fetch_next_item() function.
    unsafe fn fetch_item(&self, mut index: usize) -> Self::Item {
        let mut multi_dim_index = [0; DIM];
        // Scan the length backwards because we want the last component of the index to
        // increase the fastest.
        for i in (0..DIM).rev() {
            let ilen = self.ndim_len[i];
            multi_dim_index[i] = index % ilen;
            index /= ilen;
        }
        // SAFETY: TODO
        unsafe { self.inner.fetch_item(multi_dim_index) }
    }
}

#[must_use = "iterator adaptors are lazy"]
pub struct Permuted<const DIM: usize, T> {
    inner: T,
    permutation: [usize; DIM],
    inv_permutation: [usize; DIM],
}

impl<const DIM: usize, T: MultiDimParallelSource<DIM>> MultiDimParallelSource<DIM>
    for Permuted<DIM, T>
{
    type Item = T::Item;

    fn descriptor(self) -> impl MultiDimSourceDescriptor<DIM, Item = Self::Item> + Sync {
        PermutedSourceDescriptor {
            inner: self.inner.descriptor(),
            permutation: self.permutation,
            inv_permutation: self.inv_permutation,
        }
    }
}

struct PermutedSourceDescriptor<const DIM: usize, Inner> {
    inner: Inner,
    permutation: [usize; DIM],
    inv_permutation: [usize; DIM],
}

impl<const DIM: usize, Inner: MultiDimSourceDescriptor<DIM>> SourceCleanup
    for PermutedSourceDescriptor<DIM, Inner>
{
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // TODO
            todo!()
        }
    }
}

impl<const DIM: usize, Inner: MultiDimSourceDescriptor<DIM>> MultiDimSourceDescriptor<DIM>
    for PermutedSourceDescriptor<DIM, Inner>
{
    type Item = Inner::Item;

    fn len(&self) -> [usize; DIM] {
        let inner_len = self.inner.len();
        self.permutation.map(|i| inner_len[i])
    }

    unsafe fn fetch_item(&self, index: [usize; DIM]) -> Self::Item {
        let inner_index = self.inv_permutation.map(|i| index[i]);
        // SAFETY: TODO
        unsafe { self.inner.fetch_item(inner_index) }
    }
}
