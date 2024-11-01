// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

pub mod slice;
pub mod zip;

/// An object describing how to fetch items from a [`ParallelSource`].
pub struct SourceDescriptor<Item: Send, FetchItem: Fn(usize) -> Item + Sync> {
    /// Number of items that the source produces.
    pub len: usize,
    /// A function to fetch the item at the given index.
    pub fetch_item: FetchItem,
}

/// A source to produce items in parallel.
///
/// This can be turned into a [`ParallelIterator`](super::ParallelIterator) by
/// attaching a [`ThreadPool`](crate::ThreadPool) via the
/// [`with_thread_pool()`](super::WithThreadPool::with_thread_pool) function.
pub trait ParallelSource: Sized {
    /// The type of items that this parallel source produces.
    ///
    /// Items are sent to worker threads (where they are then consumed by the
    /// `process_item` function parameter of the
    /// [`ParallelIterator::pipeline()`](super::ParallelIterator::pipeline)),
    /// hence the required [`Send`] bound.
    type Item: Send;

    /// Returns an object that describes how to fetch items from this source.
    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync>;
}

/// Trait for converting into a [`ParallelSource`].
pub trait IntoParallelSource {
    /// The type of items that this parallel source produces.
    ///
    /// Items are sent to worker threads (where they are then consumed by the
    /// `process_item` function parameter of the
    /// [`ParallelIterator::pipeline()`](super::ParallelIterator::pipeline)),
    /// hence the required [`Send`] bound.
    type Item: Send;

    /// Target parallel source type.
    type Source: ParallelSource<Item = Self::Item>;

    /// Converts `self` into a parallel source.
    fn into_par_iter(self) -> Self::Source;
}

/// Trait for converting into a [`ParallelSource`] that produces references.
///
/// This trait is automatically implemented for `T` where [`&T`](reference)
/// implements [`IntoParallelSource`].
pub trait IntoParallelRefSource<'data> {
    /// The type of items that this parallel source produces.
    ///
    /// Like for [`IntoParallelSource`], items are sent to worker threads (where
    /// they are then consumed by the `process_item` function parameter of the
    /// [`ParallelIterator::pipeline()`](super::ParallelIterator::pipeline)),
    /// hence the required [`Send`] bound.
    type Item: Send;

    /// Target parallel source type.
    type Source: ParallelSource<Item = Self::Item>;

    /// Converts `&self` into a parallel source to be processed on the given
    /// thread pool.
    ///
    /// ```rust
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn par_iter(&'data self) -> Self::Source;
}

impl<'data, T: 'data + ?Sized> IntoParallelRefSource<'data> for T
where
    &'data T: IntoParallelSource,
{
    type Item = <&'data T as IntoParallelSource>::Item;
    type Source = <&'data T as IntoParallelSource>::Source;

    fn par_iter(&'data self) -> Self::Source {
        self.into_par_iter()
    }
}

/// Trait for converting into a [`ParallelSource`] that produces mutable
/// references.
///
/// This trait is automatically implemented for `T` where [`&mut T`](reference)
/// implements [`IntoParallelSource`].
pub trait IntoParallelRefMutSource<'data> {
    /// The type of items that this parallel source produces.
    ///
    /// Like for [`IntoParallelSource`], items are sent to worker threads (where
    /// they are then consumed by the `process_item` function parameter of the
    /// [`ParallelIterator::pipeline()`](super::ParallelIterator::pipeline)),
    /// hence the required [`Send`] bound.
    type Item: Send;

    /// Target parallel source type.
    type Source: ParallelSource<Item = Self::Item>;

    /// Converts `&mut self` into a parallel source to be processed on the given
    /// thread pool.
    ///
    /// ```rust
    /// # use paralight::iter::{IntoParallelRefMutSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let mut values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = values
    ///     .par_iter_mut()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each(|x| {
    ///         *x *= 2;
    ///     });
    /// assert_eq!(values, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    /// # });
    /// ```
    fn par_iter_mut(&'data mut self) -> Self::Source;
}

impl<'data, T: 'data + ?Sized> IntoParallelRefMutSource<'data> for T
where
    &'data mut T: IntoParallelSource,
{
    type Item = <&'data mut T as IntoParallelSource>::Item;
    type Source = <&'data mut T as IntoParallelSource>::Source;

    fn par_iter_mut(&'data mut self) -> Self::Source {
        self.into_par_iter()
    }
}
