// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::ParallelIterator;
use crate::ThreadPool;

pub mod slice;

/// Trait for converting into a [`ParallelIterator`] on a [`ThreadPool`].
pub trait IntoParallelIterator {
    /// The type of items that this parallel iterator produces.
    ///
    /// Items are sent to worker threads (where they are then consumed by the
    /// `process_item` function parameter of the
    /// [`ParallelIterator::pipeline()`]), hence the required [`Send`] bound.
    type Item: Send;

    /// Target parallel iterator type.
    type Iter<'pool, 'scope: 'pool>: ParallelIterator<Item = Self::Item>;

    /// Converts `self` into a parallel iterator to be processed on the given
    /// thread pool.
    fn into_par_iter<'pool, 'scope: 'pool>(
        self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope>;
}

/// Trait for converting into a [`ParallelIterator`] that produces references.
///
/// This trait is automatically implemented for `T` where [`&T`](reference)
/// implements [`IntoParallelIterator`].
pub trait IntoParallelRefIterator<'data> {
    /// The type of items that this parallel iterator produces.
    ///
    /// Like for [`IntoParallelIterator`], items are sent to worker threads
    /// (where they are then consumed by the `process_item` function
    /// parameter of the [`ParallelIterator::pipeline()`]), hence the required
    /// [`Send`] bound.
    type Item: Send;

    /// Target parallel iterator type.
    type Iter<'pool, 'scope: 'pool>: ParallelIterator
    where
        Self: 'data;

    /// Converts `&self` into a parallel iterator to be processed on the given
    /// thread pool.
    ///
    /// ```rust
    /// # use paralight::iter::{IntoParallelRefIterator, ParallelIteratorExt};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter(&mut thread_pool)
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn par_iter<'pool, 'scope: 'pool>(
        &'data self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope>;
}

impl<'data, T: 'data + ?Sized> IntoParallelRefIterator<'data> for T
where
    &'data T: IntoParallelIterator,
{
    type Item = <&'data T as IntoParallelIterator>::Item;
    type Iter<'pool, 'scope: 'pool> = <&'data T as IntoParallelIterator>::Iter<'pool, 'scope>;

    fn par_iter<'pool, 'scope: 'pool>(
        &'data self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope> {
        self.into_par_iter(thread_pool)
    }
}

/// Trait for converting into a [`ParallelIterator`] that produces mutable
/// references.
///
/// This trait is automatically implemented for `T` where [`&mut T`](reference)
/// implements [`IntoParallelIterator`].
pub trait IntoParallelRefMutIterator<'data> {
    /// The type of items that this parallel iterator produces.
    ///
    /// Like for [`IntoParallelIterator`], items are sent to worker threads
    /// (where they are then consumed by the `process_item` function
    /// parameter of the [`ParallelIterator::pipeline()`]), hence the required
    /// [`Send`] bound.
    type Item: Send;

    /// Target parallel iterator type.
    type Iter<'pool, 'scope: 'pool>: ParallelIterator
    where
        Self: 'data;

    /// Converts `&mut self` into a parallel iterator to be processed on the
    /// given thread pool.
    ///
    /// ```rust
    /// # use paralight::iter::{IntoParallelRefMutIterator, ParallelIteratorExt};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let mut values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = values.par_iter_mut(&mut thread_pool).for_each(|x| {
    ///     *x *= 2;
    /// });
    /// assert_eq!(values, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    /// # });
    /// ```
    fn par_iter_mut<'pool, 'scope: 'pool>(
        &'data mut self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope>;
}

impl<'data, T: 'data + ?Sized> IntoParallelRefMutIterator<'data> for T
where
    &'data mut T: IntoParallelIterator,
{
    type Item = <&'data mut T as IntoParallelIterator>::Item;
    type Iter<'pool, 'scope: 'pool> = <&'data mut T as IntoParallelIterator>::Iter<'pool, 'scope>;

    fn par_iter_mut<'pool, 'scope: 'pool>(
        &'data mut self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope> {
        self.into_par_iter(thread_pool)
    }
}
