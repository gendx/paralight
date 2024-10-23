// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::ParallelIterator;
use crate::ThreadPool;

/// Trait for converting into a [`ParallelIterator`] on a [`ThreadPool`].
pub trait IntoParallelIterator {
    /// The type of items that this parallel iterator produces.
    type Item;

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
    type Item;

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

/// A parallel iterator over a slice. This struct is created by the
/// [`par_iter()`](IntoParallelRefIterator::par_iter) method on
/// [`IntoParallelRefIterator`].
#[must_use = "iterator adaptors are lazy"]
pub struct SliceParallelIterator<'pool, 'scope: 'pool, 'data, T> {
    thread_pool: &'pool mut ThreadPool<'scope>,
    slice: &'data [T],
}

impl<'data, T: Sync> IntoParallelIterator for &'data [T] {
    type Item = &'data T;
    type Iter<'pool, 'scope: 'pool> = SliceParallelIterator<'pool, 'scope, 'data, T>;

    fn into_par_iter<'pool, 'scope: 'pool>(
        self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope> {
        SliceParallelIterator {
            thread_pool,
            slice: self,
        }
    }
}

impl<'pool, 'scope: 'pool, 'data, T: Sync> ParallelIterator
    for SliceParallelIterator<'pool, 'scope, 'data, T>
{
    type Item = &'data T;

    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> Accum + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.thread_pool
            .pipeline(self.slice, init, process_item, finalize, reduce)
    }
}
