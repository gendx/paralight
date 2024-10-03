// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Iterator adaptors to define parallel pipelines more conveniently.

use crate::ThreadPool;

/// Trait for converting into a parallel iterator on a [`ThreadPool`].
pub trait IntoParallelIterator {
    /// Target parallel iterator type.
    type ParIter<'pool, 'scope: 'pool>: ParallelIterator;

    /// Converts `self` into a parallel iterator.
    ///
    /// ```rust
    /// # use paralight::iter::{IntoParallelIterator, ParallelIterator};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input.par_iter(&mut thread_pool).pipeline(
    ///     || 0u64,
    ///     |acc, _, x| *acc += *x,
    ///     |acc| acc,
    ///     |a, b| a + b,
    /// );
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn par_iter<'pool, 'scope: 'pool>(
        self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::ParIter<'pool, 'scope>;
}

/// An iterator to process items in parallel.
pub trait ParallelIterator: Sized {
    /// The type of items that this parallel iterator produces.
    type Item: Sync;

    /// Runs the pipeline defined by the given functions on this iterator.
    ///
    /// # Parameters
    ///
    /// - `init` function to create a new (per-thread) accumulator,
    /// - `process_item` function to accumulate an item from the slice into the
    ///   accumulator,
    /// - `finalize` function to transform an accumulator into an output,
    /// - `reduce` function to reduce a pair of outputs into one output.
    ///
    /// ```rust
    /// # use paralight::iter::{IntoParallelIterator, ParallelIterator};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input.par_iter(&mut thread_pool).pipeline(
    ///     || 0u64,
    ///     |acc, _, x| *acc += *x,
    ///     |acc| acc,
    ///     |a, b| a + b,
    /// );
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(&mut Accum, usize, &Self::Item) + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output;
}

/// A parallel iterator over a slice.
#[must_use = "iterator adaptors are lazy"]
pub struct SliceParallelIterator<'pool, 'scope: 'pool, 'data, T> {
    thread_pool: &'pool mut ThreadPool<'scope>,
    slice: &'data [T],
}

impl<'data, T: Sync> IntoParallelIterator for &'data [T] {
    type ParIter<'pool, 'scope: 'pool> = SliceParallelIterator<'pool, 'scope, 'data, T>;

    fn par_iter<'pool, 'scope: 'pool>(
        self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::ParIter<'pool, 'scope> {
        SliceParallelIterator {
            thread_pool,
            slice: self,
        }
    }
}

impl<'pool, 'scope: 'pool, 'data, T: Sync> ParallelIterator
    for SliceParallelIterator<'pool, 'scope, 'data, T>
{
    type Item = T;

    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(&mut Accum, usize, &Self::Item) + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.thread_pool
            .pipeline(self.slice, init, process_item, finalize, reduce)
    }
}
