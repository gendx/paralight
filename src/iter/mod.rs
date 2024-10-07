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
    /// # use paralight::iter::{IntoParallelIterator, ParallelIteratorExt};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter(&mut thread_pool)
    ///     .map(|&x| x)
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn par_iter<'pool, 'scope: 'pool>(
        self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::ParIter<'pool, 'scope>;
}

/// An iterator to process items in parallel. The [`ParallelIteratorExt`] trait
/// provides additional methods (iterator adaptors) as an extension of this
/// trait.
pub trait ParallelIterator: Sized {
    /// The type of items that this parallel iterator produces.
    type Item;

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
    /// # use paralight::iter::{IntoParallelIterator, ParallelIteratorExt};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter(&mut thread_pool)
    ///     .map(|&x| x)
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> Accum + Sync,
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

/// Additional methods provided for types that implement [`ParallelIterator`].
pub trait ParallelIteratorExt: ParallelIterator {
    /// Returns a parallel iterator that yields only the items for which the
    /// predicate `f` returns `true`.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelIterator, ParallelIteratorExt};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum_even = input
    ///     .par_iter(&mut thread_pool)
    ///     .filter(|&&x| x % 2 == 0)
    ///     .map(|&x| x)
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum_even, 5 * 6);
    /// # });
    /// ```
    fn filter<F>(self, f: F) -> Filter<Self, F>
    where
        F: Fn(&Self::Item) -> bool + Sync,
    {
        Filter { inner: self, f }
    }

    /// Runs `f` on each item of this parallel iterator.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelIterator, ParallelIteratorExt};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::collections::HashSet;
    /// # use std::num::NonZeroUsize;
    /// # use std::sync::Mutex;
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let set = Mutex::new(HashSet::new());
    /// input.par_iter(&mut thread_pool).for_each(|&x| {
    ///     set.lock().unwrap().insert(x);
    /// });
    /// assert_eq!(set.into_inner().unwrap(), (1..=10).collect());
    /// # });
    /// ```
    fn for_each<F>(self, f: F)
    where
        F: Fn(Self::Item) + Sync,
    {
        self.pipeline(
            /* init */ || (),
            /* process_item */ |(), _index, item| f(item),
            /* finalize */ |()| (),
            /* reduce */ |(), ()| (),
        )
    }

    /// Applies the function `f` to each item of this iterator, returning a
    /// parallel iterator with the mapped items.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelIterator, ParallelIteratorExt};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let double_sum = input
    ///     .par_iter(&mut thread_pool)
    ///     .map(|&x| x * 2)
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(double_sum, 10 * 11);
    /// # });
    /// ```
    ///
    /// Mapping to a non-[`Send`] non-[`Sync`] type such as [`Rc`](std::rc::Rc)
    /// is fine.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelIterator, ParallelIterator, ParallelIteratorExt};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # use std::rc::Rc;
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter(&mut thread_pool)
    ///     .map(|&x| Rc::new(x))
    ///     .pipeline(|| 0u64, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn map<T, F>(self, f: F) -> Map<Self, F>
    where
        F: Fn(Self::Item) -> T + Sync,
    {
        Map { inner: self, f }
    }

    /// Reduces the items produced by this iterator into a single item, using
    /// `f` to collapse pairs of items.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelIterator, ParallelIteratorExt};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter(&mut thread_pool)
    ///     .map(|&x| x)
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    ///
    /// Under the hood, each worker thread may call `init` and `f` in arbitrary
    /// order, therefore this method is mainly useful under the following
    /// conditions.
    /// - The reduction function `f` should be both [commutative](https://en.wikipedia.org/wiki/Commutative_property)
    ///   and [associative](https://en.wikipedia.org/wiki/Associative_property),
    ///   otherwise the result isn't deterministic. For example, addition on
    ///   floating-point numbers (e.g. [`f32`], [`f64`]) is commutative but not
    ///   associative.
    /// - The `init` function should return a neutral (a.k.a. identity) element
    ///   with respect to `f`, that may be inserted anywhere in the computation.
    ///   In particular, `reduce()` returns `init()` if the iterator is empty.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelIterator, ParallelIteratorExt};
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let sum = []
    ///     .par_iter(&mut thread_pool)
    ///     .map(|&x| x)
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 0);
    /// # });
    /// ```
    fn reduce<Init, F>(self, init: Init, f: F) -> Self::Item
    where
        Init: Fn() -> Self::Item + Sync,
        F: Fn(Self::Item, Self::Item) -> Self::Item + Sync,
        Self::Item: Send,
    {
        self.pipeline(init, |acc, _index, item| f(acc, item), |acc| acc, &f)
    }
}

impl<T: ParallelIterator> ParallelIteratorExt for T {}

/// This struct is created by the [`filter()`](ParallelIteratorExt::filter)
/// method on [`ParallelIteratorExt`].
#[must_use = "iterator adaptors are lazy"]
pub struct Filter<Inner: ParallelIterator, F> {
    inner: Inner,
    f: F,
}

impl<Inner: ParallelIterator, F> ParallelIterator for Filter<Inner, F>
where
    F: Fn(&Inner::Item) -> bool + Sync,
{
    type Item = Inner::Item;

    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> Accum + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.inner.pipeline(
            init,
            |accum, index, item| {
                if (self.f)(&item) {
                    process_item(accum, index, item)
                } else {
                    accum
                }
            },
            finalize,
            reduce,
        )
    }
}

/// This struct is created by the [`map()`](ParallelIteratorExt::map) method on
/// [`ParallelIteratorExt`].
#[must_use = "iterator adaptors are lazy"]
pub struct Map<Inner: ParallelIterator, F> {
    inner: Inner,
    f: F,
}

impl<Inner: ParallelIterator, T, F> ParallelIterator for Map<Inner, F>
where
    F: Fn(Inner::Item) -> T + Sync,
{
    type Item = T;

    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> Accum + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.inner.pipeline(
            init,
            |accum, index, item| process_item(accum, index, (self.f)(item)),
            finalize,
            reduce,
        )
    }
}
