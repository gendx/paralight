// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Iterator adaptors to define parallel pipelines more conveniently.

mod source;

use crate::ThreadPool;
pub use source::slice::{MutSliceParallelSource, SliceParallelSource};
pub use source::zip::{ZipEq, ZipMax, ZipMin, ZipableSource};
pub use source::{
    IntoParallelRefMutSource, IntoParallelRefSource, IntoParallelSource, ParallelSource,
    SourceDescriptor,
};
use std::cmp::Ordering;

/// Trait to attach a [`ThreadPool`] to a [`ParallelSource`] and obtain a
/// [`ParallelIterator`].
pub trait WithThreadPool: ParallelSource {
    /// Attaches the given [`ThreadPool`] to this [`ParallelSource`].
    fn with_thread_pool<'pool, 'scope: 'pool>(
        self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> BaseParallelIterator<'pool, 'scope, Self> {
        BaseParallelIterator {
            thread_pool,
            source: self,
        }
    }
}

impl<S: ParallelSource> WithThreadPool for S {}

/// This struct is created by the
/// [`with_thread_pool()`](WithThreadPool::with_thread_pool) method on
/// [`WithThreadPool`].
#[must_use = "iterator adaptors are lazy"]
pub struct BaseParallelIterator<'pool, 'scope: 'pool, S: ParallelSource> {
    thread_pool: &'pool mut ThreadPool<'scope>,
    source: S,
}

impl<'pool, 'scope: 'pool, S: ParallelSource> ParallelIterator
    for BaseParallelIterator<'pool, 'scope, S>
{
    type Item = S::Item;

    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> Accum + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        let source_descriptor = self.source.descriptor();
        self.thread_pool.pipeline(
            source_descriptor.len,
            init,
            |acc, index| process_item(acc, index, (source_descriptor.fetch_item)(index)),
            finalize,
            reduce,
        )
    }
}

/// An iterator to process items in parallel. The [`ParallelIteratorExt`] trait
/// provides additional methods (iterator adaptors) as an extension of this
/// trait.
pub trait ParallelIterator: Sized {
    /// The type of items that this parallel iterator produces.
    ///
    /// Note that this type has no particular [`Send`] nor [`Sync`] bounds, as
    /// items may be created locally on a worker thread, for example via the
    /// [`map()`](ParallelIteratorExt::map) adaptor. However, initial
    /// sources of parallel iterators require the items to be [`Send`],
    /// via the [`IntoParallelSource`] trait.
    type Item;

    /// Runs the pipeline defined by the given functions on this iterator.
    ///
    /// # Parameters
    ///
    /// - `init` function to create a new (per-thread) accumulator,
    /// - `process_item` function to accumulate an item into the accumulator,
    /// - `finalize` function to transform an accumulator into an output,
    /// - `reduce` function to reduce a pair of outputs into one output.
    ///
    /// ```rust
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIterator, WithThreadPool};
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
    ///     .pipeline(|| 0, |acc, _, x| acc + x, |acc| acc, |x, y| x + y);
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

/// Additional methods provided for types that implement [`ParallelIterator`].
pub trait ParallelIteratorExt: ParallelIterator {
    /// Returns a parallel iterator that produces items that are cloned from the
    /// items of this iterator. This is useful if you have an iterator over
    /// [`&T`](reference) and want an iterator over `T`, when `T` is
    /// [`Clone`].
    ///
    /// This is equivalent to calling `.map(|x| x.clone())`.
    ///
    /// See also [`copied()`](Self::copied).
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(Box::new);
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .cloned()
    ///     .reduce(
    ///         || Box::new(0),
    ///         |mut x, y| {
    ///             *x += *y;
    ///             x
    ///         },
    ///     );
    /// assert_eq!(*sum, 5 * 11);
    /// # });
    /// ```
    fn cloned<'a, T>(self) -> Cloned<Self>
    where
        T: Clone + 'a,
        Self: ParallelIterator<Item = &'a T>,
    {
        Cloned { inner: self }
    }

    /// Returns a parallel iterator that produces items that are copied from the
    /// items of this iterator. This is useful if you have an iterator over
    /// [`&T`](reference) and want an iterator over `T`, when `T` is
    /// [`Copy`].
    ///
    /// This is equivalent to calling `.map(|x| *x)`.
    ///
    /// See also [`cloned()`](Self::cloned).
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn copied<'a, T>(self) -> Copied<Self>
    where
        T: Copy + 'a,
        Self: ParallelIterator<Item = &'a T>,
    {
        Copied { inner: self }
    }

    /// Returns a parallel iterator that produces only the items for which the
    /// predicate `f` returns `true`.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum_even = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .filter(|&&x| x % 2 == 0)
    ///     .copied()
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

    /// Applies the function `f` to each item of this iterator, returning a
    /// parallel iterator that produces the mapped items `x` for which `f`
    /// returns `Some(x)` and skips the items for which `f` returns `None`.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .filter_map(|&x| if x != 2 { Some(x * 3) } else { None })
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 3 * (5 * 11 - 2));
    /// # });
    /// ```
    ///
    /// Mapping to a non-[`Send`] non-[`Sync`] type such as [`Rc`](std::rc::Rc)
    /// is fine.
    ///
    /// ```
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIterator, ParallelIteratorExt, WithThreadPool,
    /// # };
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # use std::rc::Rc;
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum_even = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .filter_map(|&x| if x % 2 == 0 { Some(Rc::new(x)) } else { None })
    ///     .pipeline(|| 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);
    /// assert_eq!(sum_even, 5 * 6);
    /// # });
    /// ```
    fn filter_map<T, F>(self, f: F) -> FilterMap<Self, F>
    where
        F: Fn(Self::Item) -> Option<T> + Sync,
    {
        FilterMap { inner: self, f }
    }

    /// Runs `f` on each item of this parallel iterator.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let set = Mutex::new(HashSet::new());
    /// input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each(|&x| {
    ///         set.lock().unwrap().insert(x);
    ///     });
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

    /// Runs the function `f` on each item of this iterator in a pass-through
    /// manner, returning a parallel iterator producing the original items.
    ///
    /// This is useful to help debug intermediate stages of an iterator adaptor
    /// pipeline.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # use std::sync::atomic::{AtomicUsize, Ordering};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let inspections = AtomicUsize::new(0);
    ///
    /// let min = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .inspect(|x| {
    ///         println!("[{:?}] x = {x}", std::thread::current().id());
    ///         inspections.fetch_add(1, Ordering::Relaxed);
    ///     })
    ///     .min();
    ///
    /// assert_eq!(min, Some(&1));
    /// assert_eq!(inspections.load(Ordering::Relaxed), 10);
    /// # });
    /// ```
    fn inspect<F>(self, f: F) -> Inspect<Self, F>
    where
        F: Fn(&Self::Item) + Sync,
    {
        Inspect { inner: self, f }
    }

    /// Applies the function `f` to each item of this iterator, returning a
    /// parallel iterator producing the mapped items.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let double_sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIterator, ParallelIteratorExt, WithThreadPool,
    /// # };
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # use std::rc::Rc;
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .map(|&x| Rc::new(x))
    ///     .pipeline(|| 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn map<T, F>(self, f: F) -> Map<Self, F>
    where
        F: Fn(Self::Item) -> T + Sync,
    {
        Map { inner: self, f }
    }

    /// Returns the maximal item of this iterator, or [`None`] if this iterator
    /// is empty.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let max = input.par_iter().with_thread_pool(&mut thread_pool).max();
    /// assert_eq!(max, Some(&10));
    /// # });
    /// ```
    fn max(self) -> Option<Self::Item>
    where
        Self::Item: Ord + Send,
    {
        self.pipeline(
            || None,
            |max, _, x| match max {
                None => Some(x),
                Some(max) => Some(max.max(x)),
            },
            |max| max,
            |x, y| match (x, y) {
                (None, None) => None,
                (Some(a), None) | (None, Some(a)) => Some(a),
                (Some(a), Some(b)) => Some(a.max(b)),
            },
        )
    }

    /// Returns the maximal item of this iterator according to the comparison
    /// function `f`, or [`None`] if this iterator is empty.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// // Custom comparison function where even numbers are smaller than all odd numbers.
    /// let max = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .max_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
    /// assert_eq!(max, Some(&9));
    /// # });
    /// ```
    fn max_by<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> Ordering + Sync,
        Self::Item: Send,
    {
        self.pipeline(
            || None,
            |max, _, x| match max {
                None => Some(x),
                Some(max) => match f(&max, &x) {
                    Ordering::Greater | Ordering::Equal => Some(max),
                    Ordering::Less => Some(x),
                },
            },
            |max| max,
            |x, y| match (x, y) {
                (None, None) => None,
                (Some(a), None) | (None, Some(a)) => Some(a),
                (Some(a), Some(b)) => match f(&a, &b) {
                    Ordering::Greater | Ordering::Equal => Some(a),
                    Ordering::Less => Some(b),
                },
            },
        )
    }

    /// Returns the maximal item of this iterator according to the keys derived
    /// from the mapping function `f`, or [`None`] if this iterator is
    /// empty.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = ["ccc", "aaaaa", "dd", "e", "bbbb"];
    ///
    /// let max = input.par_iter().with_thread_pool(&mut thread_pool).max();
    /// assert_eq!(max, Some(&"e"));
    ///
    /// let max_by_len = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .max_by_key(|x| x.len());
    /// assert_eq!(max_by_len, Some(&"aaaaa"));
    /// # });
    /// ```
    fn max_by_key<T, F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item) -> T + Sync,
        T: Ord + Send,
        Self::Item: Send,
    {
        self.map(|x| (f(&x), x))
            .max_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, x)| x)
    }

    /// Returns the minimal item of this iterator, or [`None`] if this iterator
    /// is empty.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let min = input.par_iter().with_thread_pool(&mut thread_pool).min();
    /// assert_eq!(min, Some(&1));
    /// # });
    /// ```
    fn min(self) -> Option<Self::Item>
    where
        Self::Item: Ord + Send,
    {
        self.pipeline(
            || None,
            |min, _, x| match min {
                None => Some(x),
                Some(min) => Some(min.min(x)),
            },
            |min| min,
            |x, y| match (x, y) {
                (None, None) => None,
                (Some(a), None) | (None, Some(a)) => Some(a),
                (Some(a), Some(b)) => Some(a.min(b)),
            },
        )
    }

    /// Returns the minimal item of this iterator according to the comparison
    /// function `f`, or [`None`] if this iterator is empty.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// // Custom comparison function where even numbers are smaller than all odd numbers.
    /// let min = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .min_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
    /// assert_eq!(min, Some(&2));
    /// # });
    /// ```
    fn min_by<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> Ordering + Sync,
        Self::Item: Send,
    {
        self.pipeline(
            || None,
            |min, _, x| match min {
                None => Some(x),
                Some(min) => match f(&min, &x) {
                    Ordering::Less | Ordering::Equal => Some(min),
                    Ordering::Greater => Some(x),
                },
            },
            |min| min,
            |x, y| match (x, y) {
                (None, None) => None,
                (Some(a), None) | (None, Some(a)) => Some(a),
                (Some(a), Some(b)) => match f(&a, &b) {
                    Ordering::Less | Ordering::Equal => Some(a),
                    Ordering::Greater => Some(b),
                },
            },
        )
    }

    /// Returns the minimal item of this iterator according to the keys derived
    /// from the mapping function `f`, or [`None`] if this iterator is
    /// empty.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = ["ccc", "aaaaa", "dd", "e", "bbbb"];
    ///
    /// let min = input.par_iter().with_thread_pool(&mut thread_pool).min();
    /// assert_eq!(min, Some(&"aaaaa"));
    ///
    /// let min_by_len = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .min_by_key(|x| x.len());
    /// assert_eq!(min_by_len, Some(&"e"));
    /// # });
    /// ```
    fn min_by_key<T, F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item) -> T + Sync,
        T: Ord + Send,
        Self::Item: Send,
    {
        self.map(|x| (f(&x), x))
            .min_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, x)| x)
    }

    /// Reduces the items produced by this iterator into a single item, using
    /// `f` to collapse pairs of items.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool.scope(|mut thread_pool| {
    /// let sum = []
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
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

/// This struct is created by the [`cloned()`](ParallelIteratorExt::cloned)
/// method on [`ParallelIteratorExt`].
#[must_use = "iterator adaptors are lazy"]
pub struct Cloned<Inner: ParallelIterator> {
    inner: Inner,
}

impl<'a, T, Inner: ParallelIterator> ParallelIterator for Cloned<Inner>
where
    T: Clone + 'a,
    Inner: ParallelIterator<Item = &'a T>,
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
            |accum, index, item| process_item(accum, index, item.clone()),
            finalize,
            reduce,
        )
    }
}

/// This struct is created by the [`copied()`](ParallelIteratorExt::copied)
/// method on [`ParallelIteratorExt`].
#[must_use = "iterator adaptors are lazy"]
pub struct Copied<Inner: ParallelIterator> {
    inner: Inner,
}

impl<'a, T, Inner: ParallelIterator> ParallelIterator for Copied<Inner>
where
    T: Copy + 'a,
    Inner: ParallelIterator<Item = &'a T>,
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
            |accum, index, item| process_item(accum, index, *item),
            finalize,
            reduce,
        )
    }
}

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

/// This struct is created by the
/// [`filter_map()`](ParallelIteratorExt::filter_map) method on
/// [`ParallelIteratorExt`].
#[must_use = "iterator adaptors are lazy"]
pub struct FilterMap<Inner: ParallelIterator, F> {
    inner: Inner,
    f: F,
}

impl<Inner: ParallelIterator, T, F> ParallelIterator for FilterMap<Inner, F>
where
    F: Fn(Inner::Item) -> Option<T> + Sync,
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
            |accum, index, item| match (self.f)(item) {
                Some(mapped_item) => process_item(accum, index, mapped_item),
                None => accum,
            },
            finalize,
            reduce,
        )
    }
}

/// This struct is created by the [`inspect()`](ParallelIteratorExt::inspect)
/// method on [`ParallelIteratorExt`].
#[must_use = "iterator adaptors are lazy"]
pub struct Inspect<Inner: ParallelIterator, F> {
    inner: Inner,
    f: F,
}

impl<Inner: ParallelIterator, F> ParallelIterator for Inspect<Inner, F>
where
    F: Fn(&Inner::Item) + Sync,
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
                (self.f)(&item);
                process_item(accum, index, item)
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
