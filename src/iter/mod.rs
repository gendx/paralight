// Copyright 2024-2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Iterator adaptors to define parallel pipelines more conveniently.

mod detail;
mod sink;
mod source;

use detail::{
    AdaptorAccumulator, Fuse, IterAccumulator, IterCollector, IterFolder, IterReducer,
    MinMaxAccumulator, ProductAccumulator, ShortCircuitingAccumulator, SumAccumulator,
    TryIterCollector, TryIterFolder,
};
pub use detail::{Map, MinMaxResult};
#[cfg(feature = "nightly")]
pub use sink::array::ArrayParallelSink;
pub use sink::vec::VecParallelSink;
pub use sink::{ExactParallelSink, FromExactParallelSink};
#[cfg(feature = "nightly")]
pub use source::array::ArrayParallelSource;
#[cfg(all(test, any(feature = "rayon", feature = "default-thread-pool")))]
pub(crate) use source::hashset::MyHashSet;
pub use source::range::{RangeInclusiveParallelSource, RangeParallelSource};
pub use source::slice::{MutSliceParallelSource, SliceParallelSource};
pub use source::vec::VecParallelSource;
pub use source::vec_deque::{VecDequeRefMutParallelSource, VecDequeRefParallelSource};
pub use source::zip::{ZipEq, ZipMax, ZipMin, ZipableSource};
pub use source::{
    BaseExactParallelIterator, BaseParallelIterator, Chain, Cloned, Copied, Enumerate,
    ExactParallelSource, ExactParallelSourceExt, ExactSourceDescriptor, Filter, FilterMap, Inspect,
    IntoExactParallelRefMutSource, IntoExactParallelRefSource, IntoExactParallelSource,
    IntoParallelRefMutSource, IntoParallelRefSource, IntoParallelSource, MapInit, ParallelSource,
    ParallelSourceExt, Rev, SimpleExactSourceDescriptor, SimpleSourceDescriptor, Skip, SkipExact,
    SourceCleanup, SourceDescriptor, StepBy, Take, TakeExact,
};
use std::cmp::Ordering;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::ControlFlow;
#[cfg(feature = "nightly")]
use std::ops::{FromResidual, Residual, Try};

/// Interface for an operation that accumulates items from an iterator into an
/// output.
///
/// You can think of it as a variant of `Fn(impl Iterator) -> Output` made
/// generic over the item and output types.
pub trait Accumulator<Item, Output> {
    /// Accumulates the items from the given iterator into an output.
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output;
}

/// Interface for an operation that accumulates items from an iterator of exact
/// size into an output.
///
/// See also [`Accumulator`].
pub trait ExactSizeAccumulator<Item, Output> {
    /// Accumulates the items from the given iterator into an output.
    fn accumulate_exact(&self, iter: impl ExactSizeIterator<Item = Item>) -> Output;
}

/// A thread pool backend that can execute parallel iterators.
///
/// You most likely won't have to interact with this trait directly, as it is
/// implemented for [`&mut ThreadPool`](crate::threads::ThreadPool), and
/// interacting with a thread pool is done via the
/// [`with_thread_pool()`](ParallelSourceExt::with_thread_pool) iterator
/// adaptor. You can implement this trait if you want to use Paralight iterators
/// with an alternate thread pool implementation that you provide.
///
/// # Safety
///
/// This trait is marked as `unsafe`, because implementers **must** ensure the
/// safety guarantees of
/// [`upper_bounded_pipeline()`](Self::upper_bounded_pipeline) and
/// [`iter_pipeline()`](Self::iter_pipeline).
pub unsafe trait GenericThreadPool {
    /// Processes an input of the given length in parallel and returns the
    /// aggregated output.
    ///
    /// With this variant, the pipeline may skip processing items at larger
    /// indices whenever a call to `process_item` returns
    /// [`ControlFlow::Break`].
    ///
    /// # Safety guarantees
    ///
    /// This function guarantees that:
    /// - the indices passed to `process_item()` are in `0..input_len`,
    /// - the ranges passed to `cleanup.cleanup_item_range()` are included in
    ///   `0..input_len`,
    /// - each index in `0..inner_len` is passed exactly once in calls to
    ///   `process_item()` and `cleanup.cleanup_item_range()`.
    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        input_len: usize,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output;

    /// Processes an input of the given length in parallel and returns the
    /// aggregated output.
    ///
    /// # Safety guarantees
    ///
    /// This function guarantees that:
    /// - the indices passed to `accum.accumulate()` are in `0..input_len`,
    /// - the ranges passed to `cleanup.cleanup_item_range()` are included in
    ///   `0..input_len`,
    /// - each index in `0..inner_len` is passed exactly once in calls to
    ///   `accum.accumulate()` and `cleanup.cleanup_item_range()`.
    fn iter_pipeline<Output, Accum: Send>(
        self,
        input_len: usize,
        accum: impl Accumulator<usize, Accum> + Sync,
        reduce: impl ExactSizeAccumulator<Accum, Output>,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output;
}

// SAFETY: The implementation is the same as the one on `&T`, which safely
// implements `GenericThreadPool`.
unsafe impl<'a, T> GenericThreadPool for &'a mut T
where
    &'a T: GenericThreadPool,
{
    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        input_len: usize,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        (self as &'a T).upper_bounded_pipeline(
            input_len,
            init,
            process_item,
            finalize,
            reduce,
            cleanup,
        )
    }

    fn iter_pipeline<Output, Accum: Send>(
        self,
        input_len: usize,
        accum: impl Accumulator<usize, Accum> + Sync,
        reduce: impl ExactSizeAccumulator<Accum, Output>,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        (self as &'a T).iter_pipeline(input_len, accum, reduce, cleanup)
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
    /// [`map()`](ParallelIteratorExt::map) adaptor. However, initial sources of
    /// parallel iterators require the items to be [`Send`], via the
    /// [`IntoParallelSource`] family of traits.
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
    ///     .pipeline(|| 0, |acc, x| acc + x, |acc| acc, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, Self::Item) -> Accum + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.iter_pipeline(
            IterAccumulator {
                init: &init,
                process_item,
                finalize: &finalize,
            },
            IterReducer { reduce },
        )
        .unwrap_or_else(|| finalize(init()))
    }

    /// Runs the pipeline defined by the given functions on this iterator.
    ///
    /// # Parameters
    ///
    /// - `init` function to create a new (per-thread) accumulator,
    /// - `process_item` function to accumulate an item into the accumulator,
    /// - `finalize` function to transform an accumulator into an output,
    /// - `reduce` function to reduce a pair of outputs into one output.
    ///
    /// Contrary to [`pipeline()`](Self::pipeline), the `process_item` function
    /// can return [`ControlFlow::Break`] to indicate that the pipeline should
    /// terminate early.
    ///
    /// ```
    /// # #![allow(clippy::bool_assert_comparison)]
    /// # use paralight::prelude::*;
    /// # use std::ops::ControlFlow;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let any_even = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .short_circuiting_pipeline(
    ///         || (),
    ///         |_, x| {
    ///             if x % 2 == 0 {
    ///                 ControlFlow::Break(())
    ///             } else {
    ///                 ControlFlow::Continue(())
    ///             }
    ///         },
    ///         |acc| acc.is_break(),
    ///         |x, y| x || y,
    ///     );
    /// assert_eq!(any_even, true);
    /// ```
    fn short_circuiting_pipeline<Output: Send, Accum, Break>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, Self::Item) -> ControlFlow<Break, Accum> + Sync,
        finalize: impl Fn(ControlFlow<Break, Accum>) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.iter_pipeline(
            ShortCircuitingAccumulator {
                fuse: Fuse::new(),
                init: &init,
                process_item,
                finalize: &finalize,
            },
            IterReducer { reduce },
        )
        .unwrap_or_else(|| finalize(ControlFlow::Continue(init())))
    }

    /// Runs the pipeline defined by the given functions on this iterator.
    ///
    /// # Parameters
    ///
    /// - `init` function to create a new (per-thread) accumulator,
    /// - `process_item` function to accumulate an item into the accumulator,
    /// - `finalize` function to transform an accumulator into an output,
    /// - `reduce` function to reduce a pair of outputs into one output.
    ///
    /// Contrary to
    /// [`short_circuiting_pipeline()`](Self::short_circuiting_pipeline), the
    /// `process_item` function can return any type that implements the
    /// [`Try`] trait to indicate that the pipeline should terminate early
    /// (failures being represented by the corresponding [`Try::Residual`]
    /// type).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let any_even = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_short_circuiting_pipeline(
    ///         || (),
    ///         |(), x| {
    ///             if x % 2 == 0 {
    ///                 Err(x)
    ///             } else {
    ///                 Ok(())
    ///             }
    ///         },
    ///         |acc| acc,
    ///         |x, y| x.and(y),
    ///     );
    /// assert!(any_even.is_err());
    /// assert!(any_even.unwrap_err() % 2 == 0);
    /// ```
    #[cfg(feature = "nightly")]
    fn try_short_circuiting_pipeline<Output: Send, Accum, R: Try<Output = Accum>>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, Self::Item) -> R + Sync,
        finalize: impl Fn(R) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.iter_pipeline(
            ShortCircuitingAccumulator {
                fuse: Fuse::new(),
                init: &init,
                process_item,
                finalize: &finalize,
            },
            IterReducer { reduce },
        )
        .unwrap_or_else(|| finalize(Try::from_output(init())))
    }

    /// Runs the pipeline defined by the given functions on this iterator.
    ///
    /// # Parameters
    ///
    /// - `init` function to create a new (per-thread) accumulator,
    /// - `process_item` function to accumulate an item into the accumulator,
    /// - `finalize` function to transform an accumulator into an output,
    /// - `reduce` function to reduce a pair of outputs into one output.
    ///
    /// Contrary to [`pipeline()`](Self::pipeline), the `process_item` function
    /// can return [`ControlFlow::Break`] to indicate that the pipeline should
    /// skip processing items at larger indices.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::ops::ControlFlow;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let first_even = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .upper_bounded_pipeline(
    ///         || None,
    ///         |acc, i, x| {
    ///             match acc {
    ///                 // Early return if we found something at a previous index.
    ///                 Some((j, _)) if j < i => ControlFlow::Continue(acc),
    ///                 _ => match x % 2 == 0 {
    ///                     true => ControlFlow::Break(Some((i, x))),
    ///                     false => ControlFlow::Continue(acc),
    ///                 },
    ///             }
    ///         },
    ///         |acc| acc,
    ///         |x, y| match (x, y) {
    ///             (None, None) => None,
    ///             (Some(found), None) | (None, Some(found)) => Some(found),
    ///             (Some((i, a)), Some((j, b))) => {
    ///                 if i < j {
    ///                     Some((i, a))
    ///                 } else {
    ///                     Some((j, b))
    ///                 }
    ///             }
    ///         },
    ///     )
    ///     .map(|(_, x)| x);
    /// assert_eq!(first_even, Some(&2));
    /// ```
    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output;

    /// Runs the pipeline defined by the given functions on this iterator.
    ///
    /// # Parameters
    ///
    /// - `accum` function to accumulate items into an output,
    /// - `reduce` function to reduce a sequence of outputs into the final
    ///   output.
    ///
    /// ```
    /// # use paralight::iter::{Accumulator, ExactSizeAccumulator};
    /// # use paralight::prelude::*;
    /// use std::iter::Sum;
    ///
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
    ///     .iter_pipeline(SumAccumulator, SumAccumulator);
    /// assert_eq!(sum, 5 * 11);
    ///
    /// // Definition of an accumulator that sums items into an integer.
    /// struct SumAccumulator;
    ///
    /// impl<Item> Accumulator<Item, i32> for SumAccumulator
    /// where
    ///     i32: Sum<Item>,
    /// {
    ///     fn accumulate(&self, iter: impl Iterator<Item = Item>) -> i32 {
    ///         iter.sum()
    ///     }
    /// }
    ///
    /// impl<Item> ExactSizeAccumulator<Item, i32> for SumAccumulator
    /// where
    ///     i32: Sum<Item>,
    /// {
    ///     fn accumulate_exact(&self, iter: impl ExactSizeIterator<Item = Item>) -> i32 {
    ///         iter.sum()
    ///     }
    /// }
    /// ```
    fn iter_pipeline<Output, Accum: Send>(
        self,
        accum: impl Accumulator<Self::Item, Accum> + Sync,
        reduce: impl ExactSizeAccumulator<Accum, Output>,
    ) -> Output;
}

/// An internal object describing how to transform items in a
/// [`ParallelAdaptor`].
pub struct ParallelAdaptorDescriptor<
    Item,
    Inner: ParallelIterator,
    TransformItem: Fn(Inner::Item) -> Option<Item> + Sync,
> {
    /// The underlying parallel iterator that this adaptor builds on top of.
    inner: Inner,
    /// Function to transform an item from the inner iterator into the adapted
    /// iterator.
    transform_item: TransformItem,
}

/// An internal trait to define iterator adaptors. Types that implement this
/// trait automatically implement [`ParallelIterator`] and
/// [`ParallelIteratorExt`].
pub trait ParallelAdaptor {
    /// The type of items that this parallel iterator adaptor produces.
    ///
    /// As for [`ParallelIterator`] this type has no particular [`Send`] nor
    /// [`Sync`] bounds.
    type Item;
    /// The underlying parallel iterator that this type adapts on top of.
    type Inner: ParallelIterator;

    /// Definition of the parallel adaptor.
    // We can't really avoid the complexity in the result type as the function may come from an
    // anonymous lambda.
    #[allow(clippy::type_complexity)]
    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    >;
}

impl<T: ParallelAdaptor> ParallelIterator for T {
    type Item = T::Item;

    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        let descriptor = self.descriptor();
        descriptor.inner.upper_bounded_pipeline(
            init,
            |accum, index, item| match (descriptor.transform_item)(item) {
                Some(item) => process_item(accum, index, item),
                None => ControlFlow::Continue(accum),
            },
            finalize,
            reduce,
        )
    }

    fn iter_pipeline<Output, Accum: Send>(
        self,
        accum: impl Accumulator<Self::Item, Accum> + Sync,
        reduce: impl ExactSizeAccumulator<Accum, Output>,
    ) -> Output {
        let descriptor = self.descriptor();
        let accumulator = AdaptorAccumulator {
            inner: accum,
            transform_item: descriptor.transform_item,
        };
        descriptor.inner.iter_pipeline(accumulator, reduce)
    }
}

impl<T: ParallelIterator> ParallelIteratorExt for T {}

/// Additional methods provided for types that implement [`ParallelIterator`].
///
/// See also [`ParallelSourceExt`] for more adaptors that only apply to parallel
/// sources (earlier in the pipeline).
pub trait ParallelIteratorExt: ParallelIterator {
    /// Returns [`true`] if all items produced by this iterator satisfy the
    /// predicate `f`, and [`false`] otherwise.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    ///
    /// let all_even = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .all(|&x| x % 2 == 0);
    /// assert!(!all_even);
    ///
    /// let all_positive = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .all(|&x| x > 0);
    /// assert!(all_positive);
    /// ```
    ///
    /// This returns [`true`] if the iterator is empty.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input: [i32; 0] = [];
    ///
    /// let all_empty = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .all(|_| false);
    /// assert!(all_empty);
    /// ```
    fn all<F>(self, f: F) -> bool
    where
        F: Fn(Self::Item) -> bool + Sync,
    {
        self.short_circuiting_pipeline(
            || (),
            |_, item| match f(item) {
                true => ControlFlow::Continue(()),
                false => ControlFlow::Break(()),
            },
            |acc| acc.is_continue(),
            |x, y| x && y,
        )
    }

    /// Returns [`true`] if any item produced by this iterator satisfies the
    /// predicate `f`, and [`false`] otherwise.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    ///
    /// let any_even = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .any(|&x| x % 2 == 0);
    /// assert!(any_even);
    ///
    /// let any_zero = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .any(|&x| x == 0);
    /// assert!(!any_zero);
    /// ```
    ///
    /// This returns [`false`] if the iterator is empty.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input: [i32; 0] = [];
    ///
    /// let any_empty = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .any(|_| true);
    /// assert!(!any_empty);
    /// ```
    fn any<F>(self, f: F) -> bool
    where
        F: Fn(Self::Item) -> bool + Sync,
    {
        self.short_circuiting_pipeline(
            || (),
            |_, item| match f(item) {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            },
            |acc| acc.is_break(),
            |x, y| x || y,
        )
    }

    /// Compare the pairs of items produced by this iterator using
    /// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order)
    /// order.
    ///
    /// See also [`cmp_by()`](Self::cmp_by), [`cmp_by_key()`](Self::cmp_by_key)
    /// and [`cmp_by_keys()`](Self::cmp_by_keys) to use custom comparison
    /// functions, and [`partial_cmp()`](Self::partial_cmp) and its variants if
    /// the items only implement [`PartialOrd`].
    ///
    /// If you only care about equality, [`eq()`](Self::eq), [`ne()`](Self::ne)
    /// and their variants are more efficient.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let rhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .cmp();
    /// assert_eq!(ordering, Ordering::Equal);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let rhs = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .cmp();
    /// assert_eq!(ordering, Ordering::Less);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [10, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    /// let rhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .cmp();
    /// assert_eq!(ordering, Ordering::Greater);
    /// ```
    ///
    /// This returns [`Ordering::Equal`] if the iterator is empty.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let pairs: [(i32, i32); 0] = [];
    /// let ordering = pairs
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .cmp();
    /// assert_eq!(ordering, Ordering::Equal);
    /// ```
    fn cmp<T>(self) -> Ordering
    where
        Self: ParallelIterator<Item = (T, T)>,
        T: Ord,
    {
        self.cmp_by(|x, y| x.cmp(&y))
    }

    /// Compare the pairs of items produced by this iterator using
    /// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order)
    /// order with the comparison function `f`.
    ///
    /// See also the other comparison adaptors such as [`cmp()`](Self::cmp) or
    /// [`partial_cmp_by()`](Self::partial_cmp_by).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// /// Compare every other item of the given slices.
    /// fn compare_even_indices(&lhs: &&[i32], &rhs: &&[i32]) -> Ordering {
    ///     lhs.iter().step_by(2).cmp(rhs.iter().step_by(2))
    /// }
    ///
    /// let lhs: [&[_]; 5] = [&[1, 1], &[2, 2], &[3, 3], &[4, 4], &[5, 5]];
    /// let rhs: [&[_]; 5] = [&[1, 6], &[2, 7], &[3, 8], &[4, 9], &[5, 10]];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .cmp_by(compare_even_indices);
    /// // The comparison is [Equal, Equal, Equal, Equal, Equal] -> Equal
    /// assert_eq!(ordering, Ordering::Equal);
    ///
    /// let lhs: [&[_]; 5] = [&[1, 1, 1], &[2, 2, 3], &[4, 3], &[4, 4, 5], &[5, 5, 6]];
    /// let rhs: [&[_]; 5] = [&[1, 6, 2], &[2, 7], &[3, 8, 3], &[4, 9, 4], &[5, 10, 5]];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .cmp_by(compare_even_indices);
    /// // The comparison is [Less, Greater, Greater, Greater, Greater] -> Less
    /// assert_eq!(ordering, Ordering::Less);
    /// ```
    fn cmp_by<T, U, F>(self, f: F) -> Ordering
    where
        Self: ParallelIterator<Item = (T, U)>,
        F: Fn(T, U) -> Ordering + Sync,
    {
        self.map(|(t, u)| f(t, u))
            .find_first(|&ordering| ordering != Ordering::Equal)
            .unwrap_or(Ordering::Equal)
    }

    /// Compare the pairs of items produced by this iterator using
    /// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order)
    /// order, after mapping them with `f`.
    ///
    /// See also the other comparison adaptors such as
    /// [`cmp_by_keys()`](Self::cmp_by_keys),
    /// [`partial_cmp_by_key()`](Self::partial_cmp_by_key) or
    /// [`eq_by_key()`](Self::eq_by_key).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [
    ///     (1, "Lorem"),
    ///     (2, "ipsum"),
    ///     (3, "dolor"),
    ///     (4, "sit"),
    ///     (5, "amet"),
    /// ];
    /// let rhs = [
    ///     (5, "Lorem"),
    ///     (1, "ipsum"),
    ///     (2, "dolor"),
    ///     (3, "sit"),
    ///     (4, "amet"),
    /// ];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .cmp_by_key(|x| x.0);
    /// assert_eq!(ordering, Ordering::Less);
    /// ```
    fn cmp_by_key<T, A, F>(self, f: F) -> Ordering
    where
        Self: ParallelIterator<Item = (T, T)>,
        F: Fn(T) -> A + Sync,
        A: Ord,
    {
        self.cmp_by(|x, y| f(x).cmp(&f(y)))
    }

    /// Compare the pairs of items produced by this iterator using
    /// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order)
    /// order, after mapping them with `f` and `g`.
    ///
    /// See also the other comparison adaptors such as
    /// [`cmp_by_key()`](Self::cmp_by_key),
    /// [`partial_cmp_by_keys()`](Self::partial_cmp_by_keys) or
    /// [`eq_by_keys()`](Self::eq_by_keys).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0)];
    /// let rhs = [
    ///     (5, "Lorem"),
    ///     (1, "ipsum"),
    ///     (2, "dolor"),
    ///     (3, "sit"),
    ///     (4, "amet"),
    /// ];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .cmp_by_keys(|x| x.0, |y| y.0);
    /// assert_eq!(ordering, Ordering::Less);
    /// ```
    fn cmp_by_keys<T, U, A, F, G>(self, f: F, g: G) -> Ordering
    where
        Self: ParallelIterator<Item = (T, U)>,
        F: Fn(T) -> A + Sync,
        G: Fn(U) -> A + Sync,
        A: Ord,
    {
        self.cmp_by(|t, u| f(t).cmp(&g(u)))
    }

    /// Collect items from this iterator into a per-thread collection of type
    /// `C`, then aggregates these into a `Vec<C>` without further
    /// flattening.
    ///
    /// See also [`try_collect_per_thread()`](Self::try_collect_per_thread) and
    /// [`fold_per_thread()`](Self::fold_per_thread).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let collection: Vec<Vec<i32>> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .collect_per_thread();
    ///
    /// let mut values: Vec<i32> = collection.into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    /// ```
    ///
    /// The outer type of the result is currently always [`Vec`], but the inner
    /// type can be anything that implements [`FromIterator`].
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let collection: Vec<HashSet<i32>> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .collect_per_thread();
    ///
    /// let mut values: Vec<i32> = collection.into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    /// ```
    fn collect_per_thread<C>(self) -> Vec<C>
    where
        C: FromIterator<Self::Item> + Send,
    {
        self.iter_pipeline(
            IterCollector::<C> {
                _phantom: PhantomData,
            },
            IterFolder {
                init: Vec::with_capacity,
                fold: |mut vec: Vec<C>, c| {
                    vec.push(c);
                    vec
                },
            },
        )
    }

    /// Returns [`true`] if all pairs of items produced by this iterator are
    /// equal, and [`false`] otherwise.
    ///
    /// This returns the opposite of [`ne()`](Self::ne).
    ///
    /// See also [`eq_by_key()`](Self::eq_by_key) and
    /// [`eq_by_keys()`](Self::eq_by_keys) to use custom comparison functions,
    /// or the more general [`all()`](Self::all) and [`any()`](Self::any).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let rhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .eq();
    /// assert!(equal);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let rhs = [1, 2, 3, 42, 5, 6, 7, 8, 9, 10];
    /// let equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .eq();
    /// assert!(!equal);
    /// ```
    ///
    /// The underlying items only need to implement [`PartialEq`], full equality
    /// with [`Eq`] isn't required.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1.0, f64::NAN];
    /// let rhs = [1.0, f64::NAN];
    /// let equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .eq();
    /// assert!(!equal);
    /// ```
    ///
    /// The underlying items don't need to be of the same type, as long as they
    /// implement [`PartialEq`] with each other.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs: &[[i32; 2]] = &[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]];
    /// let rhs: &[&[i32]] = &[&[1, 2], &[3, 4], &[5, 6], &[7, 8], &[9, 10]];
    /// let equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .eq();
    /// assert!(equal);
    /// ```
    ///
    /// This returns [`true`] if the iterator is empty.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let pairs: [(i32, i32); 0] = [];
    /// let eq_empty = pairs
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .eq();
    /// assert!(eq_empty);
    /// ```
    fn eq<T, U>(self) -> bool
    where
        Self: ParallelIterator<Item = (T, U)>,
        T: PartialEq<U>,
    {
        self.all(|(t, u)| t == u)
    }

    /// Returns [`true`] if all pairs of items produced by this iterator yield
    /// equal keys when mapped by `f`, and [`false`] otherwise.
    ///
    /// This returns the opposite of [`ne_by_key()`](Self::ne_by_key).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0)];
    /// let rhs = [(1, 6.0), (2, 7.0), (3, 8.0), (4, 9.0), (5, 10.0)];
    /// let equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .eq_by_key(|x| x.0);
    /// assert!(equal);
    /// ```
    ///
    /// See also:
    /// - [`eq()`](Self::eq) for the simplest equality adaptor over
    ///   [`PartialEq`] types,
    /// - [`eq_by_keys()`](Self::eq_by_keys) if the left-hand and right-hand
    ///   sides are of different types or have different key extraction
    ///   functions,
    /// - [`all()`](Self::all) and [`any()`](Self::any) for more general
    ///   adaptors.
    fn eq_by_key<T, A, F>(self, f: F) -> bool
    where
        Self: ParallelIterator<Item = (T, T)>,
        F: Fn(T) -> A + Sync,
        A: PartialEq,
    {
        self.all(|(t, u)| f(t) == f(u))
    }

    /// Returns [`true`] if all pairs of items produced by this iterator yield
    /// equal keys when mapped by `f` and `g`, and [`false`] otherwise.
    ///
    /// This returns the opposite of [`ne_by_keys()`](Self::ne_by_keys).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0)];
    /// let rhs = [
    ///     (1, "Lorem"),
    ///     (2, "ipsum"),
    ///     (3, "dolor"),
    ///     (4, "sit"),
    ///     (5, "amet"),
    /// ];
    /// let equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .eq_by_keys(|x| x.0, |y| y.0);
    /// assert!(equal);
    /// ```
    ///
    /// See also:
    /// - [`eq()`](Self::eq) for the simplest equality adaptor over
    ///   [`PartialEq`] types,
    /// - [`eq_by_key()`](Self::eq_by_key) if the left-hand and right-hand key
    ///   extraction functions are the same and on the same types,
    /// - [`all()`](Self::all) and [`any()`](Self::any) for more general
    ///   adaptors.
    fn eq_by_keys<T, U, A, B, F, G>(self, f: F, g: G) -> bool
    where
        Self: ParallelIterator<Item = (T, U)>,
        F: Fn(T) -> A + Sync,
        G: Fn(U) -> B + Sync,
        A: PartialEq<B>,
    {
        self.all(|(t, u)| f(t) == g(u))
    }

    /// Returns any item that satisfies the predicate `f`, or [`None`] if no
    /// item satisfies it.
    ///
    /// If multiple items satisfy `f`, an arbitrary one is returned.
    ///
    /// See also [`find_map_any()`](Self::find_map_any), useful if the
    /// [`Item`](ParallelIterator::Item) type isn't [`Send`].
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    ///
    /// let four = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .find_any(|&&x| x == 4);
    /// assert_eq!(four, Some(&4));
    ///
    /// let twenty = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .find_any(|&&x| x == 20);
    /// assert_eq!(twenty, None);
    ///
    /// let any_even = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .find_any(|&x| x % 2 == 0);
    /// assert!(any_even.unwrap() % 2 == 0);
    /// ```
    fn find_any<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item) -> bool + Sync,
        Self::Item: Send,
    {
        self.short_circuiting_pipeline(
            || (),
            |_, item| match f(&item) {
                true => ControlFlow::Break(item),
                false => ControlFlow::Continue(()),
            },
            |acc| match acc {
                ControlFlow::Break(item) => Some(item),
                ControlFlow::Continue(()) => None,
            },
            |x, y| x.or(y),
        )
    }

    /// Returns the first item that satisfies the predicate `f`, or [`None`] if
    /// no item satisfies it.
    ///
    /// See also [`find_map_first()`](Self::find_map_first), useful if the
    /// [`Item`](ParallelIterator::Item) type isn't [`Send`].
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    ///
    /// let four = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .find_first(|&&x| x == 4);
    /// assert_eq!(four, Some(&4));
    ///
    /// let twenty = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .find_first(|&&x| x == 20);
    /// assert_eq!(twenty, None);
    ///
    /// let first_even = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .find_first(|&x| x % 2 == 0);
    /// assert_eq!(first_even, Some(2));
    /// ```
    fn find_first<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item) -> bool + Sync,
        Self::Item: Send,
    {
        self.upper_bounded_pipeline(
            || None,
            |acc, i, item| {
                match acc {
                    // Early return if we found something at a previous index.
                    Some((j, _)) if j < i => ControlFlow::Continue(acc),
                    _ => match f(&item) {
                        true => ControlFlow::Break(Some((i, item))),
                        false => ControlFlow::Continue(acc),
                    },
                }
            },
            |acc| acc,
            |x, y| match (x, y) {
                (None, None) => None,
                (Some(found), None) | (None, Some(found)) => Some(found),
                (Some((i, a)), Some((j, b))) => {
                    if i < j {
                        Some((i, a))
                    } else {
                        Some((j, b))
                    }
                }
            },
        )
        .map(|(_, item)| item)
    }

    /// Applies the function `f` to items of this iterator, returning any
    /// non-[`None`] result.
    ///
    /// If multiple items map to [`Some(_)`](Option::Some) result, an arbitrary
    /// one is returned.
    ///
    /// Contrary to [`find_any()`](Self::find_any), the
    /// [`Item`](ParallelIterator::Item) type doesn't need to be [`Send`], only
    /// the result type.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [
    ///     "Lorem",
    ///     "ipsum",
    ///     "dolor",
    ///     "sit",
    ///     "amet",
    ///     "consectetur",
    ///     "adipiscing",
    ///     "elit",
    ///     "sed",
    ///     "do",
    /// ];
    ///
    /// let four_bytes_word = input
    ///     .par_iter()
    ///     .enumerate()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .find_map_any(|(i, x)| if x.len() == 4 { Some(i) } else { None });
    ///
    /// assert!(four_bytes_word.is_some());
    /// let four_bytes_word = four_bytes_word.unwrap();
    /// assert!(four_bytes_word == 4 || four_bytes_word == 7);
    /// assert_eq!(input[four_bytes_word].len(), 4);
    /// ```
    fn find_map_any<T, F>(self, f: F) -> Option<T>
    where
        F: Fn(Self::Item) -> Option<T> + Sync,
        T: Send,
    {
        self.short_circuiting_pipeline(
            || (),
            |_, item| match f(item) {
                Some(t) => ControlFlow::Break(t),
                None => ControlFlow::Continue(()),
            },
            |acc| match acc {
                ControlFlow::Break(t) => Some(t),
                ControlFlow::Continue(()) => None,
            },
            |x, y| x.or(y),
        )
    }

    /// Applies the function `f` to items of this iterator, returning the first
    /// non-[`None`] result.
    ///
    /// Contrary to [`find_first()`](Self::find_first), the
    /// [`Item`](ParallelIterator::Item) type doesn't need to be [`Send`], only
    /// the result type.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [
    ///     "Lorem",
    ///     "ipsum",
    ///     "dolor",
    ///     "sit",
    ///     "amet",
    ///     "consectetur",
    ///     "adipiscing",
    ///     "elit",
    ///     "sed",
    ///     "do",
    /// ];
    ///
    /// let four_bytes_word = input
    ///     .par_iter()
    ///     .enumerate()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .find_map_first(|(i, x)| if x.len() == 4 { Some(i) } else { None });
    ///
    /// // Position of "amet" in the input array.
    /// assert_eq!(four_bytes_word, Some(4));
    /// ```
    fn find_map_first<T, F>(self, f: F) -> Option<T>
    where
        F: Fn(Self::Item) -> Option<T> + Sync,
        T: Send,
    {
        self.upper_bounded_pipeline(
            || None,
            |acc, i, item| {
                match acc {
                    // Early return if we found something at a previous index.
                    Some((j, _)) if j < i => ControlFlow::Continue(acc),
                    _ => match f(item) {
                        Some(t) => ControlFlow::Break(Some((i, t))),
                        None => ControlFlow::Continue(acc),
                    },
                }
            },
            |acc| acc,
            |x, y| match (x, y) {
                (None, None) => None,
                (Some(found), None) | (None, Some(found)) => Some(found),
                (Some((i, a)), Some((j, b))) => {
                    if i < j {
                        Some((i, a))
                    } else {
                        Some((j, b))
                    }
                }
            },
        )
        .map(|(_, t)| t)
    }

    /// Folds items of this parallel iterator into a per-thread accumulator of
    /// type `T`, then folds again the per-thread results into an output of
    /// type `U`.
    ///
    /// Per-thread accumulation is done with the `init_per_thread` and
    /// `fold_per_thread` functions, and final accumulation with the
    /// `init_final` and `fold_final` functions. The `init_final`
    /// function receives the number of threads as input. The
    /// [`Item`](ParallelIterator::Item) and `U` types can be non-[`Send`]: only
    /// the per-thread accumulator type `T` needs to be [`Send`].
    ///
    /// See also [`try_fold_per_thread()`](Self::try_fold_per_thread) and
    /// [`collect_per_thread()`](Self::collect_per_thread).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let collection: Vec<Vec<i32>> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .fold_per_thread(
    ///         Vec::new,
    ///         |mut vec, x| {
    ///             vec.push(x);
    ///             vec
    ///         },
    ///         Vec::with_capacity,
    ///         |mut vecvec, vec| {
    ///             vecvec.push(vec);
    ///             vecvec
    ///         },
    ///     );
    ///
    /// let mut values: Vec<i32> = collection.into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    /// ```
    fn fold_per_thread<T, U, InitPerThread, FoldPerThread, InitFinal, FoldFinal>(
        self,
        init_per_thread: InitPerThread,
        fold_per_thread: FoldPerThread,
        init_final: InitFinal,
        fold_final: FoldFinal,
    ) -> U
    where
        InitPerThread: Fn() -> T + Sync,
        FoldPerThread: Fn(T, Self::Item) -> T + Sync,
        InitFinal: Fn(usize) -> U,
        FoldFinal: Fn(U, T) -> U,
        T: Send,
    {
        self.iter_pipeline(
            IterAccumulator {
                init: init_per_thread,
                process_item: fold_per_thread,
                finalize: |t| t,
            },
            IterFolder {
                init: init_final,
                fold: fold_final,
            },
        )
    }

    /// Runs `f` on each item of this parallel iterator.
    ///
    /// See also [`for_each_init()`](Self::for_each_init) if you need to
    /// initialize a per-thread value and pass it together with each item.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let set = Mutex::new(HashSet::new());
    /// input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each(|&x| {
    ///         set.lock().unwrap().insert(x);
    ///     });
    /// assert_eq!(set.into_inner().unwrap(), (1..=10).collect());
    /// ```
    fn for_each<F>(self, f: F)
    where
        F: Fn(Self::Item) + Sync,
    {
        self.pipeline(
            /* init */ || (),
            /* process_item */ |(), item| f(item),
            /* finalize */ |()| (),
            /* reduce */ |(), ()| (),
        )
    }

    /// Runs `f` on each item of this iterator, together with a per-thread
    /// mutable value returned by `init`.
    ///
    /// The `init` function will be called only once per worker thread. The
    /// companion value returned by `init` doesn't need to be [`Send`] nor
    /// [`Sync`].
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// use rand::Rng;
    ///
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let mut bits: [bool; 128] = [true; 128];
    /// bits.par_iter_mut()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each_init(
    ///         rand::rng, // A thread-local RNG that is neither Send nor Sync.
    ///         |rng, bit| {
    ///             if rng.random() {
    ///                 *bit = false;
    ///             }
    ///         },
    ///     );
    ///
    /// // The probability that these checks fail is negligible.
    /// assert!(bits.iter().any(|&x| x));
    /// assert!(bits.iter().any(|&x| !x));
    /// ```
    fn for_each_init<T, Init, F>(self, init: Init, f: F)
    where
        Init: Fn() -> T + Sync,
        F: Fn(&mut T, Self::Item) + Sync,
    {
        self.pipeline(
            init,
            |mut t, item| {
                f(&mut t, item);
                t
            },
            /* finalize */ |_| (),
            /* reduce */ |(), ()| (),
        )
    }

    /// Applies the function `f` to each item of this iterator, returning a
    /// parallel iterator producing the mapped items.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let double_sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .map(|&x| x * 2)
    ///     .sum::<i32>();
    /// assert_eq!(double_sum, 10 * 11);
    /// ```
    ///
    /// Mapping to a non-[`Send`] non-[`Sync`] type such as [`Rc`](std::rc::Rc)
    /// is fine.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::rc::Rc;
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
    ///     .map(|&x| Rc::new(x))
    ///     .pipeline(|| 0, |acc, x| acc + *x, |acc| acc, |a, b| a + b);
    /// assert_eq!(sum, 5 * 11);
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
    /// If there are several maximal items, an arbitrary one is returned.
    ///
    /// See also [`minmax()`](Self::minmax).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let max = input.par_iter().with_thread_pool(&mut thread_pool).max();
    /// assert_eq!(max, Some(&10));
    /// ```
    fn max(self) -> Option<Self::Item>
    where
        Self::Item: Ord + Send,
    {
        self.pipeline(
            || None,
            |max, x| match max {
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
    /// If there are several maximal items, an arbitrary one is returned.
    ///
    /// See also [`minmax_by()`](Self::minmax_by).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let max = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     // Custom comparison function where even numbers are smaller than all odd numbers.
    ///     .max_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
    /// assert_eq!(max, Some(&9));
    /// ```
    fn max_by<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> Ordering + Sync,
        Self::Item: Send,
    {
        self.pipeline(
            || None,
            |max, x| match max {
                None => Some(x),
                Some(max) => match f(&max, &x) {
                    Ordering::Greater => Some(max),
                    Ordering::Less | Ordering::Equal => Some(x),
                },
            },
            |max| max,
            |x, y| match (x, y) {
                (None, None) => None,
                (Some(a), None) | (None, Some(a)) => Some(a),
                (Some(a), Some(b)) => match f(&a, &b) {
                    Ordering::Greater => Some(a),
                    Ordering::Less | Ordering::Equal => Some(b),
                },
            },
        )
    }

    /// Returns the maximal item of this iterator according to the keys derived
    /// from the mapping function `f`, or [`None`] if this iterator is
    /// empty.
    ///
    /// If there are several maximal items, an arbitrary one is returned.
    ///
    /// See also [`minmax_by_key()`](Self::minmax_by_key).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
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
    /// If there are several minimal items, an arbitrary one is returned.
    ///
    /// See also [`minmax()`](Self::minmax).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let min = input.par_iter().with_thread_pool(&mut thread_pool).min();
    /// assert_eq!(min, Some(&1));
    /// ```
    fn min(self) -> Option<Self::Item>
    where
        Self::Item: Ord + Send,
    {
        self.pipeline(
            || None,
            |min, x| match min {
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
    /// If there are several minimal items, an arbitrary one is returned.
    ///
    /// See also [`minmax_by()`](Self::minmax_by).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let min = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     // Custom comparison function where even numbers are smaller than all odd numbers.
    ///     .min_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
    /// assert_eq!(min, Some(&2));
    /// ```
    fn min_by<F>(self, f: F) -> Option<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> Ordering + Sync,
        Self::Item: Send,
    {
        self.pipeline(
            || None,
            |min, x| match min {
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
    /// If there are several minimal items, an arbitrary one is returned.
    ///
    /// See also [`minmax_by_key()`](Self::minmax_by_key).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
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

    /// Returns the minimal and maximal items of this iterator.
    ///
    /// If there are several minimal and/or maximal items, arbirary ones are
    /// returned.
    ///
    /// This performs only `1.5 * n` comparisons for an iterator of length `n`,
    /// which is more efficient than calling [`min()`](Self::min) and
    /// [`max()`](Self::max) separately.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// use paralight::iter::MinMaxResult;
    ///
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let minmax = input.par_iter().with_thread_pool(&mut thread_pool).minmax();
    /// assert_eq!(minmax, MinMaxResult::MinMax { min: &1, max: &10 });
    ///
    /// let input = [1];
    /// let minmax = input.par_iter().with_thread_pool(&mut thread_pool).minmax();
    /// assert_eq!(minmax, MinMaxResult::OneElement(&1));
    ///
    /// let input: [i32; 0] = [];
    /// let minmax = input.par_iter().with_thread_pool(&mut thread_pool).minmax();
    /// assert_eq!(minmax, MinMaxResult::NoElements);
    /// ```
    fn minmax(self) -> MinMaxResult<Self::Item>
    where
        Self::Item: Ord + Send,
    {
        self.minmax_by(Self::Item::cmp)
    }

    /// Returns the minimal and maximal items of this iterator according to the
    /// comparison function `f`.
    ///
    /// If there are several minimal and/or maximal items, arbirary ones are
    /// returned.
    ///
    /// This performs only `1.5 * n` comparisons for an iterator of length `n`,
    /// which is more efficient than calling [`min_by()`](Self::min_by) and
    /// [`max_by()`](Self::max_by) separately.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// use paralight::iter::MinMaxResult;
    ///
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let minmax = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     // Custom comparison function where even numbers are smaller than all odd numbers.
    ///     .minmax_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
    /// assert_eq!(minmax, MinMaxResult::MinMax { min: &2, max: &9 });
    /// ```
    fn minmax_by<F>(self, f: F) -> MinMaxResult<Self::Item>
    where
        F: Fn(&Self::Item, &Self::Item) -> Ordering + Sync,
        Self::Item: Send,
    {
        let accumulator = MinMaxAccumulator { f };
        self.iter_pipeline(&accumulator, &accumulator)
    }

    /// Returns the minimal and maximal items of this iterator according to the
    /// keys derived from the mapping function `f`.
    ///
    /// If there are several minimal and/or maximal items, arbirary ones are
    /// returned.
    ///
    /// This performs only `1.5 * n` comparisons for an iterator of length `n`,
    /// which is more efficient than calling
    /// [`min_by_key()`](Self::min_by_key) and
    /// [`max_by_key()`](Self::max_by_key) separately.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// use paralight::iter::MinMaxResult;
    ///
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = ["ccc", "aaaaa", "dd", "e", "bbbb"];
    ///
    /// let minmax = input.par_iter().with_thread_pool(&mut thread_pool).minmax();
    /// assert_eq!(
    ///     minmax,
    ///     MinMaxResult::MinMax {
    ///         min: &"aaaaa",
    ///         max: &"e"
    ///     }
    /// );
    ///
    /// let minmax_by_len = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .minmax_by_key(|x| x.len());
    /// assert_eq!(
    ///     minmax_by_len,
    ///     MinMaxResult::MinMax {
    ///         min: &"e",
    ///         max: &"aaaaa"
    ///     }
    /// );
    /// ```
    fn minmax_by_key<T, F>(self, f: F) -> MinMaxResult<Self::Item>
    where
        F: Fn(&Self::Item) -> T + Sync,
        T: Ord + Send,
        Self::Item: Send,
    {
        self.map(|x| (f(&x), x))
            .minmax_by(|a, b| a.0.cmp(&b.0))
            .map(|(_, x)| x)
    }

    /// Returns [`true`] if any pair of items produced by this iterator consists
    /// of unequal items, and [`false`] otherwise.
    ///
    /// This returns the opposite of [`eq()`](Self::eq).
    ///
    /// See also [`ne_by_key()`](Self::ne_by_key) and
    /// [`ne_by_keys()`](Self::ne_by_keys) to use custom comparison functions,
    /// or the more general [`all()`](Self::all) and [`any()`](Self::any).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let rhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let not_equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .ne();
    /// assert!(!not_equal);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let rhs = [1, 2, 3, 42, 5, 6, 7, 8, 9, 10];
    /// let not_equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .ne();
    /// assert!(not_equal);
    /// ```
    ///
    /// The underlying items only need to implement [`PartialEq`], full equality
    /// with [`Eq`] isn't required.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1.0, f64::NAN];
    /// let rhs = [1.0, f64::NAN];
    /// let not_equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .ne();
    /// assert!(not_equal);
    /// ```
    ///
    /// The underlying items don't need to be of the same type, as long as they
    /// implement [`PartialEq`] with each other.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs: &[[i32; 2]] = &[[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]];
    /// let rhs: &[&[i32]] = &[&[1, 2], &[3, 4], &[5, 6], &[7, 8], &[9, 10]];
    /// let not_equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .ne();
    /// assert!(!not_equal);
    /// ```
    ///
    /// This returns [`false`] if the iterator is empty.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let pairs: [(i32, i32); 0] = [];
    /// let ne_empty = pairs
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .ne();
    /// assert!(!ne_empty);
    /// ```
    fn ne<T, U>(self) -> bool
    where
        Self: ParallelIterator<Item = (T, U)>,
        T: PartialEq<U>,
    {
        !self.eq()
    }

    /// Returns [`true`] if any pair of items produced by this iterator yields
    /// unequal keys when mapped by `f`, and [`false`] otherwise.
    ///
    /// This returns the opposite of [`eq_by_key()`](Self::eq_by_key).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0)];
    /// let rhs = [(1, 6.0), (2, 7.0), (3, 8.0), (4, 9.0), (5, 10.0)];
    /// let not_equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .ne_by_key(|x| x.0);
    /// assert!(!not_equal);
    /// ```
    ///
    /// See also:
    /// - [`ne()`](Self::ne) for the simplest equality adaptor over
    ///   [`PartialEq`] types,
    /// - [`ne_by_keys()`](Self::ne_by_keys) if the left-hand and right-hand
    ///   sides are of different types or have different key extraction
    ///   functions,
    /// - [`all()`](Self::all) and [`any()`](Self::any) for more general
    ///   adaptors.
    fn ne_by_key<T, A, F>(self, f: F) -> bool
    where
        Self: ParallelIterator<Item = (T, T)>,
        F: Fn(T) -> A + Sync,
        A: PartialEq,
    {
        !self.eq_by_key(f)
    }

    /// Returns [`true`] if any pair of items produced by this iterator yields
    /// unequal keys when mapped by `f` and `g`, and [`false`] otherwise.
    ///
    /// This returns the opposite of [`eq_by_keys()`](Self::eq_by_keys).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [(1, 1.0), (2, 2.0), (3, 3.0), (4, 4.0), (5, 5.0)];
    /// let rhs = [
    ///     (1, "Lorem"),
    ///     (2, "ipsum"),
    ///     (3, "dolor"),
    ///     (4, "sit"),
    ///     (5, "amet"),
    /// ];
    /// let not_equal = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .ne_by_keys(|x| x.0, |y| y.0);
    /// assert!(!not_equal);
    /// ```
    ///
    /// See also:
    /// - [`ne()`](Self::ne) for the simplest equality adaptor over
    ///   [`PartialEq`] types,
    /// - [`ne_by_key()`](Self::ne_by_key) if the left-hand and right-hand key
    ///   extraction functions are the same and on the same types,
    /// - [`all()`](Self::all) and [`any()`](Self::any) for more general
    ///   adaptors.
    fn ne_by_keys<T, U, A, B, F, G>(self, f: F, g: G) -> bool
    where
        Self: ParallelIterator<Item = (T, U)>,
        F: Fn(T) -> A + Sync,
        G: Fn(U) -> B + Sync,
        A: PartialEq<B>,
    {
        !self.eq_by_keys(f, g)
    }

    /// Compare the pairs of items produced by this iterator using
    /// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order)
    /// order.
    ///
    /// See also [`partial_cmp_by()`](Self::partial_cmp_by),
    /// [`partial_cmp_by_key()`](Self::partial_cmp_by_key) and
    /// [`partial_cmp_by_keys()`](Self::partial_cmp_by_keys) to use custom
    /// comparison functions, and [`cmp()`](Self::cmp) and its variants if the
    /// items implement [`Ord`].
    ///
    /// If you only care about equality, [`eq()`](Self::eq), [`ne()`](Self::ne)
    /// and their variants are more efficient.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    /// let rhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp();
    /// assert_eq!(ordering, Some(Ordering::Equal));
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    /// let rhs = [10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp();
    /// assert_eq!(ordering, Some(Ordering::Less));
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, f64::NAN];
    /// let rhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, f64::NAN];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp();
    /// assert_eq!(ordering, None);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [2.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    /// let rhs = [1.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp();
    /// assert_eq!(ordering, Some(Ordering::Greater));
    /// ```
    ///
    /// This returns [`Some(Ordering::Equal)`](Option::Some) if the iterator is
    /// empty.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let pairs: [(f64, f64); 0] = [];
    /// let ordering = pairs
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp();
    /// assert_eq!(ordering, Some(Ordering::Equal));
    /// ```
    fn partial_cmp<T, U>(self) -> Option<Ordering>
    where
        Self: ParallelIterator<Item = (T, U)>,
        T: PartialOrd<U>,
    {
        self.partial_cmp_by(|x, y| x.partial_cmp(&y))
    }

    /// Compare the pairs of items produced by this iterator using
    /// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order)
    /// order with the comparison function `f`.
    ///
    /// See also the other comparison adaptors such as
    /// [`partial_cmp()`](Self::partial_cmp) or [`cmp_by()`](Self::cmp_by).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// /// Compare if one set is a subset of the other. Inputs must be sorted.
    /// fn compare_sets(&lhs: &&[i32], &rhs: &&[i32]) -> Option<Ordering> {
    ///     match (is_subset(lhs, rhs), is_subset(rhs, lhs)) {
    ///         (false, false) => None,
    ///         (true, false) => Some(Ordering::Less),
    ///         (false, true) => Some(Ordering::Greater),
    ///         (true, true) => Some(Ordering::Equal),
    ///     }
    /// }
    ///
    /// /// Returns true if the first argument is a subset of the second. Inputs must be sorted.
    /// # // TODO(MSRV >= 1.82.0): Remove this allowance
    /// # #[allow(clippy::incompatible_msrv)]
    /// fn is_subset(lhs: &[i32], rhs: &[i32]) -> bool {
    ///     /* Implementation details omitted */
    /// #    assert!(lhs.is_sorted());
    /// #    assert!(rhs.is_sorted());
    /// #    let mut lit = lhs.iter().peekable();
    /// #    let mut rit = rhs.iter().peekable();
    /// #    loop {
    /// #        match (lit.peek(), rit.peek()) {
    /// #            (None, _) => return true,
    /// #            (Some(_), None) => return false,
    /// #            (Some(a), Some(b)) => match a.cmp(b) {
    /// #                Ordering::Less => return false,
    /// #                Ordering::Equal => {
    /// #                    lit.next();
    /// #                    rit.next();
    /// #                }
    /// #                Ordering::Greater => {
    /// #                    rit.next();
    /// #                }
    /// #            },
    /// #        }
    /// #    }
    /// }
    ///
    /// let lhs: [&[_]; 5] = [&[1], &[2], &[3], &[4], &[5]];
    /// let rhs: [&[_]; 5] = [&[1], &[2], &[3], &[4], &[5]];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp_by(compare_sets);
    /// // The comparison is [Equal, Equal, Equal, Equal, Equal] -> Equal
    /// assert_eq!(ordering, Some(Ordering::Equal));
    ///
    /// let lhs: [&[_]; 5] = [&[1], &[2, 7], &[3], &[4], &[5]];
    /// let rhs: [&[_]; 5] = [&[1, 6], &[2], &[8], &[9], &[10]];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp_by(compare_sets);
    /// // The comparison is [Less, Greater, None, None, None] -> Less
    /// assert_eq!(ordering, Some(Ordering::Less));
    ///
    /// let lhs: [&[_]; 5] = [&[1], &[2], &[3], &[4], &[5]];
    /// let rhs: [&[_]; 5] = [&[6], &[2], &[3], &[4], &[5]];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp_by(compare_sets);
    /// // The comparison is [None, Equal, Equal, Equal, Equal] -> None
    /// assert_eq!(ordering, None);
    /// ```
    fn partial_cmp_by<T, U, F>(self, f: F) -> Option<Ordering>
    where
        Self: ParallelIterator<Item = (T, U)>,
        F: Fn(T, U) -> Option<Ordering> + Sync,
    {
        self.map(|(t, u)| f(t, u))
            .find_first(|&ordering| ordering != Some(Ordering::Equal))
            .unwrap_or(Some(Ordering::Equal))
    }

    /// Compare the pairs of items produced by this iterator using
    /// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order)
    /// order, after mapping them with `f`.
    ///
    /// See also the other comparison adaptors such as
    /// [`partial_cmp_by_keys()`](Self::partial_cmp_by_keys),
    /// [`cmp_by_key()`](Self::cmp_by_key) or [`eq_by_key()`](Self::eq_by_key).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [
    ///     (1.0, "Lorem"),
    ///     (2.0, "ipsum"),
    ///     (f64::NAN, "dolor"),
    ///     (4.0, "sit"),
    ///     (5.0, "amet"),
    /// ];
    /// let rhs = [
    ///     (5.0, "Lorem"),
    ///     (1.0, "ipsum"),
    ///     (2.0, "dolor"),
    ///     (3.0, "sit"),
    ///     (f64::NAN, "amet"),
    /// ];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp_by_key(|x| x.0);
    /// assert_eq!(ordering, Some(Ordering::Less));
    /// ```
    fn partial_cmp_by_key<T, A, F>(self, f: F) -> Option<Ordering>
    where
        Self: ParallelIterator<Item = (T, T)>,
        F: Fn(T) -> A + Sync,
        A: PartialOrd,
    {
        self.partial_cmp_by(|x, y| f(x).partial_cmp(&f(y)))
    }

    /// Compare the pairs of items produced by this iterator using
    /// [lexicographic](https://en.wikipedia.org/wiki/Lexicographic_order)
    /// order, after mapping them with `f` and `g`.
    ///
    /// See also the other comparison adaptors such as
    /// [`partial_cmp_by_key()`](Self::partial_cmp_by_key),
    /// [`cmp_by_keys()`](Self::cmp_by_keys) or
    /// [`eq_by_keys()`](Self::eq_by_keys).
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::cmp::Ordering;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let lhs = [(1, 1.0), (2, 2.0), (3, 3.0), (4, f64::NAN), (5, 5.0)];
    /// let rhs = [
    ///     (5.0, "Lorem"),
    ///     (1.0, "ipsum"),
    ///     (f64::NAN, "dolor"),
    ///     (3.0, "sit"),
    ///     (4.0, "amet"),
    /// ];
    /// let ordering = (lhs.par_iter(), rhs.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .partial_cmp_by_keys(|x| x.1, |y| y.0);
    /// assert_eq!(ordering, Some(Ordering::Less));
    /// ```
    fn partial_cmp_by_keys<T, U, A, B, F, G>(self, f: F, g: G) -> Option<Ordering>
    where
        Self: ParallelIterator<Item = (T, U)>,
        F: Fn(T) -> A + Sync,
        G: Fn(U) -> B + Sync,
        A: PartialOrd<B>,
    {
        self.partial_cmp_by(|t, u| f(t).partial_cmp(&g(u)))
    }

    /// Returns the product of the items produced by this iterator.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let product = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .product::<i32>();
    /// assert_eq!(product, 3628800);
    /// ```
    fn product<T>(self) -> T
    where
        T: Product<Self::Item> + Product<T> + Send,
    {
        self.iter_pipeline::<T, T>(ProductAccumulator, ProductAccumulator)
    }

    /// Reduces the items produced by this iterator into a single item, using
    /// `f` to collapse pairs of items.
    ///
    /// See also [`try_reduce()`](Self::try_reduce).
    ///
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
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
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
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let sum = []
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 0);
    /// ```
    fn reduce<Init, F>(self, init: Init, f: F) -> Self::Item
    where
        Init: Fn() -> Self::Item + Sync,
        F: Fn(Self::Item, Self::Item) -> Self::Item + Sync,
        Self::Item: Send,
    {
        self.pipeline(init, &f, |acc| acc, &f)
    }

    /// Returns the sum of the items produced by this iterator.
    ///
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
    fn sum<T>(self) -> T
    where
        T: Sum<Self::Item> + Sum<T> + Send,
    {
        self.iter_pipeline::<T, T>(SumAccumulator, SumAccumulator)
    }

    /// Try collecting items from this iterator into a per-thread collection of
    /// type `C`, then aggregating these into a `Vec<C>` without further
    /// flattening, breaking early and returning the failure if any item
    /// contains a failure.
    ///
    /// See also [`collect_per_thread()`](Self::collect_per_thread) and
    /// [`fold_per_thread()`](Self::fold_per_thread).
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] items. Items of arbitrary [`Try`] types are only available on
    /// Rust nightly with the `nightly` feature of Paralight enabled. This
    /// is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) and
    /// [`try_trait_v2_residual`](https://github.com/rust-lang/rust/issues/91285)
    /// nightly Rust features.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Ok(1), Ok(2), Ok(3), Ok(4), Ok(5), Ok(6), Ok(7), Ok(8)];
    /// let collection: Result<Vec<Vec<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_collect_per_thread();
    /// assert!(collection.is_ok());
    ///
    /// let mut values: Vec<i32> = collection.unwrap().into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Ok(1), Err(2), Ok(3), Err(4), Ok(5), Err(6), Ok(7), Err(8)];
    /// let collection: Result<Vec<Vec<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_collect_per_thread();
    /// assert!(collection.is_err());
    /// assert!(collection.unwrap_err() % 2 == 0);
    /// ```
    ///
    /// The outer type of the result is currently always [`Vec`], but the inner
    /// type can be anything that implements [`FromIterator`].
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Ok(1), Ok(2), Ok(3), Ok(4), Ok(5), Ok(6), Ok(7), Ok(8)];
    /// let collection: Result<Vec<HashSet<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_collect_per_thread();
    /// assert!(collection.is_ok());
    ///
    /// let mut values: Vec<i32> = collection.unwrap().into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    #[cfg(not(feature = "nightly"))]
    fn try_collect_per_thread<T, E, C>(self) -> Result<Vec<C>, E>
    where
        Self: ParallelIterator<Item = Result<T, E>>,
        C: FromIterator<T> + Send,
        E: Send,
    {
        self.iter_pipeline(
            TryIterCollector::<C> {
                fuse: Fuse::new(),
                _phantom: PhantomData,
            },
            TryIterFolder {
                init: |num_threads| Vec::with_capacity(num_threads),
                try_fold: |mut vec: Vec<C>, c: Result<C, E>| {
                    c.map(|c| {
                        vec.push(c);
                        vec
                    })
                },
            },
        )
    }

    /// Try collecting items from this iterator into a per-thread collection of
    /// type `C`, then aggregating these into a `Vec<C>` without further
    /// flattening, breaking early and returning the failure if any item
    /// contains a failure.
    ///
    /// See also [`collect_per_thread()`](Self::collect_per_thread) and
    /// [`fold_per_thread()`](Self::fold_per_thread).
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] items. Items of arbitrary [`Try`] types are only available on
    /// Rust nightly with the `nightly` feature of Paralight enabled. This
    /// is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) and
    /// [`try_trait_v2_residual`](https://github.com/rust-lang/rust/issues/91285)
    /// nightly Rust features.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Ok(1), Ok(2), Ok(3), Ok(4), Ok(5), Ok(6), Ok(7), Ok(8)];
    /// let collection: Result<Vec<Vec<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_collect_per_thread();
    /// assert!(collection.is_ok());
    ///
    /// let mut values: Vec<i32> = collection.unwrap().into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Ok(1), Err(2), Ok(3), Err(4), Ok(5), Err(6), Ok(7), Err(8)];
    /// let collection: Result<Vec<Vec<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_collect_per_thread();
    /// assert!(collection.is_err());
    /// assert!(collection.unwrap_err() % 2 == 0);
    /// ```
    ///
    /// The outer type of the result is currently always [`Vec`], but the inner
    /// type can be anything that implements [`FromIterator`].
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Ok(1), Ok(2), Ok(3), Ok(4), Ok(5), Ok(6), Ok(7), Ok(8)];
    /// let collection: Result<Vec<HashSet<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_collect_per_thread();
    /// assert!(collection.is_ok());
    ///
    /// let mut values: Vec<i32> = collection.unwrap().into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    ///
    /// With the `nightly` feature on a nightly compiler:
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)];
    /// let collection: Option<Vec<Vec<i32>>> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_collect_per_thread();
    /// assert!(collection.is_some());
    ///
    /// let mut values: Vec<i32> = collection.unwrap().into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6]);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Some(1), None, Some(3), None, Some(5), None];
    /// let collection: Option<Vec<Vec<i32>>> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_collect_per_thread();
    /// assert_eq!(collection, None);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)];
    /// let collection: Option<Vec<HashSet<i32>>> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_collect_per_thread();
    /// assert!(collection.is_some());
    ///
    /// let mut values: Vec<i32> = collection.unwrap().into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6]);
    /// ```
    #[cfg(feature = "nightly")]
    fn try_collect_per_thread<C>(
        self,
    ) -> <<Self::Item as Try>::Residual as Residual<Vec<C>>>::TryType
    where
        // ~ Result<T, E>
        Self::Item: Try,
        // ~ Result<!, E>: Result<C, E> + Result<Vec<C>, E>
        <Self::Item as Try>::Residual: Residual<C> + Residual<Vec<C>>,
        // ~ C: FromIterator<T> + Send
        C: FromIterator<<Self::Item as Try>::Output> + Send,
        // ~ Result<C, E>: Send
        <<Self::Item as Try>::Residual as Residual<C>>::TryType: Send,
    {
        // Helper function instead of a lambda to avoid overlong type annotations.
        #[inline(always)]
        fn try_fold<C, R>(
            mut vec: Vec<C>,
            c: <R as Residual<C>>::TryType,
        ) -> <R as Residual<Vec<C>>>::TryType
        where
            R: Residual<C> + Residual<Vec<C>>,
        {
            match c.branch() {
                ControlFlow::Continue(c) => {
                    vec.push(c);
                    Try::from_output(vec)
                }
                ControlFlow::Break(e) => FromResidual::from_residual(e),
            }
        }

        self.iter_pipeline(
            TryIterCollector::<C> {
                fuse: Fuse::new(),
                _phantom: PhantomData,
            },
            TryIterFolder {
                init: |num_threads| Vec::with_capacity(num_threads),
                try_fold: try_fold::<C, <Self::Item as Try>::Residual>,
            },
        )
    }

    /// Try folding items of this parallel iterator into a per-thread
    /// accumulator of type `T`, then folds again the per-thread results
    /// into an output of type `U`, breaking early and returning the failure
    /// if any operation returns a failure.
    ///
    /// See also [`fold_per_thread()`](Self::fold_per_thread).
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] outputs. Outputs of arbitrary [`Try`] types are only
    /// available on Rust nightly with the `nightly` feature of Paralight
    /// enabled. This is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) nightly
    /// Rust feature.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input: [Result<i32, i32>; 8] = [Ok(1), Ok(2), Ok(3), Ok(4), Ok(5), Ok(6), Ok(7), Ok(8)];
    /// let collection: Result<Vec<Vec<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_fold_per_thread(
    ///         Vec::new,
    ///         |mut vec, x| -> Result<Vec<i32>, i32> {
    ///             vec.push(x?);
    ///             Ok(vec)
    ///         },
    ///         Vec::with_capacity,
    ///         |mut vecvec, vec| {
    ///             vecvec.push(vec?);
    ///             Ok(vecvec)
    ///         },
    ///     );
    /// assert!(collection.is_ok());
    ///
    /// let mut values: Vec<i32> = collection.unwrap().into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Ok(1), Ok(2), Ok(3), Ok(4), Err(5), Ok(6), Ok(7), Ok(8)];
    /// let collection: Result<Vec<Vec<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_fold_per_thread(
    ///         Vec::new,
    ///         |mut vec, x| -> Result<Vec<i32>, i32> {
    ///             vec.push(x?);
    ///             Ok(vec)
    ///         },
    ///         Vec::with_capacity,
    ///         |mut vecvec, vec| {
    ///             vecvec.push(vec?);
    ///             Ok(vecvec)
    ///         },
    ///     );
    /// assert_eq!(collection, Err(5));
    /// ```
    #[cfg(not(feature = "nightly"))]
    fn try_fold_per_thread<T, U, E, InitPerThread, TryFoldPerThread, InitFinal, TryFoldFinal>(
        self,
        init_per_thread: InitPerThread,
        try_fold_per_thread: TryFoldPerThread,
        init_final: InitFinal,
        try_fold_final: TryFoldFinal,
    ) -> Result<U, E>
    where
        InitPerThread: Fn() -> T + Sync,
        TryFoldPerThread: Fn(T, Self::Item) -> Result<T, E> + Sync,
        InitFinal: Fn(usize) -> U,
        TryFoldFinal: Fn(U, Result<T, E>) -> Result<U, E>,
        T: Send,
        E: Send,
    {
        self.iter_pipeline(
            ShortCircuitingAccumulator {
                fuse: Fuse::new(),
                init: init_per_thread,
                process_item: |acc, item| match try_fold_per_thread(acc, item) {
                    Ok(x) => ControlFlow::Continue(x),
                    Err(e) => ControlFlow::Break(e),
                },
                finalize: |acc| match acc {
                    ControlFlow::Continue(x) => Ok(x),
                    ControlFlow::Break(e) => Err(e),
                },
            },
            TryIterFolder {
                init: init_final,
                try_fold: try_fold_final,
            },
        )
    }

    /// Try folding items of this parallel iterator into a per-thread
    /// accumulator of type `T`, then folds again the per-thread results
    /// into an output of type `U`, breaking early and returning the failure
    /// if any operation returns a failure.
    ///
    /// See also [`fold_per_thread()`](Self::fold_per_thread).
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] outputs. Outputs of arbitrary [`Try`] types are only
    /// available on Rust nightly with the `nightly` feature of Paralight
    /// enabled. This is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) nightly
    /// Rust feature.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input: [Result<i32, i32>; 8] = [Ok(1), Ok(2), Ok(3), Ok(4), Ok(5), Ok(6), Ok(7), Ok(8)];
    /// let collection: Result<Vec<Vec<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_fold_per_thread(
    ///         Vec::new,
    ///         |mut vec, x| -> Result<Vec<i32>, i32> {
    ///             vec.push(x?);
    ///             Ok(vec)
    ///         },
    ///         Vec::with_capacity,
    ///         |mut vecvec, vec| {
    ///             vecvec.push(vec?);
    ///             Ok(vecvec)
    ///         },
    ///     );
    /// assert!(collection.is_ok());
    ///
    /// let mut values: Vec<i32> = collection.unwrap().into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6, 7, 8]);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Ok(1), Ok(2), Ok(3), Ok(4), Err(5), Ok(6), Ok(7), Ok(8)];
    /// let collection: Result<Vec<Vec<i32>>, i32> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_fold_per_thread(
    ///         Vec::new,
    ///         |mut vec, x| -> Result<Vec<i32>, i32> {
    ///             vec.push(x?);
    ///             Ok(vec)
    ///         },
    ///         Vec::with_capacity,
    ///         |mut vecvec, vec| {
    ///             vecvec.push(vec?);
    ///             Ok(vecvec)
    ///         },
    ///     );
    /// assert_eq!(collection, Err(5));
    /// ```
    ///
    /// With the `nightly` feature on a nightly compiler:
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Some(1), Some(2), Some(3), Some(4), Some(5), Some(6)];
    /// let collection: Option<Vec<Vec<i32>>> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_fold_per_thread(
    ///         Vec::new,
    ///         |mut vec, x| {
    ///             vec.push(x?);
    ///             Some(vec)
    ///         },
    ///         Vec::with_capacity,
    ///         |mut vecvec, vec| {
    ///             vecvec.push(vec?);
    ///             Some(vecvec)
    ///         },
    ///     );
    /// assert!(collection.is_some());
    ///
    /// let mut values: Vec<i32> = collection.unwrap().into_iter().flatten().collect();
    /// values.sort_unstable();
    /// assert_eq!(values, &[1, 2, 3, 4, 5, 6]);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [Some(1), Some(2), Some(3), Some(4), None, Some(6)];
    /// let collection: Option<Vec<Vec<i32>>> = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_fold_per_thread(
    ///         Vec::new,
    ///         |mut vec, x| {
    ///             vec.push(x?);
    ///             Some(vec)
    ///         },
    ///         Vec::with_capacity,
    ///         |mut vecvec, vec| {
    ///             vecvec.push(vec?);
    ///             Some(vecvec)
    ///         },
    ///     );
    /// assert_eq!(collection, None);
    /// ```
    #[cfg(feature = "nightly")]
    fn try_fold_per_thread<T, U, R, S, InitPerThread, TryFoldPerThread, InitFinal, TryFoldFinal>(
        self,
        init_per_thread: InitPerThread,
        try_fold_per_thread: TryFoldPerThread,
        init_final: InitFinal,
        try_fold_final: TryFoldFinal,
    ) -> S
    where
        InitPerThread: Fn() -> T + Sync,
        TryFoldPerThread: Fn(T, Self::Item) -> R + Sync,
        InitFinal: Fn(usize) -> U,
        TryFoldFinal: Fn(U, R) -> S,
        R: Try<Output = T> + Send,
        S: Try<Output = U>,
    {
        self.iter_pipeline(
            ShortCircuitingAccumulator {
                fuse: Fuse::new(),
                init: init_per_thread,
                process_item: try_fold_per_thread,
                finalize: |acc| acc,
            },
            TryIterFolder {
                init: init_final,
                try_fold: try_fold_final,
            },
        )
    }

    /// Runs the fallible function `f` on each item of this parallel iterator,
    /// breaking early and returning the failure in case of failure.
    ///
    /// If multiple items cause a failure, an arbitrary one is returned.
    ///
    /// See also [`try_for_each_init()`](Self::try_for_each_init) if you need to
    /// initialize a per-thread value and pass it together with each item.
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] outputs. Outputs of arbitrary [`Try`] types are only
    /// available on Rust nightly with the `nightly` feature of Paralight
    /// enabled. This is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) nightly
    /// Rust feature.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let set = Mutex::new(HashSet::new());
    /// let result = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each(|&x| {
    ///         if set.lock().unwrap().insert(x) {
    ///             Ok(())
    ///         } else {
    ///             Err(x)
    ///         }
    ///     });
    /// assert_eq!(result, Ok(()));
    /// assert_eq!(set.into_inner().unwrap(), (1..=10).collect());
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 1, 1, 1, 1, 6, 7, 8, 9, 10];
    /// let set = Mutex::new(HashSet::new());
    /// let result = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each(|&x| {
    ///         if set.lock().unwrap().insert(x) {
    ///             Ok(())
    ///         } else {
    ///             Err(x)
    ///         }
    ///     });
    /// assert_eq!(result, Err(1));
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let result = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each(|&x| Err(x));
    /// assert!(result.is_err());
    /// ```
    #[cfg(not(feature = "nightly"))]
    fn try_for_each<E, F>(self, f: F) -> Result<(), E>
    where
        F: Fn(Self::Item) -> Result<(), E> + Sync,
        E: Send,
    {
        self.short_circuiting_pipeline(
            || (),
            |_, item| match f(item) {
                Ok(()) => ControlFlow::Continue(()),
                Err(e) => ControlFlow::Break(e),
            },
            |result| match result {
                ControlFlow::Continue(()) => Ok(()),
                ControlFlow::Break(e) => Err(e),
            },
            |x, y| x.and(y),
        )
    }

    /// Runs the fallible function `f` on each item of this parallel iterator,
    /// breaking early and returning the failure in case of failure.
    ///
    /// If multiple items cause a failure, an arbitrary one is returned.
    ///
    /// See also [`try_for_each_init()`](Self::try_for_each_init) if you need to
    /// initialize a per-thread value and pass it together with each item.
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] outputs. Outputs of arbitrary [`Try`] types are only
    /// available on Rust nightly with the `nightly` feature of Paralight
    /// enabled. This is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) nightly
    /// Rust feature.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let set = Mutex::new(HashSet::new());
    /// let result = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each(|&x| {
    ///         if set.lock().unwrap().insert(x) {
    ///             Ok(())
    ///         } else {
    ///             Err(x)
    ///         }
    ///     });
    /// assert_eq!(result, Ok(()));
    /// assert_eq!(set.into_inner().unwrap(), (1..=10).collect());
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 1, 1, 1, 1, 6, 7, 8, 9, 10];
    /// let set = Mutex::new(HashSet::new());
    /// let result = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each(|&x| {
    ///         if set.lock().unwrap().insert(x) {
    ///             Ok(())
    ///         } else {
    ///             Err(x)
    ///         }
    ///     });
    /// assert_eq!(result, Err(1));
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let result = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each(|&x| Err(x));
    /// assert!(result.is_err());
    /// ```
    ///
    /// With the `nightly` feature on a nightly compiler:
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let set = Mutex::new(HashSet::new());
    /// let result = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each(|&x| {
    ///         if set.lock().unwrap().insert(x) {
    ///             Some(())
    ///         } else {
    ///             None
    ///         }
    ///     });
    /// assert_eq!(result, Some(()));
    /// assert_eq!(set.into_inner().unwrap(), (1..=10).collect());
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 1, 1, 1, 1, 6, 7, 8, 9, 10];
    /// let set = Mutex::new(HashSet::new());
    /// let result = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each(|&x| {
    ///         if set.lock().unwrap().insert(x) {
    ///             Some(())
    ///         } else {
    ///             None
    ///         }
    ///     });
    /// assert_eq!(result, None);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let result = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each(|_| None);
    /// assert!(result.is_none());
    /// ```
    #[cfg(feature = "nightly")]
    fn try_for_each<R, F>(self, f: F) -> R
    where
        F: Fn(Self::Item) -> R + Sync,
        R: Try<Output = ()> + Send,
    {
        self.try_short_circuiting_pipeline(
            || (),
            |_, item| f(item),
            |result| result,
            |x, y| match (x.branch(), y.branch()) {
                (ControlFlow::Continue(()), ControlFlow::Continue(())) => R::from_output(()),
                (ControlFlow::Break(e), _) | (_, ControlFlow::Break(e)) => R::from_residual(e),
            },
        )
    }

    /// Runs the fallible function `f` on each item of this parallel iterator,
    /// together with a per-thread mutable value returned by `init`,
    /// breaking early and returning the failure in case of failure.
    ///
    /// If multiple items cause a failure, an arbitrary one is returned.
    ///
    /// The `init` function will be called only once per worker thread. The
    /// companion value returned by `init` doesn't need to be [`Send`] nor
    /// [`Sync`].
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] outputs. Outputs of arbitrary [`Try`] types are only
    /// available on Rust nightly with the `nightly` feature of Paralight
    /// enabled. This is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) nightly
    /// Rust feature.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// use rand::Rng;
    ///
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let set = Mutex::new(HashSet::new());
    /// let result = (0..257)
    ///     .into_par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each_init(
    ///         rand::rng, // A thread-local RNG that is neither Send nor Sync.
    ///         |rng, _| {
    ///             let byte: u8 = rng.random();
    ///             if set.lock().unwrap().insert(byte) {
    ///                 Ok(())
    ///             } else {
    ///                 Err(byte)
    ///             }
    ///         },
    ///     );
    /// // Even a biased random number generator cannot yield more than 256 distinct bytes.
    /// assert!(result.is_err());
    /// ```
    #[cfg(not(feature = "nightly"))]
    fn try_for_each_init<T, Init, E, F>(self, init: Init, f: F) -> Result<(), E>
    where
        Init: Fn() -> T + Sync,
        F: Fn(&mut T, Self::Item) -> Result<(), E> + Sync,
        E: Send,
    {
        self.short_circuiting_pipeline(
            init,
            |mut t, item| match f(&mut t, item) {
                Ok(()) => ControlFlow::Continue(t),
                Err(e) => ControlFlow::Break(e),
            },
            |result| match result {
                ControlFlow::Continue(_) => Ok(()),
                ControlFlow::Break(e) => Err(e),
            },
            |x, y| x.and(y),
        )
    }

    /// Runs the fallible function `f` on each item of this parallel iterator,
    /// together with a per-thread mutable value returned by `init`,
    /// breaking early and returning the failure in case of failure.
    ///
    /// If multiple items cause a failure, an arbitrary one is returned.
    ///
    /// The `init` function will be called only once per worker thread. The
    /// companion value returned by `init` doesn't need to be [`Send`] nor
    /// [`Sync`].
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] outputs. Outputs of arbitrary [`Try`] types are only
    /// available on Rust nightly with the `nightly` feature of Paralight
    /// enabled. This is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) nightly
    /// Rust feature.
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// use rand::Rng;
    ///
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let set = Mutex::new(HashSet::new());
    /// let result = (0..257)
    ///     .into_par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each_init(
    ///         rand::rng, // A thread-local RNG that is neither Send nor Sync.
    ///         |rng, _| {
    ///             let byte: u8 = rng.random();
    ///             if set.lock().unwrap().insert(byte) {
    ///                 Ok(())
    ///             } else {
    ///                 Err(byte)
    ///             }
    ///         },
    ///     );
    /// // Even a biased random number generator cannot yield more than 256 distinct bytes.
    /// assert!(result.is_err());
    /// ```
    ///
    /// With the `nightly` feature on a nightly compiler:
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// use rand::Rng;
    ///
    /// # use std::collections::HashSet;
    /// # use std::sync::Mutex;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let set = Mutex::new(HashSet::new());
    /// let result = (0..257)
    ///     .into_par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_for_each_init(
    ///         rand::rng, // A thread-local RNG that is neither Send nor Sync.
    ///         |rng, _| {
    ///             let byte: u8 = rng.random();
    ///             if set.lock().unwrap().insert(byte) {
    ///                 Some(())
    ///             } else {
    ///                 None
    ///             }
    ///         },
    ///     );
    /// // Even a biased random number generator cannot yield more than 256 distinct bytes.
    /// assert!(result.is_none());
    /// ```
    #[cfg(feature = "nightly")]
    fn try_for_each_init<T, Init, R, F>(self, init: Init, f: F) -> R
    where
        Init: Fn() -> T + Sync,
        F: Fn(&mut T, Self::Item) -> R + Sync,
        R: Try<Output = ()> + Send,
    {
        self.short_circuiting_pipeline(
            init,
            // TODO(MSRV >= 1.83.0): Use ControlFlow::map_continue().
            |mut t, item| match f(&mut t, item).branch() {
                ControlFlow::Continue(()) => ControlFlow::Continue(t),
                ControlFlow::Break(e) => ControlFlow::Break(e),
            },
            |result| match result {
                ControlFlow::Continue(_) => R::from_output(()),
                ControlFlow::Break(e) => R::from_residual(e),
            },
            |x, y| match (x.branch(), y.branch()) {
                (ControlFlow::Continue(()), ControlFlow::Continue(())) => R::from_output(()),
                (ControlFlow::Break(e), _) | (_, ControlFlow::Break(e)) => R::from_residual(e),
            },
        )
    }

    /// Try reducing the items produced by this iterator into a single item,
    /// using `f` to collapse pairs of items, breaking early and returning
    /// the failure if `f` returns a failure.
    ///
    /// If multiple failures happen, an arbitrary one is returned.
    ///
    /// See also [`reduce()`](Self::reduce).
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] outputs. Outputs of arbitrary [`Try`] types are only
    /// available on Rust nightly with the `nightly` feature of Paralight
    /// enabled. This is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) nightly
    /// Rust feature.
    ///
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
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_reduce(|| 0i32, |x, y| x.checked_add(y).ok_or("error"));
    /// assert_eq!(sum, Ok(5 * 11));
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, i32::MAX, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_reduce(|| 0i32, |x, y| x.checked_add(y).ok_or("error"));
    /// assert_eq!(sum, Err("error"));
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let sum = []
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_reduce(|| 0i32, |x, y| x.checked_add(y).ok_or("error"));
    /// assert_eq!(sum, Ok(0));
    /// ```
    #[cfg(not(feature = "nightly"))]
    fn try_reduce<E, Init, F>(self, init: Init, f: F) -> Result<Self::Item, E>
    where
        Init: Fn() -> Self::Item + Sync,
        F: Fn(Self::Item, Self::Item) -> Result<Self::Item, E> + Sync,
        Self::Item: Send,
        E: Send,
    {
        self.short_circuiting_pipeline(
            init,
            |a, b| match f(a, b) {
                Ok(x) => ControlFlow::Continue(x),
                Err(e) => ControlFlow::Break(e),
            },
            |acc| match acc {
                ControlFlow::Continue(x) => Ok(x),
                ControlFlow::Break(e) => Err(e),
            },
            |x, y| match (x, y) {
                (Ok(a), Ok(b)) => f(a, b),
                (Err(e), _) | (_, Err(e)) => Err(e),
            },
        )
    }

    /// Try reducing the items produced by this iterator into a single item,
    /// using `f` to collapse pairs of items, breaking early and returning
    /// the failure if `f` returns a failure.
    ///
    /// If multiple failures happen, an arbitrary one is returned.
    ///
    /// See also [`reduce()`](Self::reduce).
    ///
    /// # Stability blockers
    ///
    /// On stable Rust, this adaptor is currently only implemented for
    /// [`Result`] outputs. Outputs of arbitrary [`Try`] types are only
    /// available on Rust nightly with the `nightly` feature of Paralight
    /// enabled. This is because the implementation depends on the
    /// [`try_trait_v2`](https://github.com/rust-lang/rust/issues/84277) nightly
    /// Rust feature.
    ///
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
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_reduce(|| 0i32, |x, y| x.checked_add(y).ok_or("error"));
    /// assert_eq!(sum, Ok(5 * 11));
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, i32::MAX, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_reduce(|| 0i32, |x, y| x.checked_add(y).ok_or("error"));
    /// assert_eq!(sum, Err("error"));
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let sum = []
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_reduce(|| 0i32, |x, y| x.checked_add(y).ok_or("error"));
    /// assert_eq!(sum, Ok(0));
    /// ```
    ///
    /// With the `nightly` feature on a nightly compiler:
    ///
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
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_reduce(|| 0i32, |x, y| x.checked_add(y));
    /// assert_eq!(sum, Some(5 * 11));
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, i32::MAX, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_reduce(|| 0i32, |x, y| x.checked_add(y));
    /// assert_eq!(sum, None);
    /// ```
    ///
    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let sum = []
    ///     .par_iter()
    ///     .copied()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .try_reduce(|| 0i32, |x, y| x.checked_add(y));
    /// assert_eq!(sum, Some(0));
    /// ```
    #[cfg(feature = "nightly")]
    fn try_reduce<R, Init, F>(self, init: Init, f: F) -> R
    where
        Init: Fn() -> Self::Item + Sync,
        F: Fn(Self::Item, Self::Item) -> R + Sync,
        R: Try<Output = Self::Item> + Send,
    {
        self.try_short_circuiting_pipeline(
            init,
            &f,
            |acc| acc,
            |x, y| match (x.branch(), y.branch()) {
                (ControlFlow::Continue(a), ControlFlow::Continue(b)) => f(a, b),
                (ControlFlow::Break(e), _) | (_, ControlFlow::Break(e)) => R::from_residual(e),
            },
        )
    }
}
