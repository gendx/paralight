// Copyright 2024-2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Iterator adaptors to define parallel pipelines more conveniently.

mod source;

use crossbeam_utils::CachePadded;
#[cfg(feature = "nightly")]
pub use source::array::ArrayParallelSource;
pub use source::range::{RangeInclusiveParallelSource, RangeParallelSource};
pub use source::slice::{MutSliceParallelSource, SliceParallelSource};
pub use source::vec::VecParallelSource;
pub use source::vec_deque::{VecDequeRefMutParallelSource, VecDequeRefParallelSource};
pub use source::zip::{ZipEq, ZipMax, ZipMin, ZipableSource};
pub use source::{
    BaseParallelIterator, IntoParallelRefMutSource, IntoParallelRefSource, IntoParallelSource,
    ParallelSource, ParallelSourceExt, SourceCleanup, SourceDescriptor,
};
use std::cmp::Ordering;
use std::iter::{Product, Sum};
use std::ops::ControlFlow;
#[cfg(feature = "nightly")]
use std::ops::Try;
use std::sync::atomic::AtomicBool;

/// Interface for an operation that accumulates items from an iterator into an
/// output.
///
/// You can think of it as a variant of `Fn(impl Iterator) -> Output` made
/// generic over the item and output types.
pub trait Accumulator<Item, Output> {
    /// Accumulates the items from the given iterator into an output.
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output;
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
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIterator, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
                init,
                process_item,
                finalize,
            },
            IterReducer { reduce },
        )
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIterator, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
                init,
                process_item,
                finalize,
            },
            IterReducer { reduce },
        )
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIterator, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     Accumulator, IntoParallelRefSource, ParallelIterator, ParallelSourceExt,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// ```
    fn iter_pipeline<Output: Send>(
        self,
        accum: impl Accumulator<Self::Item, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
    ) -> Output;
}

struct IterReducer<Reduce> {
    reduce: Reduce,
}

impl<Output, Reduce> Accumulator<Output, Output> for IterReducer<Reduce>
where
    Reduce: Fn(Output, Output) -> Output,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Output>) -> Output {
        iter.reduce(&self.reduce).unwrap()
    }
}

struct IterAccumulator<Init, ProcessItem, Finalize> {
    init: Init,
    process_item: ProcessItem,
    finalize: Finalize,
}

impl<Item, Accum, Output, Init, ProcessItem, Finalize> Accumulator<Item, Output>
    for IterAccumulator<Init, ProcessItem, Finalize>
where
    Init: Fn() -> Accum,
    ProcessItem: Fn(Accum, Item) -> Accum,
    Finalize: Fn(Accum) -> Output,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output {
        let mut accumulator = (self.init)();
        for item in iter {
            accumulator = (self.process_item)(accumulator, item);
        }
        (self.finalize)(accumulator)
    }
}

struct ShortCircuitingAccumulator<Init, ProcessItem, Finalize> {
    fuse: Fuse,
    init: Init,
    process_item: ProcessItem,
    finalize: Finalize,
}

impl<Item, Accum, Break, Output, Init, ProcessItem, Finalize> Accumulator<Item, Output>
    for ShortCircuitingAccumulator<Init, ProcessItem, Finalize>
where
    Init: Fn() -> Accum,
    ProcessItem: Fn(Accum, Item) -> ControlFlow<Break, Accum>,
    Finalize: Fn(ControlFlow<Break, Accum>) -> Output,
{
    #[inline(always)]
    fn accumulate(&self, mut iter: impl Iterator<Item = Item>) -> Output {
        let mut accumulator = (self.init)();
        let result = 'outer: {
            while let FuseState::Unset = self.fuse.load() {
                let Some(item) = iter.next() else {
                    break;
                };

                match (self.process_item)(accumulator, item) {
                    ControlFlow::Continue(acc) => {
                        accumulator = acc;
                        continue;
                    }
                    control_flow @ ControlFlow::Break(_) => {
                        self.fuse.set();
                        break 'outer control_flow;
                    }
                }
            }
            ControlFlow::Continue(accumulator)
        };
        (self.finalize)(result)
    }
}

/// A fuse is an atomic object that starts unset and can transition once to the
/// set state.
///
/// Under the hood, this contains an atomic boolean aligned to a cache line to
/// avoid any risk of false sharing performance overhead.
struct Fuse(CachePadded<AtomicBool>);

/// State of a [`Fuse`].
enum FuseState {
    Unset,
    Set,
}

impl Fuse {
    /// Creates a new fuse in the [`Unset`](FuseState::Unset) state.
    fn new() -> Self {
        Fuse(CachePadded::new(AtomicBool::new(false)))
    }

    /// Reads the current state of this fuse.
    fn load(&self) -> FuseState {
        use std::sync::atomic::Ordering;

        match self.0.load(Ordering::Relaxed) {
            false => FuseState::Unset,
            true => FuseState::Set,
        }
    }

    /// Sets this fuse to the [`Set`](FuseState::Set) state.
    fn set(&self) {
        use std::sync::atomic::Ordering;

        self.0.store(true, Ordering::Relaxed)
    }
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

struct AdaptorAccumulator<Inner, TransformItem> {
    inner: Inner,
    transform_item: TransformItem,
}

impl<InnerItem, Item, Output, Inner, TransformItem> Accumulator<InnerItem, Output>
    for AdaptorAccumulator<Inner, TransformItem>
where
    Inner: Accumulator<Item, Output>,
    TransformItem: Fn(InnerItem) -> Option<Item>,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = InnerItem>) -> Output {
        self.inner.accumulate(iter.filter_map(&self.transform_item))
    }
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

    fn iter_pipeline<Output: Send>(
        self,
        accum: impl Accumulator<Self::Item, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
    ) -> Output {
        let descriptor = self.descriptor();
        let accumulator = AdaptorAccumulator {
            inner: accum,
            transform_item: descriptor.transform_item,
        };
        descriptor.inner.iter_pipeline(accumulator, reduce)
    }
}

/// Additional methods provided for types that implement [`ParallelIterator`].
///
/// See also [`ParallelSourceExt`] for more adaptors that only apply to parallel
/// sources (earlier in the pipeline).
pub trait ParallelIteratorExt: ParallelIterator {
    /// Returns [`true`] if all items produced by this iterator satisfy the
    /// predicate `f`, and [`false`] otherwise.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
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
    /// ```
    fn cloned<'a, T>(self) -> Cloned<Self>
    where
        Self: ParallelIterator<Item = &'a T>,
        T: Clone + 'a,
    {
        Cloned { inner: self }
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn copied<'a, T>(self) -> Copied<Self>
    where
        Self: ParallelIterator<Item = &'a T>,
        T: Copy + 'a,
    {
        Copied { inner: self }
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let pairs: [(i32, i32); 0] = [];
    /// let eq_empty = pairs
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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

    /// Returns a parallel iterator that produces only the items for which the
    /// predicate `f` returns [`true`].
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum_even = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .filter(|&&x| x % 2 == 0)
    ///     .sum::<i32>();
    /// assert_eq!(sum_even, 5 * 6);
    /// ```
    fn filter<F>(self, f: F) -> Filter<Self, F>
    where
        F: Fn(&Self::Item) -> bool + Sync,
    {
        Filter { inner: self, f }
    }

    /// Applies the function `f` to each item of this iterator, returning a
    /// parallel iterator that produces the mapped items `x` for which `f`
    /// returns [`Some(x)`](Option::Some) and skips the items for which `f`
    /// returns [`None`].
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    ///     .filter_map(|&x| if x != 2 { Some(x * 3) } else { None })
    ///     .sum::<i32>();
    /// assert_eq!(sum, 3 * (5 * 11 - 2));
    /// ```
    ///
    /// Mapping to a non-[`Send`] non-[`Sync`] type such as [`Rc`](std::rc::Rc)
    /// is fine.
    ///
    /// ```
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIterator, ParallelIteratorExt, ParallelSourceExt,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # use std::rc::Rc;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum_even = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .filter_map(|&x| if x % 2 == 0 { Some(Rc::new(x)) } else { None })
    ///     .pipeline(|| 0, |acc, x| acc + *x, |acc| acc, |a, b| a + b);
    /// assert_eq!(sum_even, 5 * 6);
    /// ```
    fn filter_map<T, F>(self, f: F) -> FilterMap<Self, F>
    where
        F: Fn(Self::Item) -> Option<T> + Sync,
    {
        FilterMap { inner: self, f }
    }

    /// Returns any item that satisfies the predicate `f`, or [`None`] if no
    /// item satisfies it.
    ///
    /// If multiple items satisfy `f`, an arbitrary one is returned.
    ///
    /// See also [`find_map_any()`](Self::find_map_any).
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
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
    /// See also [`find_map_first()`](Self::find_map_first).
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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

    /// Runs `f` on each item of this parallel iterator.
    ///
    /// See also [`for_each_init()`](Self::for_each_init) if you need to
    /// initialize a per-thread value and pass it together with each item.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefMutSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// use rand::Rng;
    ///
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let mut bits: [bool; 128] = [true; 128];
    /// bits
    ///     .par_iter_mut()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each_init(
    ///         rand::rng,
    ///         |rng, bit| if rng.random() { *bit = false; },
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

    /// Runs the function `f` on each item of this iterator in a pass-through
    /// manner, returning a parallel iterator producing the original items.
    ///
    /// This is useful to help debug intermediate stages of an iterator adaptor
    /// pipeline.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # use std::sync::atomic::{AtomicUsize, Ordering};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
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
    /// See also [`map_init()`](Self::map_init) if you need to initialize a
    /// per-thread value and pass it together with each item.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIterator, ParallelIteratorExt, ParallelSourceExt,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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

    /// Applies the function `f` to each item of this iterator, together with a
    /// per-thread mutable value returned by `init`, and returns a parallel
    /// iterator producing the mapped items.
    ///
    /// The `init` function will be called only once per worker thread. The
    /// companion value returned by `init` doesn't need to be [`Send`] nor
    /// [`Sync`].
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// use rand::Rng;
    ///
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let randomized_sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .map_init(
    ///         rand::rng,
    ///         |rng, &x| if rng.random() { x * 2 } else { x * 3 },
    ///     )
    ///     .sum::<i32>();
    ///
    /// assert!(randomized_sum >= 10 * 11);
    /// assert!(randomized_sum <= 15 * 11);
    /// ```
    fn map_init<I, Init, T, F>(self, init: Init, f: F) -> MapInit<Self, Init, F>
    where
        Init: Fn() -> I + Sync,
        F: Fn(&mut I, Self::Item) -> T + Sync,
    {
        MapInit {
            inner: self,
            init,
            f,
        }
    }

    /// Returns the maximal item of this iterator, or [`None`] if this iterator
    /// is empty.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let pairs: [(i32, i32); 0] = [];
    /// let ne_empty = pairs
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// #            (Some(a), Some(b)) => match a.cmp(&b) {
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{
    /// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    /// # };
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
        self.iter_pipeline(ProductAccumulator, ProductAccumulator)
    }

    /// Reduces the items produced by this iterator into a single item, using
    /// `f` to collapse pairs of items.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    ///     .copied()
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let sum = []
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .copied()
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
        self.iter_pipeline(SumAccumulator, SumAccumulator)
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    #[cfg(feature = "nightly")]
    fn try_for_each<R, F>(self, f: F) -> R
    where
        F: Fn(Self::Item) -> R + Sync,
        R: Try<Output = ()> + Send,
    {
        self.short_circuiting_pipeline(
            || (),
            |_, item| f(item).branch(),
            |result| match result {
                ControlFlow::Continue(()) => R::from_output(()),
                ControlFlow::Break(e) => R::from_residual(e),
            },
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
    /// # use paralight::iter::{IntoParallelSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    ///     .try_for_each_init(rand::rng, |rng, _| {
    ///         let byte: u8 = rng.random();
    ///         if set.lock().unwrap().insert(byte) {
    ///             Ok(())
    ///         } else {
    ///             Err(byte)
    ///         }
    ///     });
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
    /// # use paralight::iter::{IntoParallelSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
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
    ///     .try_for_each_init(rand::rng, |rng, _| {
    ///         let byte: u8 = rng.random();
    ///         if set.lock().unwrap().insert(byte) {
    ///             Ok(())
    ///         } else {
    ///             Err(byte)
    ///         }
    ///     });
    /// // Even a biased random number generator cannot yield more than 256 distinct bytes.
    /// assert!(result.is_err());
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
}

/// A thread pool backend that can process parallel iterators.
///
/// # Safety
///
/// This trait is marked as `unsafe`, because implementers **must**
/// ensure the safety guarantees of
/// [`GenericThreadPool::upper_bounded_pipeline`]
/// and [`GenericThreadPool::iter_pipeline`].
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
    fn iter_pipeline<Output: Send>(
        self,
        input_len: usize,
        accum: impl Accumulator<usize, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output;
}

struct SumAccumulator;

impl<Item, Output> Accumulator<Item, Output> for SumAccumulator
where
    Output: Sum<Item>,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output {
        iter.sum()
    }
}

struct ProductAccumulator;

impl<Item, Output> Accumulator<Item, Output> for ProductAccumulator
where
    Output: Product<Item>,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output {
        iter.product()
    }
}

impl<T: ParallelIterator> ParallelIteratorExt for T {}

/// This struct is created by the [`cloned()`](ParallelIteratorExt::cloned)
/// method on [`ParallelIteratorExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and [`ParallelIteratorExt`] traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Cloned<Inner: ParallelIterator> {
    inner: Inner,
}

impl<'a, T, Inner: ParallelIterator> ParallelAdaptor for Cloned<Inner>
where
    T: Clone + 'a,
    Inner: ParallelIterator<Item = &'a T>,
{
    type Item = T;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: |item| Some(item.clone()),
        }
    }
}

/// This struct is created by the [`copied()`](ParallelIteratorExt::copied)
/// method on [`ParallelIteratorExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and [`ParallelIteratorExt`] traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Copied<Inner: ParallelIterator> {
    inner: Inner,
}

impl<'a, T, Inner: ParallelIterator> ParallelAdaptor for Copied<Inner>
where
    T: Copy + 'a,
    Inner: ParallelIterator<Item = &'a T>,
{
    type Item = T;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: |item| Some(*item),
        }
    }
}

/// This struct is created by the [`filter()`](ParallelIteratorExt::filter)
/// method on [`ParallelIteratorExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and [`ParallelIteratorExt`] traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Filter<Inner: ParallelIterator, F> {
    inner: Inner,
    f: F,
}

impl<Inner: ParallelIterator, F> ParallelAdaptor for Filter<Inner, F>
where
    F: Fn(&Inner::Item) -> bool + Sync,
{
    type Item = Inner::Item;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: move |item| {
                if (self.f)(&item) {
                    Some(item)
                } else {
                    None
                }
            },
        }
    }
}

/// This struct is created by the
/// [`filter_map()`](ParallelIteratorExt::filter_map) method on
/// [`ParallelIteratorExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and [`ParallelIteratorExt`] traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct FilterMap<Inner: ParallelIterator, F> {
    inner: Inner,
    f: F,
}

impl<Inner: ParallelIterator, T, F> ParallelAdaptor for FilterMap<Inner, F>
where
    F: Fn(Inner::Item) -> Option<T> + Sync,
{
    type Item = T;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: self.f,
        }
    }
}

/// This struct is created by the [`inspect()`](ParallelIteratorExt::inspect)
/// method on [`ParallelIteratorExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and [`ParallelIteratorExt`] traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Inspect<Inner: ParallelIterator, F> {
    inner: Inner,
    f: F,
}

impl<Inner: ParallelIterator, F> ParallelAdaptor for Inspect<Inner, F>
where
    F: Fn(&Inner::Item) + Sync,
{
    type Item = Inner::Item;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: move |item| {
                (self.f)(&item);
                Some(item)
            },
        }
    }
}

/// This struct is created by the [`map()`](ParallelIteratorExt::map) method on
/// [`ParallelIteratorExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and [`ParallelIteratorExt`] traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Map<Inner: ParallelIterator, F> {
    inner: Inner,
    f: F,
}

impl<Inner: ParallelIterator, T, F> ParallelAdaptor for Map<Inner, F>
where
    F: Fn(Inner::Item) -> T + Sync,
{
    type Item = T;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: move |item| Some((self.f)(item)),
        }
    }
}

/// This struct is created by the [`map_init()`](ParallelIteratorExt::map_init)
/// method on [`ParallelIteratorExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and [`ParallelIteratorExt`] traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct MapInit<Inner: ParallelIterator, Init, F> {
    inner: Inner,
    init: Init,
    f: F,
}

impl<Inner: ParallelIterator, I, Init, T, F> ParallelIterator for MapInit<Inner, Init, F>
where
    Init: Fn() -> I + Sync,
    F: Fn(&mut I, Inner::Item) -> T + Sync,
{
    type Item = T;

    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.inner.upper_bounded_pipeline(
            || ((self.init)(), init()),
            |(mut i, accum), index, item| {
                let accum = process_item(accum, index, (self.f)(&mut i, item));
                match accum {
                    ControlFlow::Continue(accum) => ControlFlow::Continue((i, accum)),
                    ControlFlow::Break(accum) => ControlFlow::Break((i, accum)),
                }
            },
            |(_, accum)| finalize(accum),
            reduce,
        )
    }

    fn iter_pipeline<Output: Send>(
        self,
        accum: impl Accumulator<Self::Item, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
    ) -> Output {
        let accumulator = MapInitAccumulator {
            inner: accum,
            init: self.init,
            f: self.f,
        };
        self.inner.iter_pipeline(accumulator, reduce)
    }
}

struct MapInitAccumulator<Inner, Init, F> {
    inner: Inner,
    init: Init,
    f: F,
}

impl<Item, Output, Inner, I, Init, T, F> Accumulator<Item, Output>
    for MapInitAccumulator<Inner, Init, F>
where
    Inner: Accumulator<T, Output>,
    Init: Fn() -> I,
    F: Fn(&mut I, Item) -> T,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output {
        let mut i = (self.init)();
        self.inner
            .accumulate(iter.map(|item| (self.f)(&mut i, item)))
    }
}
