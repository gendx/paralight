// Copyright 2024-2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Parallel sources from which parallel iterators are derived.

#[cfg(feature = "nightly")]
pub mod array;
pub mod range;
pub mod slice;
pub mod vec;
pub mod vec_deque;
pub mod zip;

use super::{Accumulator, ParallelIterator};
use crate::ThreadPool;
use std::ops::ControlFlow;

/// An interface describing how to fetch items from a [`ParallelSource`].
#[allow(clippy::len_without_is_empty)]
pub trait SourceDescriptor: SourceCleanup {
    /// The type of items that this parallel source produces.
    type Item: Send;

    /// Returns the number of items that this source produces.
    fn len(&self) -> usize;

    /// Fetch the item at the given index.
    ///
    /// # Safety
    ///
    /// Given the length `len` returned by [`len()`](Self::len):
    /// - indices passed to [`fetch_item()`](Self::fetch_item) must be in the
    ///   `0..len` range,
    /// - each index in `0..len` must be present at most once in all indices
    ///   passed to calls to [`fetch_item()`](Self::fetch_item) and ranges
    ///   passed to calls to [`SourceCleanup::cleanup_item_range()`].
    ///
    /// It is therefore undefined behavior to call this function twice with the
    /// same index, with an index contained in a range for which
    /// [`cleanup_item_range()`](SourceCleanup::cleanup_item_range) was
    /// invoked, etc.
    ///
    /// You normally shouldn't have to worry about this, because this API is
    /// intended to be called by Paralight's internal multi-threading
    /// engine. This API is public to allow others to implement parallel
    /// sources: when implementing your own source(s), you can rely on these
    /// `unsafe` pre-conditions.
    unsafe fn fetch_item(&self, index: usize) -> Self::Item;
}

/// An interface to cleanup a range of items that aren't fetched from a source.
///
/// There are two reasons why manual cleanup of items is sometimes needed.
/// - If a short-circuiting combinator such as
///   [`find_any()`](super::ParallelIteratorExt::find_any) is used, the pipeline
///   will skip remaining items once a match is found.
/// - If a function in an iterator pipeline panics, the remaining items are
///   skipped but should nevertheless be dropped as part of unwinding.
///
/// A non-trivial cleanup is needed for parallel sources that drain items, such
/// as calling [`into_par_iter()`](IntoParallelSource::into_par_iter) on a
/// [`Vec`], and must correspond to [`drop()`]-ing items.
pub trait SourceCleanup {
    /// Set to [`false`] if the cleanup function is guaranteed to be a noop.
    ///
    /// Typically, cleanup is a noop for sources over [references](reference).
    /// For draining sources, this should follow the
    /// [`std::mem::needs_drop()`] hint.
    const NEEDS_CLEANUP: bool;

    /// Clean up the given range of items from the source.
    ///
    /// As with [`Drop`], this should not panic (but that's not a safety
    /// requirement).
    ///
    /// # Safety
    ///
    /// Given the length `len` returned by [`len()`](SourceDescriptor::len) in
    /// [`SourceDescriptor`]:
    /// - ranges passed to [`cleanup_item_range()`](Self::cleanup_item_range)
    ///   must be included in the `0..len` range,
    /// - each index in `0..len` must be present at most once in all indices
    ///   passed to calls to [`SourceDescriptor::fetch_item()`] and ranges
    ///   passed to calls to [`cleanup_item_range()`](Self::cleanup_item_range).
    ///
    /// It is therefore undefined behavior to call this function twice with the
    /// same range, with overlapping ranges, with a range that contains an
    /// index for which [`fetch_item()`](SourceDescriptor::fetch_item) was
    /// invoked, etc.
    ///
    /// You normally shouldn't have to worry about this, because this API is
    /// intended to be called by Paralight's internal multi-threading
    /// engine. This API is public to allow others to implement parallel
    /// sources: when implementing your own source(s), you can rely on these
    /// `unsafe` pre-conditions.
    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>);
}

/// A source to produce items in parallel. The [`ParallelSourceExt`] trait
/// provides additional methods (iterator adaptors) as an extension of this
/// trait.
///
/// This can be turned into a [`ParallelIterator`] by attaching a [`ThreadPool`]
/// via the [`with_thread_pool()`](ParallelSourceExt::with_thread_pool)
/// function.
pub trait ParallelSource: Sized {
    /// The type of items that this parallel source produces.
    ///
    /// Items are sent to worker threads (where they are then consumed by the
    /// `process_item` function parameter of the
    /// [`ParallelIterator::pipeline()`](super::ParallelIterator::pipeline)),
    /// hence the required [`Send`] bound.
    type Item: Send;

    /// Returns an object that describes how to fetch items from this source.
    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync;
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
    fn par_iter(&'data self) -> Self::Source;
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
    /// ```
    /// # use paralight::iter::{IntoParallelRefMutSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let mut values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = values
    ///     .par_iter_mut()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each(|x| {
    ///         *x *= 2;
    ///     });
    /// assert_eq!(values, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    /// ```
    fn par_iter_mut(&'data mut self) -> Self::Source;
}

/// Additional methods provided for types that implement [`ParallelSource`].
pub trait ParallelSourceExt: ParallelSource {
    /// Returns a parallel source that produces items from this source followed
    /// by items from the next source.
    ///
    /// Note: Given that items are processed in arbitrary order (in
    /// [`WorkStealing`](crate::RangeStrategy::WorkStealing) mode), the order in
    /// which sources are chained doesn't necessarily matter, but can be
    /// relevant when combined with order-sensitive adaptors (e.g.
    /// [`enumerate()`](Self::enumerate), [`take()`](Self::take), etc.).
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
    /// let first = [1, 2, 3, 4, 5, 6, 7];
    /// let second = [8, 9, 10, 11];
    /// let sum = first
    ///     .par_iter()
    ///     .chain(second.par_iter())
    ///     .take_exact(10)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn chain<T: ParallelSource<Item = Self::Item>>(self, next: T) -> Chain<Self, T> {
        Chain {
            first: self,
            second: next,
        }
    }

    /// Returns a parallel source that produces pairs of (index, item) for the
    /// items of this source.
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
    /// let indexed_max = input
    ///     .par_iter()
    ///     .enumerate()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .max_by_key(|(_, &x)| x);
    /// assert_eq!(indexed_max, Some((9, &10)));
    /// ```
    fn enumerate(self) -> Enumerate<Self> {
        Enumerate { inner: self }
    }

    /// Returns a parallel source that produces items from this source in
    /// reverse order.
    ///
    /// Note: Given that items are processed in arbitrary order (in
    /// [`WorkStealing`](crate::RangeStrategy::WorkStealing) mode), this isn't
    /// very useful on its own, but can be relevant when combined with
    /// order-sensitive adaptors (e.g. [`enumerate()`](Self::enumerate),
    /// [`take()`](Self::take), etc.).
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
    /// let input = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    /// let sum = input
    ///     .par_iter()
    ///     .rev()
    ///     .take_exact(10)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn rev(self) -> Rev<Self> {
        Rev { inner: self }
    }

    /// Returns a parallel source that skips the first `n` items from this
    /// source, or all the items if this source has fewer than `n` items, and
    /// produces the remaining items.
    ///
    /// See also [`skip_exact()`](Self::skip_exact) for a variant that panics if
    /// this source has fewer than `n` items.
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
    ///     .skip(5)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11 - 5 * 3);
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
    /// let sum = input
    ///     .par_iter()
    ///     .skip(15)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 0);
    /// ```
    fn skip(self, n: usize) -> Skip<Self> {
        Skip {
            inner: self,
            count: n,
        }
    }

    /// Returns a parallel source that skips the first `n` items from this
    /// source, panicking if this source has fewer than `n` items, and produces
    /// the remaining items.
    ///
    /// See also [`skip()`](Self::skip) for a variant that doesn't panic but
    /// produces no item if this source has fewer than `n` items.
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
    ///     .skip_exact(5)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11 - 5 * 3);
    /// ```
    ///
    /// ```should_panic
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let _ = input
    ///     .par_iter()
    ///     .skip_exact(15)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// ```
    fn skip_exact(self, n: usize) -> SkipExact<Self> {
        SkipExact {
            inner: self,
            count: n,
        }
    }

    /// Returns a parallel source that produces every `n`-th item from this
    /// source, starting with the first one.
    ///
    /// In other words, the returned source produces the items at indices `0`,
    /// `n`, `2*n`, etc.
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
    /// let sum = input
    ///     .par_iter()
    ///     .step_by(2)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 1 + 3 + 5 + 7 + 9);
    ///
    /// let sum = input
    ///     .par_iter()
    ///     .step_by(3)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 1 + 4 + 7 + 10);
    /// ```
    ///
    /// This panics if the step is zero, even if the underlying source is empty.
    ///
    /// ```should_panic
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let _ = input
    ///     .par_iter()
    ///     .step_by(0)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// ```
    ///
    /// ```should_panic
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let _ = []
    ///     .par_iter()
    ///     .step_by(0)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// ```
    fn step_by(self, n: usize) -> StepBy<Self> {
        StepBy {
            inner: self,
            step: n,
        }
    }

    /// Returns a parallel source that produces the first `n` items from this
    /// source, or all the items if this source has fewer than `n` items.
    ///
    /// See also [`take_exact()`](Self::take_exact) for a variant that panics if
    /// this source has fewer than `n` items.
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
    ///     .take(5)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 3);
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
    /// let sum = input
    ///     .par_iter()
    ///     .take(15)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn take(self, n: usize) -> Take<Self> {
        Take {
            inner: self,
            count: n,
        }
    }

    /// Returns a parallel source that produces the first `n` items from this
    /// source, panicking if this source has fewer than `n` items.
    ///
    /// See also [`take()`](Self::take) for a variant that doesn't panic but
    /// produces all the items if this source has fewer than `n` items.
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
    ///     .take_exact(5)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 3);
    /// ```
    ///
    /// ```should_panic
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let _ = input
    ///     .par_iter()
    ///     .take_exact(15)
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// ```
    fn take_exact(self, n: usize) -> TakeExact<Self> {
        TakeExact {
            inner: self,
            count: n,
        }
    }

    /// Attaches the given [`ThreadPool`] to this [`ParallelSource`] and obtain
    /// a [`ParallelIterator`].
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// let mut thread_pool = ThreadPoolBuilder {
    ///     num_threads: ThreadCount::AvailableParallelism,
    ///     range_strategy: RangeStrategy::WorkStealing,
    ///     cpu_pinning: CpuPinningPolicy::No,
    /// }
    /// .build();
    ///
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn with_thread_pool(self, thread_pool: &mut ThreadPool) -> BaseParallelIterator<'_, Self> {
        BaseParallelIterator {
            thread_pool,
            source: self,
        }
    }
}

impl<T: ParallelSource> ParallelSourceExt for T {}

/// This struct is created by the [`chain()`](ParallelSourceExt::chain) method
/// on [`ParallelSourceExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and [`ParallelSourceExt`] traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Chain<First, Second> {
    first: First,
    second: Second,
}

impl<T: Send, First: ParallelSource<Item = T>, Second: ParallelSource<Item = T>> ParallelSource
    for Chain<First, Second>
{
    type Item = T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor1 = self.first.descriptor();
        let descriptor2 = self.second.descriptor();

        let len1 = descriptor1.len();
        let len2 = descriptor2.len();
        let len = len1.checked_add(len2).unwrap_or_else(|| {
            panic!(
                "called chain() with sources that together produce more than usize::MAX items ({})",
                usize::MAX
            );
        });

        ChainSourceDescriptor {
            descriptor1,
            descriptor2,
            len,
            len1,
        }
    }
}

struct ChainSourceDescriptor<First, Second> {
    descriptor1: First,
    descriptor2: Second,
    len: usize,
    len1: usize,
}

impl<First, Second> SourceCleanup for ChainSourceDescriptor<First, Second>
where
    First: SourceCleanup,
    Second: SourceCleanup,
{
    const NEEDS_CLEANUP: bool = First::NEEDS_CLEANUP || Second::NEEDS_CLEANUP;

    // For safety comments: given two sources of lengths `len1` and `len2`, the
    // `ChainSourceDescriptor` creates a bijection of indices between `0..len1 +
    // len2` and `0..len1 | 0..len2`.
    //
    // Therefore:
    // - if the caller passes ranges included in `0..len1 + len2`, ranges passed to
    //   the two downstream `cleanup_item_range()` functions are included in their
    //   respective ranges `0..len1` and `0..len2`,
    // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
    //   and `fetch_item()`, the chain adaptor doesn't repeat indices passed to the
    //   two downstream descriptors.
    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            if range.end <= self.len1 {
                // SAFETY: See the function comment. This branch implements the mapping for a
                // range fully included in `0..len1` to `0..len1`.
                unsafe {
                    self.descriptor1.cleanup_item_range(range);
                }
            } else if range.start >= self.len1 {
                // SAFETY: See the function comment. This branch implements the mapping for a
                // range fully included in `len1..len1 + len2` to `0..len2`.
                unsafe {
                    self.descriptor2
                        .cleanup_item_range(range.start - self.len1..range.end - self.len1);
                }
            } else {
                // SAFETY: See the function comment. This branch implements the mapping for a
                // range that overlaps with `len1`.
                //
                // This line implements the mapping of the first half of the range (included in
                // `0..len1`) to `0..len1`.
                unsafe {
                    self.descriptor1.cleanup_item_range(range.start..self.len1);
                }
                // SAFETY: This line implements the mapping of the second half of the range
                // (included in `len1..len1 + len2`) to `0..len2`.
                unsafe {
                    self.descriptor2
                        .cleanup_item_range(0..range.end - self.len1);
                }
            }
        }
    }
}

impl<T: Send, First, Second> SourceDescriptor for ChainSourceDescriptor<First, Second>
where
    First: SourceDescriptor<Item = T>,
    Second: SourceDescriptor<Item = T>,
{
    type Item = T;

    fn len(&self) -> usize {
        self.len
    }

    // For safety comments: given two sources of lengths `len1` and `len2`, the
    // `ChainSourceDescriptor` creates a bijection of indices between `0..len1 +
    // len2` and `0..len1 | 0..len2`.
    //
    // Therefore:
    // - if the caller passes indices in `0..len1 + len2`, indices passed to the two
    //   downstream `fetch_item()` functions are included in their respective ranges
    //   `0..len1` and `0..len2`,
    // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
    //   and `fetch_item()`, the chain adaptor doesn't repeat indices passed to the
    //   two downstream descriptors.
    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        if index < self.len1 {
            // SAFETY: See the function comment. This branch implements the mapping for an
            // index in `0..len1` to `0..len1`.
            unsafe { self.descriptor1.fetch_item(index) }
        } else {
            // SAFETY: See the function comment. This branch implements the mapping for an
            // index in `len1..len1 + len2` to `0..len2`.
            unsafe { self.descriptor2.fetch_item(index - self.len1) }
        }
    }
}

/// This struct is created by the [`enumerate()`](ParallelSourceExt::enumerate)
/// method on [`ParallelSourceExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and [`ParallelSourceExt`] traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Enumerate<Inner> {
    inner: Inner,
}

impl<Inner: ParallelSource> ParallelSource for Enumerate<Inner> {
    type Item = (usize, Inner::Item);

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        EnumerateSourceDescriptor {
            inner: self.inner.descriptor(),
        }
    }
}

struct EnumerateSourceDescriptor<Inner> {
    inner: Inner,
}

impl<Inner: SourceCleanup> SourceCleanup for EnumerateSourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // SAFETY: The `EnumerateSourceDescriptor` only implements a mapping of items,
            // while passing through indices to the inner descriptor.
            //
            // Therefore:
            // - if the caller passes ranges included in `0..len`, ranges passed to the
            //   inner `cleanup_item_range()` function are also included in the `0..len`
            //   range,
            // - if the caller doesn't repeat indices, the enumerate adaptor doesn't repeat
            //   indices passed to the inner descriptor.
            unsafe {
                self.inner.cleanup_item_range(range);
            }
        }
    }
}

impl<Inner: SourceDescriptor> SourceDescriptor for EnumerateSourceDescriptor<Inner> {
    type Item = (usize, Inner::Item);

    fn len(&self) -> usize {
        self.inner.len()
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        // SAFETY: The `EnumerateSourceDescriptor` only implements a mapping of items,
        // while passing through indices to the inner descriptor.
        //
        // Therefore:
        // - if the caller passes indices in `0..len`, indices passed to the inner
        //   `fetch_item()` function are also in the `0..len` range,
        // - if the caller doesn't repeat indices, the enumerate adaptor doesn't repeat
        //   indices passed to the inner descriptor.
        (index, unsafe { self.inner.fetch_item(index) })
    }
}

/// This struct is created by the [`rev()`](ParallelSourceExt::rev) method on
/// [`ParallelSourceExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and [`ParallelSourceExt`] traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Rev<Inner> {
    inner: Inner,
}

impl<Inner: ParallelSource> ParallelSource for Rev<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.descriptor();
        let len = descriptor.len();
        RevSourceDescriptor {
            inner: descriptor,
            len,
        }
    }
}

struct RevSourceDescriptor<Inner> {
    inner: Inner,
    len: usize,
}

impl<Inner: SourceCleanup> SourceCleanup for RevSourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // SAFETY: Given an inner descriptor of length `len`, the `RevSourceDescriptor`
            // implements a bijective mapping of indices from `0..len` to `0..len` given by
            // `rev: x -> len - 1 - x`.
            //
            // Therefore:
            // - if the caller passes ranges included in `0..len`, ranges passed to the
            //   inner `cleanup_item_range()` function are also included in the `0..len`
            //   range,
            // - if the caller doesn't repeat indices, the rev adaptor doesn't repeat
            //   indices passed to the inner descriptor.
            //
            // Given an open-ended input range `start..end` = `start..=end - 1`, the mapped
            // range is `rev(end - 1)..rev(start) + 1` = `len - end..len - start`.
            unsafe {
                self.inner
                    .cleanup_item_range(self.len - range.end..self.len - range.start);
            }
        }
    }
}

impl<Inner: SourceDescriptor> SourceDescriptor for RevSourceDescriptor<Inner> {
    type Item = Inner::Item;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        // SAFETY: Given an inner descriptor of length `len`, the `RevSourceDescriptor`
        // implements a bijective mapping of indices from `0..len` to `0..len` given by
        // `rev: x -> len - 1 - x`.
        //
        // Therefore:
        // - if the caller passes indices in `0..len`, indices passed to the inner
        //   `fetch_item()` function are also in the `0..len` range,
        // - if the caller doesn't repeat indices, the rev adaptor doesn't repeat
        //   indices passed to the inner descriptor.
        unsafe { self.inner.fetch_item(self.len - index - 1) }
    }
}

/// This struct is created by the [`skip()`](ParallelSourceExt::skip) method on
/// [`ParallelSourceExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and [`ParallelSourceExt`] traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Skip<Inner> {
    inner: Inner,
    count: usize,
}

impl<Inner: ParallelSource> ParallelSource for Skip<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.descriptor();
        let inner_len = descriptor.len();
        let count = std::cmp::min(self.count, inner_len);
        SkipSourceDescriptor {
            inner: descriptor,
            len: inner_len - count,
            count,
        }
    }
}

struct SkipSourceDescriptor<Inner: SourceDescriptor> {
    inner: Inner,
    len: usize,
    count: usize,
}

impl<Inner: SourceDescriptor> SourceCleanup for SkipSourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // SAFETY: Given an inner descriptor of length `len` as well as a parameter
            // `count <= len`, the `SkipSourceDescriptor` implements a bijective mapping of
            // indices from `0..len - count` to `count..len` given by a translation of
            // `count` places.
            //
            // Therefore:
            // - if the caller passes ranges included in `0..len - count`, ranges passed
            //   here to the inner `cleanup_item_range()` function are included in the
            //   `count..len` range,
            // - if the caller doesn't repeat indices, the skip adaptor doesn't repeat
            //   indices passed to the inner descriptor.
            unsafe {
                self.inner
                    .cleanup_item_range(self.count + range.start..self.count + range.end);
            }
        }
    }
}

impl<Inner: SourceDescriptor> SourceDescriptor for SkipSourceDescriptor<Inner> {
    type Item = Inner::Item;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        // SAFETY: Given an inner descriptor of length `len` as well as a parameter
        // `count <= len`, the `SkipSourceDescriptor` implements a bijective mapping of
        // indices from `0..len - count` to `count..len` given by a translation of
        // `count` places.
        //
        // Therefore:
        // - if the caller passes indices in `0..len - count`, indices passed here to
        //   the inner `fetch_item()` function are in the `count..len` range,
        // - if the caller doesn't repeat indices, the skip adaptor doesn't repeat
        //   indices passed to the inner descriptor.
        unsafe { self.inner.fetch_item(self.count + index) }
    }
}

impl<Inner: SourceDescriptor> Drop for SkipSourceDescriptor<Inner> {
    fn drop(&mut self) {
        if Self::NEEDS_CLEANUP && self.count != 0 {
            // SAFETY: Given an inner descriptor of length `len` as well as a parameter
            // `count <= len`, the `SkipSourceDescriptor` implements a bijective mapping of
            // indices from `0..len - count` to `count..len` given by a translation of
            // `count` places.
            //
            // Therefore:
            // - the range `0..count` is included in the inner range `0..len`,
            // - the items in `0..count` aren't passed to the inner descriptor other than in
            //   this drop implementation.
            unsafe {
                self.inner.cleanup_item_range(0..self.count);
            }
        }
    }
}

/// This struct is created by the
/// [`skip_exact()`](ParallelSourceExt::skip_exact) method on
/// [`ParallelSourceExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and [`ParallelSourceExt`] traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct SkipExact<Inner> {
    inner: Inner,
    count: usize,
}

impl<Inner: ParallelSource> ParallelSource for SkipExact<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.descriptor();
        let inner_len = descriptor.len();
        assert!(
            self.count <= inner_len,
            "called skip_exact() with more items than this source produces"
        );
        SkipSourceDescriptor {
            inner: descriptor,
            len: inner_len - self.count,
            count: self.count,
        }
    }
}

/// This struct is created by the
/// [`step_by()`](ParallelSourceExt::step_by) method on [`ParallelSourceExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and [`ParallelSourceExt`] traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct StepBy<Inner> {
    inner: Inner,
    step: usize,
}

impl<Inner: ParallelSource> ParallelSource for StepBy<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.descriptor();
        let inner_len = descriptor.len();
        assert!(self.step != 0, "called step_by() with a step of zero");
        let len = inner_len.div_ceil(self.step);
        StepBySourceDescriptor {
            inner: descriptor,
            len,
            step: self.step,
            inner_len,
        }
    }
}

struct StepBySourceDescriptor<Inner: SourceDescriptor> {
    inner: Inner,
    len: usize,
    step: usize,
    inner_len: usize,
}

impl<Inner: SourceDescriptor> SourceCleanup for StepBySourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    // For safety comments: given an inner descriptor of length `len` as well as a
    // parameter `step != 0`, if we set `len' := ceil(len / step)` the
    // `StepBySourceDescriptor` implements a bijective mapping between `0..len'` and
    // `{0, step, 2*step, ..., (len' - 1)*step}` given by `f: x -> x * step`.
    //
    // Therefore:
    // - if the caller passes indices included in `0..len'`, indices passed to the
    //   inner descriptor are included in the `0..=(len' - 1) * step` range, itself
    //   included in `0..len`,
    // - if the caller doesn't repeat indices, the step-by adaptor doesn't repeat
    //   indices passed to the inner descriptor.
    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            if self.step == 1 {
                // SAFETY: See the function comment. When the step is 1 the mapping is the
                // identity so we just pass the range through.
                unsafe {
                    self.inner.cleanup_item_range(range);
                }
            } else {
                for i in range {
                    // SAFETY: See the function comment. This call with a length-one range cleans up
                    // the item at index `step * i`. The other items are cleaned up in the drop
                    // implementation.
                    unsafe {
                        self.inner
                            .cleanup_item_range(self.step * i..self.step * i + 1);
                    }
                }
            }
        }
    }
}

impl<Inner: SourceDescriptor> SourceDescriptor for StepBySourceDescriptor<Inner> {
    type Item = Inner::Item;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        // SAFETY: See the function comment in `Self::cleanup_item_range`. This
        // implements the mapping `i -> step * i`.
        unsafe { self.inner.fetch_item(self.step * index) }
    }
}

impl<Inner: SourceDescriptor> Drop for StepBySourceDescriptor<Inner> {
    // For safety comments: see the function comment in `Self::cleanup_item_range`.
    // This drop implementation is the only one to invoke items that aren't
    // multiples of `step`.
    fn drop(&mut self) {
        if Self::NEEDS_CLEANUP && self.step != 1 {
            let full_blocks = self.inner_len / self.step;
            for i in 0..full_blocks {
                // SAFETY: See the function comment. This line cleans up the items that aren't
                // multiples of the `step` in the `step * i..step * (i + 1)` range.
                unsafe {
                    self.inner
                        .cleanup_item_range(self.step * i + 1..self.step * (i + 1));
                }
            }
            let last_block = self.step * full_blocks;
            // This implements the comparison `last_block + 1 < inner_len` without risk of
            // overflow.
            if self.inner_len - last_block > 1 {
                // SAFETY: See the function comment. This line cleans up the items that aren't
                // multiples of the `step` beyond `step * len'`.
                unsafe {
                    self.inner
                        .cleanup_item_range(last_block + 1..self.inner_len);
                }
            }
        }
    }
}

/// This struct is created by the [`take()`](ParallelSourceExt::take) method on
/// [`ParallelSourceExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and [`ParallelSourceExt`] traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Take<Inner> {
    inner: Inner,
    count: usize,
}

impl<Inner: ParallelSource> ParallelSource for Take<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.descriptor();
        let inner_len = descriptor.len();
        let count = std::cmp::min(self.count, inner_len);
        TakeSourceDescriptor {
            inner: descriptor,
            count,
            inner_len,
        }
    }
}

struct TakeSourceDescriptor<Inner: SourceDescriptor> {
    inner: Inner,
    count: usize,
    inner_len: usize,
}

impl<Inner: SourceDescriptor> SourceCleanup for TakeSourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // SAFETY: Given an inner descriptor of length `len` as well as a parameter
            // `count <= len`, the `TakeSourceDescriptor` implements a pass-through mapping
            // of indices from `0..count` to `0..count`.
            //
            // Therefore:
            // - if the caller passes ranges included in `0..count`, ranges passed here to
            //   the inner `cleanup_item_range()` function are included in the `0..count`
            //   range (itself included in `0..len`),
            // - if the caller doesn't repeat indices, the take adaptor doesn't repeat
            //   indices passed to the inner descriptor.
            unsafe {
                self.inner.cleanup_item_range(range);
            }
        }
    }
}

impl<Inner: SourceDescriptor> SourceDescriptor for TakeSourceDescriptor<Inner> {
    type Item = Inner::Item;

    fn len(&self) -> usize {
        self.count
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        // SAFETY: Given an inner descriptor of length `len` as well as a parameter
        // `count <= len`, the `TakeSourceDescriptor` implements a pass-through mapping
        // of indices from `0..count` to `0..count`.
        //
        // Therefore:
        // - if the caller passes indices in `0..count`, indices passed here to the
        //   inner `fetch_item()` function are in the `0..count` range (itself included
        //   in `0..len`),
        // - if the caller doesn't repeat indices, the take adaptor doesn't repeat
        //   indices passed to the inner descriptor.
        unsafe { self.inner.fetch_item(index) }
    }
}

impl<Inner: SourceDescriptor> Drop for TakeSourceDescriptor<Inner> {
    fn drop(&mut self) {
        if Self::NEEDS_CLEANUP && self.count != self.inner_len {
            // SAFETY: Given an inner descriptor of length `len` as well as a parameter
            // `count <= len`, the `TakeSourceDescriptor` implements a pass-through mapping
            // of indices from `0..count` to `0..count`.
            //
            // Therefore:
            // - the range `count..len` is included in the inner range `0..len`,
            // - the items in `count..len` aren't passed to the inner descriptor other than
            //   in this drop implementation.
            unsafe {
                self.inner.cleanup_item_range(self.count..self.inner_len);
            }
        }
    }
}

/// This struct is created by the
/// [`take_exact()`](ParallelSourceExt::take_exact) method on
/// [`ParallelSourceExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and [`ParallelSourceExt`] traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct TakeExact<Inner> {
    inner: Inner,
    count: usize,
}

impl<Inner: ParallelSource> ParallelSource for TakeExact<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.descriptor();
        let inner_len = descriptor.len();
        assert!(
            self.count <= inner_len,
            "called take_exact() with more items than this source produces"
        );
        TakeSourceDescriptor {
            inner: descriptor,
            count: self.count,
            inner_len,
        }
    }
}

/// This struct is created by the
/// [`with_thread_pool()`](ParallelSourceExt::with_thread_pool) method on
/// [`ParallelSourceExt`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and [`ParallelSourceExt`] traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct BaseParallelIterator<'pool, S: ParallelSource> {
    thread_pool: &'pool mut ThreadPool,
    source: S,
}

impl<S: ParallelSource> ParallelIterator for BaseParallelIterator<'_, S> {
    type Item = S::Item;

    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        let source_descriptor = self.source.descriptor();
        self.thread_pool.upper_bounded_pipeline(
            source_descriptor.len(),
            init,
            |acc, index| {
                process_item(
                    acc,
                    index,
                    // SAFETY: The pre-conditions to the `source_descriptor`'s `fetch_item()` and
                    // `cleanup_item_range()` methods are ensured by the safety guarantees of
                    // `ThreadPool::upper_bounded_pipeline()`, i.e. that all the indices passed are
                    // in `0..len` and they are each passed exactly once.
                    unsafe { source_descriptor.fetch_item(index) },
                )
            },
            finalize,
            reduce,
            &source_descriptor,
        )
    }

    fn iter_pipeline<Output: Send>(
        self,
        accum: impl Accumulator<Self::Item, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
    ) -> Output {
        let source_descriptor = self.source.descriptor();
        let accumulator = FetchAccumulator {
            inner: accum,
            fetch_item: |index| {
                // SAFETY: The pre-conditions to the `source_descriptor`'s `fetch_item()` and
                // `cleanup_item_range()` methods are ensured by the safety guarantees of
                // `ThreadPool::iter_pipeline()`, i.e. that all the indices passed are in
                // `0..len` and they are each passed exactly once.
                unsafe { source_descriptor.fetch_item(index) }
            },
        };
        self.thread_pool.iter_pipeline(
            source_descriptor.len(),
            accumulator,
            reduce,
            &source_descriptor,
        )
    }
}

struct FetchAccumulator<Inner, FetchItem> {
    inner: Inner,
    fetch_item: FetchItem,
}

impl<Item, Output, Inner, FetchItem> Accumulator<usize, Output>
    for FetchAccumulator<Inner, FetchItem>
where
    Inner: Accumulator<Item, Output>,
    FetchItem: Fn(usize) -> Item,
{
    fn accumulate(&self, iter: impl Iterator<Item = usize>) -> Output {
        self.inner.accumulate(iter.map(&self.fetch_item))
    }
}
