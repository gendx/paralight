// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Parallel sources from which parallel iterators are derived.

pub mod range;
pub mod slice;
pub mod zip;

use super::ParallelIterator;
use crate::{PipelineCircuit, ThreadPool};

/// An object describing how to fetch items from a [`ParallelSource`].
pub struct SourceDescriptor<Item: Send, FetchItem: Fn(usize) -> Item + Sync> {
    /// Number of items that the source produces.
    pub len: usize,
    /// A function to fetch the item at the given index.
    pub fetch_item: FetchItem,
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 0);
    /// ```
    fn skip(self, n: usize) -> Skip<Self> {
        Skip {
            inner: self,
            len: n,
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
    /// ```
    fn skip_exact(self, n: usize) -> SkipExact<Self> {
        SkipExact {
            inner: self,
            len: n,
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn take(self, n: usize) -> Take<Self> {
        Take {
            inner: self,
            len: n,
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
    /// ```
    fn take_exact(self, n: usize) -> TakeExact<Self> {
        TakeExact {
            inner: self,
            len: n,
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
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
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

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let descriptor1 = self.first.descriptor();
        let descriptor2 = self.second.descriptor();
        let len = descriptor1
            .len
            .checked_add(descriptor2.len)
            .unwrap_or_else(|| {
                panic!("called chain() with sources that together produce more than usize::MAX ({}) items", usize::MAX);
            });
        SourceDescriptor {
            len,
            fetch_item: move |index| {
                if index < descriptor1.len {
                    (descriptor1.fetch_item)(index)
                } else {
                    (descriptor2.fetch_item)(index - descriptor1.len)
                }
            },
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

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let descriptor = self.inner.descriptor();
        SourceDescriptor {
            len: descriptor.len,
            fetch_item: move |index| (index, (descriptor.fetch_item)(index)),
        }
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

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let descriptor = self.inner.descriptor();
        SourceDescriptor {
            len: descriptor.len,
            fetch_item: move |index| (descriptor.fetch_item)(descriptor.len - index - 1),
        }
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
    len: usize,
}

impl<Inner: ParallelSource> ParallelSource for Skip<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let descriptor = self.inner.descriptor();
        SourceDescriptor {
            len: descriptor.len - std::cmp::min(self.len, descriptor.len),
            fetch_item: move |index| (descriptor.fetch_item)(self.len + index),
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
    len: usize,
}

impl<Inner: ParallelSource> ParallelSource for SkipExact<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let descriptor = self.inner.descriptor();
        assert!(
            self.len <= descriptor.len,
            "called skip_exact() with more items than this source produces"
        );
        SourceDescriptor {
            len: descriptor.len - self.len,
            fetch_item: move |index| (descriptor.fetch_item)(self.len + index),
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
    len: usize,
}

impl<Inner: ParallelSource> ParallelSource for Take<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let descriptor = self.inner.descriptor();
        SourceDescriptor {
            len: std::cmp::min(self.len, descriptor.len),
            fetch_item: descriptor.fetch_item,
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
    len: usize,
}

impl<Inner: ParallelSource> ParallelSource for TakeExact<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let descriptor = self.inner.descriptor();
        assert!(
            self.len <= descriptor.len,
            "called take_exact() with more items than this source produces"
        );
        SourceDescriptor {
            len: self.len,
            fetch_item: descriptor.fetch_item,
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

    fn short_circuiting_pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> (PipelineCircuit, Accum) + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        let source_descriptor = self.source.descriptor();
        self.thread_pool.short_circuiting_pipeline(
            source_descriptor.len,
            init,
            |acc, index| process_item(acc, index, (source_descriptor.fetch_item)(index)),
            finalize,
            reduce,
        )
    }
}
