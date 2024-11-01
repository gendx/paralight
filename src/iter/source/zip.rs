// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{ParallelSource, SourceDescriptor};

/// A helper trait for zipping together multiple [`ParallelSource`]s into a
/// single [`ParallelSource`] that produces items grouped from the original
/// sources.
///
/// This trait is automatically implemented for [tuples](tuple) of
/// [`ParallelSource`]s.
pub trait ZipableSource: Sized
where
    ZipEq<Self>: ParallelSource,
    ZipMax<Self>: ParallelSource,
    ZipMin<Self>: ParallelSource,
{
    /// Returns a zipped [`ParallelSource`] where all the input sources must be
    /// of equal lengths. If the lengths don't match, the obtained
    /// [`ParallelSource`] will panic.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool, ZipableSource};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let right = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];
    ///
    /// let (sum_left, sum_right) = (left.par_iter(), right.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .map(|(&a, &b)| (a, b))
    ///     .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
    ///
    /// assert_eq!(sum_left, 5 * 11); // 1 + ... + 10
    /// assert_eq!(sum_right, 10 * 21 - 5 * 11); // 11 + ... + 20
    /// # });
    /// ```
    ///
    /// ```should_panic
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool, ZipableSource};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let right = [11, 12, 13, 14, 15];
    ///
    /// let _ = (left.par_iter(), right.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .map(|(&a, &b)| (a, b))
    ///     .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
    /// # });
    /// ```
    fn zip_eq(self) -> ZipEq<Self> {
        ZipEq(self)
    }

    /// Returns a zipped [`ParallelSource`] whose length is the maximum of the
    /// input sources lengths. Produced items are [`Option`]s, equal to
    /// [`None`] for indices beyond a given source's length.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool, ZipableSource};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let right = [11, 12, 13, 14, 15];
    ///
    /// let (sum_left, sum_right) = (left.par_iter(), right.par_iter())
    ///     // Produces 10 tuples:
    ///     // - 5 tuples (Some(_), Some(_)),
    ///     // - then 5 tuples (Some(_), None).
    ///     .zip_max()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .map(|(a, b)| (a.copied().unwrap(), b.copied().unwrap_or(0)))
    ///     .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
    ///
    /// assert_eq!(sum_left, 5 * 11); // 1 + ... + 10
    /// assert_eq!(sum_right, 15 * 8 - 5 * 11); // 11 + ... + 15
    /// # });
    /// ```
    fn zip_max(self) -> ZipMax<Self> {
        ZipMax(self)
    }

    /// Returns a zipped [`ParallelSource`] whose length is the minimum of the
    /// input sources lengths.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool, ZipableSource};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let right = [11, 12, 13, 14, 15];
    ///
    /// let (sum_left, sum_right) = (left.par_iter(), right.par_iter())
    ///     // Produces 5 tuples.
    ///     .zip_min()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .map(|(&a, &b)| (a, b))
    ///     .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
    ///
    /// assert_eq!(sum_left, 5 * 3); // 1 + ... + 5
    /// assert_eq!(sum_right, 15 * 8 - 5 * 11); // 11 + ... + 15
    /// # });
    /// ```
    fn zip_min(self) -> ZipMin<Self> {
        ZipMin(self)
    }
}

/// This struct is created by the [`zip_eq()`](ZipableSource::zip_eq) method on
/// [`ZipableSource`].
#[must_use = "iterator adaptors are lazy"]
pub struct ZipEq<T>(T);

/// This struct is created by the [`zip_max()`](ZipableSource::zip_max) method
/// on [`ZipableSource`].
#[must_use = "iterator adaptors are lazy"]
pub struct ZipMax<T>(T);

/// This struct is created by the [`zip_min()`](ZipableSource::zip_min) method
/// on [`ZipableSource`].
#[must_use = "iterator adaptors are lazy"]
pub struct ZipMin<T>(T);

impl<A, B> ZipableSource for (A, B)
where
    A: ParallelSource,
    B: ParallelSource,
{
}

impl<A, B> ParallelSource for ZipEq<(A, B)>
where
    A: ParallelSource,
    B: ParallelSource,
{
    type Item = (A::Item, B::Item);

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let tuple = self.0;
        let descriptors = (tuple.0.descriptor(), tuple.1.descriptor());
        assert_eq!(
            descriptors.0.len, descriptors.1.len,
            "called zip_eq() with sources of different lengths"
        );
        SourceDescriptor {
            len: descriptors.0.len,
            fetch_item: move |index| {
                (
                    (descriptors.0.fetch_item)(index),
                    (descriptors.1.fetch_item)(index),
                )
            },
        }
    }
}

impl<A, B> ParallelSource for ZipMax<(A, B)>
where
    A: ParallelSource,
    B: ParallelSource,
{
    type Item = (Option<A::Item>, Option<B::Item>);

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let tuple = self.0;
        let descriptors = (tuple.0.descriptor(), tuple.1.descriptor());
        SourceDescriptor {
            len: descriptors.0.len.max(descriptors.1.len),
            fetch_item: move |index| {
                (
                    if index < descriptors.0.len {
                        Some((descriptors.0.fetch_item)(index))
                    } else {
                        None
                    },
                    if index < descriptors.1.len {
                        Some((descriptors.1.fetch_item)(index))
                    } else {
                        None
                    },
                )
            },
        }
    }
}

impl<A, B> ParallelSource for ZipMin<(A, B)>
where
    A: ParallelSource,
    B: ParallelSource,
{
    type Item = (A::Item, B::Item);

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let tuple = self.0;
        let descriptors = (tuple.0.descriptor(), tuple.1.descriptor());
        SourceDescriptor {
            len: descriptors.0.len.min(descriptors.1.len),
            fetch_item: move |index| {
                (
                    (descriptors.0.fetch_item)(index),
                    (descriptors.1.fetch_item)(index),
                )
            },
        }
    }
}

impl<A, B, C> ZipableSource for (A, B, C)
where
    A: ParallelSource,
    B: ParallelSource,
    C: ParallelSource,
{
}

impl<A, B, C> ParallelSource for ZipEq<(A, B, C)>
where
    A: ParallelSource,
    B: ParallelSource,
    C: ParallelSource,
{
    type Item = (A::Item, B::Item, C::Item);

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let tuple = self.0;
        let descriptors = (
            tuple.0.descriptor(),
            tuple.1.descriptor(),
            tuple.2.descriptor(),
        );
        assert_eq!(
            descriptors.0.len, descriptors.1.len,
            "called zip_eq() with sources of different lengths"
        );
        assert_eq!(
            descriptors.0.len, descriptors.2.len,
            "called zip_eq() with sources of different lengths"
        );
        SourceDescriptor {
            len: descriptors.0.len,
            fetch_item: move |index| {
                (
                    (descriptors.0.fetch_item)(index),
                    (descriptors.1.fetch_item)(index),
                    (descriptors.2.fetch_item)(index),
                )
            },
        }
    }
}

impl<A, B, C> ParallelSource for ZipMax<(A, B, C)>
where
    A: ParallelSource,
    B: ParallelSource,
    C: ParallelSource,
{
    type Item = (Option<A::Item>, Option<B::Item>, Option<C::Item>);

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let tuple = self.0;
        let descriptors = (
            tuple.0.descriptor(),
            tuple.1.descriptor(),
            tuple.2.descriptor(),
        );
        SourceDescriptor {
            len: descriptors
                .0
                .len
                .max(descriptors.1.len)
                .max(descriptors.2.len),
            fetch_item: move |index| {
                (
                    if index < descriptors.0.len {
                        Some((descriptors.0.fetch_item)(index))
                    } else {
                        None
                    },
                    if index < descriptors.1.len {
                        Some((descriptors.1.fetch_item)(index))
                    } else {
                        None
                    },
                    if index < descriptors.2.len {
                        Some((descriptors.2.fetch_item)(index))
                    } else {
                        None
                    },
                )
            },
        }
    }
}

impl<A, B, C> ParallelSource for ZipMin<(A, B, C)>
where
    A: ParallelSource,
    B: ParallelSource,
    C: ParallelSource,
{
    type Item = (A::Item, B::Item, C::Item);

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let tuple = self.0;
        let descriptors = (
            tuple.0.descriptor(),
            tuple.1.descriptor(),
            tuple.2.descriptor(),
        );
        SourceDescriptor {
            len: descriptors
                .0
                .len
                .min(descriptors.1.len)
                .min(descriptors.2.len),
            fetch_item: move |index| {
                (
                    (descriptors.0.fetch_item)(index),
                    (descriptors.1.fetch_item)(index),
                    (descriptors.2.fetch_item)(index),
                )
            },
        }
    }
}
