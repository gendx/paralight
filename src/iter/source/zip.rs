// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{ParallelSource, SourceCleanup, SourceDescriptor};

/// A helper trait for zipping together multiple [`ParallelSource`]s into a
/// single [`ParallelSource`] that produces items grouped from the original
/// sources.
///
/// This trait is automatically implemented for [tuples](tuple) of
/// [`ParallelSource`]s (with up to 12 elements).
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
    /// ```
    ///
    /// ```should_panic
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
    /// let left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let right = [11, 12, 13, 14, 15];
    ///
    /// let _ = (left.par_iter(), right.par_iter())
    ///     .zip_eq()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .map(|(&a, &b)| (a, b))
    ///     .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
    /// ```
    fn zip_eq(self) -> ZipEq<Self> {
        ZipEq(self)
    }

    /// Returns a zipped [`ParallelSource`] whose length is the maximum of the
    /// input sources lengths. Produced items are [`Option`]s, equal to
    /// [`None`] for indices beyond a given source's length.
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
    /// ```
    fn zip_max(self) -> ZipMax<Self> {
        ZipMax(self)
    }

    /// Returns a zipped [`ParallelSource`] whose length is the minimum of the
    /// input sources lengths.
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
    /// ```
    fn zip_min(self) -> ZipMin<Self> {
        ZipMin(self)
    }
}

/// This struct is created by the [`zip_eq()`](ZipableSource::zip_eq) method on
/// [`ZipableSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct ZipEq<T>(T);

/// This struct is created by the [`zip_max()`](ZipableSource::zip_max) method
/// on [`ZipableSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct ZipMax<T>(T);

/// This struct is created by the [`zip_min()`](ZipableSource::zip_min) method
/// on [`ZipableSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct ZipMin<T>(T);

macro_rules! assert_eq_lens {
    ( $descriptors:expr, $zero:tt, $($i:tt),* ) => {
        $( assert_eq!(
            $descriptors.0.len,
            $descriptors.$i.len,
            "called zip_eq() with sources of different lengths"
        ); )+
    }
}

macro_rules! min_lens {
    ( $descriptors:expr, $zero:tt, $($i:tt),* ) => {
        $descriptors.0.len $( .min($descriptors.$i.len) )+
    }
}

macro_rules! or_bools {
    ( $tuple:expr, $zero:tt, $($i:tt),* ) => {
        $tuple.0 $( || $tuple.$i )*
    }
}

/// Helper that associates a source length and its cleanup function.
struct LengthCleanup<T> {
    len: usize,
    cleanup: T,
}

/// Helper to cleanup a [`ZipEq`] source.
struct ZipEqCleanup<T>(T);

/// Helper to cleanup a [`ZipMax`] source.
struct ZipMaxCleanup<T>(T);

macro_rules! zipable_tuple {
    ( $detail:ident, $($tuple:ident $i:tt),+ ) => {
        impl<$($tuple),+> ZipableSource for ($($tuple),+)
        where $($tuple: ParallelSource),+ {}

        impl<$($tuple),+> ParallelSource for ZipEq<($($tuple),+)>
        where $($tuple: ParallelSource),+ {
            type Item = ( $($tuple::Item),+ );

            fn descriptor(
                self,
            ) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync, impl SourceCleanup + Sync>
            {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor()),+ );
                assert_eq_lens!(descriptors, $($i),+);
                SourceDescriptor {
                    len: descriptors.0.len,
                    fetch_item: move |index| {
                        ( $( (descriptors.$i.fetch_item)(index) ),+ )
                    },
                    cleanup: ZipEqCleanup(( $(descriptors.$i.cleanup),+ )),
                }
            }
        }

        impl<$($tuple),+> ParallelSource for ZipMax<($($tuple),+)>
        where $($tuple: ParallelSource),+ {
            type Item = ( $(Option<$tuple::Item>),+ );

            fn descriptor(
                self,
            ) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync, impl SourceCleanup + Sync>
            {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor()),+ );
                let mut len = 0;
                $( len = descriptors.$i.len.max(len); )+
                SourceDescriptor {
                    len,
                    fetch_item: move |index| {
                        ( $(
                            if index < descriptors.$i.len {
                                Some((descriptors.$i.fetch_item)(index))
                            } else {
                                None
                            }
                        ),+ )
                    },
                    cleanup: ZipMaxCleanup(( $( LengthCleanup {
                        len: descriptors.$i.len,
                        cleanup: descriptors.$i.cleanup,
                    } ),+ )),
                }
            }
        }

        impl<$($tuple),+> ParallelSource for ZipMin<($($tuple),+)>
        where $($tuple: ParallelSource),+ {
            type Item = ( $($tuple::Item),+ );

            fn descriptor(
                self,
            ) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync, impl SourceCleanup + Sync>
            {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor()),+ );
                let len = min_lens!(descriptors, $($i),+);
                SourceDescriptor {
                    len,
                    fetch_item: move |index| {
                        ( $( (descriptors.$i.fetch_item)(index) ),+ )
                    },
                    cleanup: $detail::ZipMinCleanup {
                        len,
                        tuple: ( $( LengthCleanup {
                            len: descriptors.$i.len,
                            cleanup: descriptors.$i.cleanup,
                        } ),+ ),
                    },
                }
            }
        }

        impl<$($tuple),+> SourceCleanup for ZipEqCleanup<($($tuple),+)>
        where $($tuple: SourceCleanup),+ {
            const NEEDS_CLEANUP: bool = {
                let need_cleanups = ( $($tuple::NEEDS_CLEANUP),+ );
                or_bools!(need_cleanups, $($i),+)
            };

            fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                if Self::NEEDS_CLEANUP {
                    $( self.0.$i.cleanup_item_range(range.clone()); )+
                }
            }
        }

        impl<$($tuple),+> SourceCleanup for ZipMaxCleanup<($( LengthCleanup<$tuple> ),+)>
        where $($tuple: SourceCleanup),+ {
            const NEEDS_CLEANUP: bool = {
                let need_cleanups = ( $($tuple::NEEDS_CLEANUP),+ );
                or_bools!(need_cleanups, $($i),+)
            };

            fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                if Self::NEEDS_CLEANUP {
                    let tuple = &self.0;
                    $( {
                        let this_len = tuple.$i.len;
                        let this_range = range.start.min(this_len)..range.end.min(this_len);
                        tuple.$i.cleanup.cleanup_item_range(this_range);
                    } )+
                }
            }
        }

        // As long as Rust lacks variadic generics (or drop specialization), we need to define a
        // separate `ZipMinCleanup` struct for each tuple length, because the `Drop` implementation
        // only applies to tuples. Defining a more general `ZipMinCleanup<T>` like for
        // `ZipEqCleanup` and `ZipMaxCleanup` would prevent defining the `Drop` implementation only
        // when `T` is a tuple, as `Drop` must be implemented for the same constraints as the
        // struct.
        mod $detail {
            use super::*;

            /// Helper to cleanup a [`ZipMin`] source.
            pub struct ZipMinCleanup<$($tuple),+>
            where $($tuple: SourceCleanup),+ {
                pub len: usize,
                pub tuple: ( $( LengthCleanup<$tuple> ),+ ),
            }

            impl<$($tuple),+> SourceCleanup for ZipMinCleanup<$($tuple),+>
            where $($tuple: SourceCleanup),+ {
                const NEEDS_CLEANUP: bool = {
                    let need_cleanups = ( $($tuple::NEEDS_CLEANUP),+ );
                    or_bools!(need_cleanups, $($i),+)
                };

                fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                    if Self::NEEDS_CLEANUP {
                        $( self.tuple.$i.cleanup.cleanup_item_range(range.clone()); )+
                    }
                }
            }

            impl<$($tuple),+> Drop for ZipMinCleanup<$($tuple),+>
            where $($tuple: SourceCleanup),+ {
                fn drop(&mut self) {
                    if Self::NEEDS_CLEANUP {
                        $( self.tuple.$i.cleanup.cleanup_item_range(self.len..self.tuple.$i.len); )+
                    }
                }
            }
        }
    }
}

zipable_tuple!(zip2, A 0, B 1);
zipable_tuple!(zip3, A 0, B 1, C 2);
zipable_tuple!(zip4, A 0, B 1, C 2, D 3);
zipable_tuple!(zip5, A 0, B 1, C 2, D 3, E 4);
zipable_tuple!(zip6, A 0, B 1, C 2, D 3, E 4, F 5);
zipable_tuple!(zip7, A 0, B 1, C 2, D 3, E 4, F 5, G 6);
zipable_tuple!(zip8, A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
zipable_tuple!(zip9, A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
zipable_tuple!(zip10, A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
zipable_tuple!(zip11, A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
zipable_tuple!(zip12, A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);

#[cfg(test)]
mod test {
    struct LenField {
        len: usize,
    }

    fn len(len: usize) -> LenField {
        LenField { len }
    }

    #[test]
    fn assert_eq_lens() {
        assert_eq_lens!((len(1), len(1)), 0, 1);
        assert_eq_lens!((len(1), len(1), len(1)), 0, 1, 2);
        assert_eq_lens!((len(1), len(1), len(1), len(1)), 0, 1, 2, 3);
    }

    #[test]
    #[should_panic(expected = "called zip_eq() with sources of different lengths")]
    fn assert_eq_lens_unequal_2() {
        assert_eq_lens!((len(1), len(2)), 0, 1);
    }

    #[test]
    #[should_panic(expected = "called zip_eq() with sources of different lengths")]
    fn assert_eq_lens_unequal_3() {
        assert_eq_lens!((len(1), len(1), len(2)), 0, 1, 2);
    }

    #[test]
    fn min_lens() {
        assert_eq!(min_lens!((len(1), len(2)), 0, 1), 1);
        assert_eq!(min_lens!((len(2), len(1)), 0, 1), 1);

        assert_eq!(min_lens!((len(1), len(2), len(3)), 0, 1, 2), 1);
        assert_eq!(min_lens!((len(1), len(3), len(2)), 0, 1, 2), 1);
        assert_eq!(min_lens!((len(2), len(1), len(3)), 0, 1, 2), 1);
        assert_eq!(min_lens!((len(2), len(3), len(1)), 0, 1, 2), 1);
        assert_eq!(min_lens!((len(3), len(1), len(2)), 0, 1, 2), 1);
        assert_eq!(min_lens!((len(3), len(2), len(1)), 0, 1, 2), 1);
    }
}
