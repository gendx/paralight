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

macro_rules! assert_all_eq {
    ( $lens:expr, $zero:tt, $($i:tt),* ) => {
        $( assert_eq!(
            $lens.0,
            $lens.$i,
            "called zip_eq() with sources of different lengths"
        ); )*
    }
}

macro_rules! min_of {
    ( $lens:expr, $zero:tt, $($i:tt),* ) => {
        $lens.0 $( .min($lens.$i) )*
    }
}

macro_rules! max_of {
    ( $lens:expr, $zero:tt, $($i:tt),* ) => {
        $lens.0 $( .max($lens.$i) )*
    }
}

macro_rules! or_bools {
    ( $tuple:expr, $zero:tt, $($i:tt),* ) => {
        $tuple.0 $( || $tuple.$i )*
    }
}

struct ZipEqSourceDescriptor<T> {
    descriptors: T,
    len: usize,
}

struct ZipMaxSourceDescriptor<T> {
    descriptors: T,
    len: usize,
}

macro_rules! zipable_tuple {
    ( $detail:ident, $($tuple:ident $i:tt),+ ) => {
        impl<$($tuple),+> ZipableSource for ($($tuple),+)
        where $($tuple: ParallelSource),+ {}

        impl<$($tuple),+> ParallelSource for ZipEq<($($tuple),+)>
        where $($tuple: ParallelSource),+ {
            type Item = ( $($tuple::Item),+ );

            fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor()),+ );
                let lens = ( $(descriptors.$i.len()),+ );
                assert_all_eq!(lens, $($i),+);
                ZipEqSourceDescriptor {
                    descriptors,
                    len: lens.0,
                }
            }
        }

        impl<$($tuple),+> ParallelSource for ZipMax<($($tuple),+)>
        where $($tuple: ParallelSource),+ {
            type Item = ( $(Option<$tuple::Item>),+ );

            fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor()),+ );
                let lens = ( $(descriptors.$i.len()),+ );
                let len = max_of!(lens, $($i),+);
                ZipMaxSourceDescriptor {
                    descriptors,
                    len,
                }
            }
        }

        impl<$($tuple),+> ParallelSource for ZipMin<($($tuple),+)>
        where $($tuple: ParallelSource),+ {
            type Item = ( $($tuple::Item),+ );

            fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor()),+ );
                let lens = ( $(descriptors.$i.len()),+ );
                let len = min_of!(lens, $($i),+);
                $detail::ZipMinSourceDescriptor {
                    descriptors,
                    len,
                }
            }
        }

        impl<$($tuple),+> SourceCleanup for ZipEqSourceDescriptor<($($tuple),+)>
        where $($tuple: SourceCleanup),+ {
            const NEEDS_CLEANUP: bool = {
                let need_cleanups = ( $($tuple::NEEDS_CLEANUP),+ );
                or_bools!(need_cleanups, $($i),+)
            };

            fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                if Self::NEEDS_CLEANUP {
                    $( self.descriptors.$i.cleanup_item_range(range.clone()); )+
                }
            }
        }

        impl<$($tuple),+> SourceDescriptor for ZipEqSourceDescriptor<($($tuple),+)>
        where $($tuple: SourceDescriptor),+ {
            type Item = ( $($tuple::Item),+ );

            fn len(&self) -> usize {
                self.len
            }

            fn fetch_item(&self, index: usize) -> Self::Item {
                ( $( self.descriptors.$i.fetch_item(index) ),+ )
            }
        }

        impl<$($tuple),+> SourceCleanup for ZipMaxSourceDescriptor<($($tuple),+)>
        where $($tuple: SourceDescriptor),+ {
            const NEEDS_CLEANUP: bool = {
                let need_cleanups = ( $($tuple::NEEDS_CLEANUP),+ );
                or_bools!(need_cleanups, $($i),+)
            };

            fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                if Self::NEEDS_CLEANUP {
                    $( {
                        let this_len = self.descriptors.$i.len();
                        let this_range = range.start.min(this_len)..range.end.min(this_len);
                        self.descriptors.$i.cleanup_item_range(this_range);
                    } )+
                }
            }
        }

        impl<$($tuple),+> SourceDescriptor for ZipMaxSourceDescriptor<($($tuple),+)>
        where $($tuple: SourceDescriptor),+ {
            type Item = ( $(Option<$tuple::Item>),+ );

            fn len(&self) -> usize {
                self.len
            }

            fn fetch_item(&self, index: usize) -> Self::Item {
                ( $( if index < self.descriptors.$i.len() {
                    Some(self.descriptors.$i.fetch_item(index))
                } else {
                    None
                } ),+ )
            }
        }

        // As long as Rust lacks variadic generics (or drop specialization), we need to define a
        // separate `ZipMinSourceDescriptor` struct for each tuple length, because the `Drop`
        // implementation only applies to tuples. Defining a more general
        // `ZipMinSourceDescriptor<T>` like for `ZipEqSourceDescriptor` and `ZipMaxSourceDescriptor`
        // would prevent defining the `Drop` implementation only when `T` is a tuple, as `Drop` must
        // be implemented for the same constraints as the struct.
        mod $detail {
            use super::*;

            pub struct ZipMinSourceDescriptor<$($tuple),+>
            where $($tuple: SourceDescriptor),+ {
                pub descriptors: ($($tuple),+),
                pub len: usize,
            }

            impl<$($tuple),+> SourceCleanup for ZipMinSourceDescriptor<$($tuple),+>
            where $($tuple: SourceDescriptor),+ {
                const NEEDS_CLEANUP: bool = {
                    let need_cleanups = ( $($tuple::NEEDS_CLEANUP),+ );
                    or_bools!(need_cleanups, $($i),+)
                };

                fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                    if Self::NEEDS_CLEANUP {
                        $( self.descriptors.$i.cleanup_item_range(range.clone()); )+
                    }
                }
            }

            impl<$($tuple),+> SourceDescriptor for ZipMinSourceDescriptor<$($tuple),+>
            where $($tuple: SourceDescriptor),+ {
                type Item = ( $($tuple::Item),+ );

                fn len(&self) -> usize {
                    self.len
                }

                fn fetch_item(&self, index: usize) -> Self::Item {
                    ( $( self.descriptors.$i.fetch_item(index) ),+ )
                }
            }

            impl<$($tuple),+> Drop for ZipMinSourceDescriptor<$($tuple),+>
            where $($tuple: SourceDescriptor),+ {
                fn drop(&mut self) {
                    if Self::NEEDS_CLEANUP {
                        $( self.descriptors.$i.cleanup_item_range(self.len..self.descriptors.$i.len()); )+
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
    #[test]
    fn assert_all_eq() {
        assert_all_eq!((1, 1), 0, 1);
        assert_all_eq!((1, 1, 1), 0, 1, 2);
        assert_all_eq!((1, 1, 1, 1), 0, 1, 2, 3);
    }

    #[test]
    #[should_panic(expected = "called zip_eq() with sources of different lengths")]
    fn assert_all_eq_unequal_2() {
        assert_all_eq!((1, 2), 0, 1);
    }

    #[test]
    #[should_panic(expected = "called zip_eq() with sources of different lengths")]
    fn assert_all_eq_unequal_3() {
        assert_all_eq!((1, 1, 2), 0, 1, 2);
    }

    #[test]
    fn min_of() {
        assert_eq!(min_of!((1, 2), 0, 1), 1);
        assert_eq!(min_of!((2, 1), 0, 1), 1);

        assert_eq!(min_of!((1, 2, 3), 0, 1, 2), 1);
        assert_eq!(min_of!((1, 3, 2), 0, 1, 2), 1);
        assert_eq!(min_of!((2, 1, 3), 0, 1, 2), 1);
        assert_eq!(min_of!((2, 3, 1), 0, 1, 2), 1);
        assert_eq!(min_of!((3, 1, 2), 0, 1, 2), 1);
        assert_eq!(min_of!((3, 2, 1), 0, 1, 2), 1);
    }
}
