// Copyright 2024-2025 Google LLC
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
/// This trait is automatically implemented for [tuples](tuple) (with up to 12
/// elements) and [arrays](array) of [`ParallelSource`]s.
///
/// ```
/// # use paralight::iter::{
/// #     IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt, ZipableSource,
/// # };
/// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
/// use std::array;
///
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// // arrays[0] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
/// // arrays[1] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
/// // ...
/// let arrays: [[i32; 10]; 20] = array::from_fn(|i| array::from_fn(|j| (10 * i + j) as i32));
///
/// let sums: [i32; 20] = arrays
///     .each_ref()
///     .map(|a| a.par_iter())
///     .zip_eq()
///     .with_thread_pool(&mut thread_pool)
///     .map(|a| a.map(|x| *x))
///     .reduce(|| [0; 20], |a, b| array::from_fn(|i| a[i] + b[i]));
///
/// // Each sum is: 100 * i + (0 + ... + 9)
/// assert_eq!(sums, array::from_fn(|i| 100 * i as i32 + 45));
/// ```
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
    /// assert_eq!(sum_right, 100 + 5 * 11); // 11 + ... + 20
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
    /// assert_eq!(sum_right, 50 + 5 * 3); // 11 + ... + 15
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
    /// assert_eq!(sum_right, 50 + 5 * 3); // 11 + ... + 15
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
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct ZipEq<T>(T);

/// This struct is created by the [`zip_max()`](ZipableSource::zip_max) method
/// on [`ZipableSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct ZipMax<T>(T);

/// This struct is created by the [`zip_min()`](ZipableSource::zip_min) method
/// on [`ZipableSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct ZipMin<T>(T);

macro_rules! assert_all_eq {
    ( $len0:expr $(, $leni:expr)* ) => {
        {
            $( assert_eq!(
                $len0,
                $leni,
                "called zip_eq() with sources of different lengths"
            ); )*
            $len0
        }
    }
}

macro_rules! min_of {
    ( $x0:expr $(, $xi:expr)* ) => {
        $x0 $( .min($xi) )*
    }
}

macro_rules! max_of {
    ( $x0:expr $(, $xi:expr)* ) => {
        $x0 $( .max($xi) )*
    }
}

macro_rules! or_bools {
    ( $x0:expr $(, $xi:expr)* ) => {
        $x0 $( || $xi )*
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
        impl<$($tuple,)+> ZipableSource for ($($tuple,)+)
        where $($tuple: ParallelSource,)+ {}

        impl<$($tuple,)+> ParallelSource for ZipEq<($($tuple,)+)>
        where $($tuple: ParallelSource,)+ {
            type Item = ( $($tuple::Item,)+ );

            fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor(),)+ );
                let len = assert_all_eq!( $(descriptors.$i.len()),+ );
                ZipEqSourceDescriptor {
                    descriptors,
                    len,
                }
            }
        }

        impl<$($tuple,)+> ParallelSource for ZipMax<($($tuple,)+)>
        where $($tuple: ParallelSource,)+ {
            type Item = ( $(Option<$tuple::Item>,)+ );

            fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor(),)+ );
                let len = max_of!( $(descriptors.$i.len()),+ );
                ZipMaxSourceDescriptor {
                    descriptors,
                    len,
                }
            }
        }

        impl<$($tuple,)+> ParallelSource for ZipMin<($($tuple,)+)>
        where $($tuple: ParallelSource,)+ {
            type Item = ( $($tuple::Item,)+ );

            fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor(),)+ );
                let len = min_of!( $(descriptors.$i.len()),+ );
                $detail::ZipMinSourceDescriptor {
                    descriptors,
                    len,
                }
            }
        }

        impl<$($tuple,)+> SourceCleanup for ZipEqSourceDescriptor<($($tuple,)+)>
        where $($tuple: SourceCleanup,)+ {
            const NEEDS_CLEANUP: bool = or_bools!( $($tuple::NEEDS_CLEANUP),+ );

            unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                if Self::NEEDS_CLEANUP {
                    $(
                        // SAFETY: Given descriptors of equal lengths `len`, the `ZipEqSourceDescriptor`
                        // implements a pass-through of indices in `0..len` to all of them.
                        //
                        // Therefore:
                        // - if the caller passes ranges included in `0..len`, ranges passed to the
                        //   downstream `cleanup_item_range()` functions are also included in the
                        //   `0..len` range,
                        // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                        //   and `fetch_item()`, the zip-eq adaptor doesn't repeat indices passed to the
                        //   downstream descriptors.
                        unsafe {
                            self.descriptors.$i.cleanup_item_range(range.clone());
                        }
                    )+
                }
            }
        }

        impl<$($tuple,)+> SourceDescriptor for ZipEqSourceDescriptor<($($tuple,)+)>
        where $($tuple: SourceDescriptor,)+ {
            type Item = ( $($tuple::Item,)+ );

            fn len(&self) -> usize {
                self.len
            }

            unsafe fn fetch_item(&self, index: usize) -> Self::Item {
                ( $(
                    // SAFETY: Given descriptors of equal lengths `len`, the `ZipEqSourceDescriptor`
                    // implements a pass-through of indices in `0..len` to all of them.
                    //
                    // Therefore:
                    // - if the caller passes indices in `0..len`, indices passed to the downstream
                    //   `fetch_item()` functions are also in the `0..len` range,
                    // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                    //   and `fetch_item()`, the zip-eq adaptor doesn't repeat indices passed to the
                    //   downstream descriptors.
                    unsafe {
                        self.descriptors.$i.fetch_item(index)
                    },
                )+ )
            }
        }

        impl<$($tuple,)+> SourceCleanup for ZipMaxSourceDescriptor<($($tuple,)+)>
        where $($tuple: SourceDescriptor,)+ {
            const NEEDS_CLEANUP: bool = or_bools!( $($tuple::NEEDS_CLEANUP),+ );

            unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                if Self::NEEDS_CLEANUP {
                    $( {
                        let this_len = self.descriptors.$i.len();
                        let this_range = range.start.min(this_len)..range.end.min(this_len);
                        // SAFETY: Given descriptors of maximal length `len`, the
                        // `ZipMaxSourceDescriptor` implements a pass-through of indices in `0..len` to
                        // all of them, padding with `None` values beyond each descriptor's length.
                        //
                        // Therefore:
                        // - ranges passed to the downstream `cleanup_item_range()` functions are in
                        //   their respective `0..len_i` ranges after clamping,
                        // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                        //   and `fetch_item()`, the zip-max adaptor doesn't repeat indices passed to
                        //   the downstream descriptors.
                        //
                        // This line implements the pass-through function, clamped to the current
                        // descriptor length.
                        unsafe {
                            self.descriptors.$i.cleanup_item_range(this_range);
                        }
                    } )+
                }
            }
        }

        impl<$($tuple,)+> SourceDescriptor for ZipMaxSourceDescriptor<($($tuple,)+)>
        where $($tuple: SourceDescriptor,)+ {
            type Item = ( $(Option<$tuple::Item>,)+ );

            fn len(&self) -> usize {
                self.len
            }

            unsafe fn fetch_item(&self, index: usize) -> Self::Item {
                ( $( if index < self.descriptors.$i.len() {
                    // SAFETY: Given descriptors of maximal length `len`, the
                    // `ZipMaxSourceDescriptor` implements a pass-through of indices in `0..len` to
                    // all of them, padding with `None` values beyond each descriptor's length.
                    //
                    // Therefore:
                    // - indices passed to the downstream `fetch_item()` functions are in their
                    //   respective `0..len_i` ranges in this branch,
                    // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                    //   and `fetch_item()`, the zip-max adaptor doesn't repeat indices passed to
                    //   the downstream descriptors.
                    //
                    // This line implements the pass-through function, when the index is lower than
                    // the current descriptor length.
                    unsafe {
                        Some(self.descriptors.$i.fetch_item(index))
                    }
                } else {
                    None
                }, )+ )
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

            pub struct ZipMinSourceDescriptor<$($tuple,)+>
            where $($tuple: SourceDescriptor,)+ {
                pub descriptors: ($($tuple,)+),
                pub len: usize,
            }

            impl<$($tuple,)+> SourceCleanup for ZipMinSourceDescriptor<$($tuple,)+>
            where $($tuple: SourceDescriptor,)+ {
                const NEEDS_CLEANUP: bool = or_bools!( $($tuple::NEEDS_CLEANUP),+ );

                unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                    if Self::NEEDS_CLEANUP {
                        $(
                            // SAFETY: Given descriptors of minimal length `len`, the
                            // `ZipMinSourceDescriptor` implements a pass-through of indices in `0..len`
                            // to all of them.
                            //
                            // Therefore:
                            // - if the caller passes ranges included in `0..len`, ranges passed to the
                            //   downstream `cleanup_item_range()` functions are also included in the
                            //   `0..len` range, itself included in the respective `0..len_i` ranges
                            //   (because `len <= len_i`),
                            // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                            //   and `fetch_item()`, the zip-min adaptor doesn't repeat indices passed to
                            //   the downstream descriptors.
                            unsafe {
                                self.descriptors.$i.cleanup_item_range(range.clone());
                            }
                        )+
                    }
                }
            }

            impl<$($tuple,)+> SourceDescriptor for ZipMinSourceDescriptor<$($tuple,)+>
            where $($tuple: SourceDescriptor,)+ {
                type Item = ( $($tuple::Item,)+ );

                fn len(&self) -> usize {
                    self.len
                }

                unsafe fn fetch_item(&self, index: usize) -> Self::Item {
                    ( $(
                        // SAFETY: Given descriptors of minimal length `len`, the
                        // `ZipMinSourceDescriptor` implements a pass-through of indices in `0..len`
                        // to all of them.
                        //
                        // Therefore:
                        // - if the caller passes indices in `0..len`, indices passed to the downstream
                        //   `fetch_item()` functions are also in the `0..len` range, itself included in
                        //   the respective `0..len_i` ranges (because `len <= len_i`),
                        // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                        //   and `fetch_item()`, the zip-min adaptor doesn't repeat indices passed to
                        //   the downstream descriptors.
                        unsafe {
                            self.descriptors.$i.fetch_item(index)
                        },
                    )+ )
                }
            }

            impl<$($tuple,)+> Drop for ZipMinSourceDescriptor<$($tuple,)+>
            where $($tuple: SourceDescriptor,)+ {
                fn drop(&mut self) {
                    if Self::NEEDS_CLEANUP {
                        $(
                            // SAFETY: Given descriptors of minimal length `len`, the
                            // `ZipMinSourceDescriptor` implements a pass-through of indices in `0..len`
                            // to all of them.
                            //
                            // Therefore:
                            // - the range `len..len_i` is included in the respective range `0..len_i`,
                            // - items beyond the common `len` aren't passed to the inner descriptor other
                            //   than in this drop implementation.
                            unsafe {
                                self.descriptors.$i.cleanup_item_range(self.len..self.descriptors.$i.len());
                            }
                        )+
                    }
                }
            }
        }
    }
}

zipable_tuple!(zip1, A 0);
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

impl<T, const N: usize> ZipableSource for [T; N] where T: ParallelSource {}

impl<T, const N: usize> ParallelSource for ZipEq<[T; N]>
where
    T: ParallelSource,
{
    type Item = [T::Item; N];

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let array = self.0;
        let descriptors = array.map(|source| source.descriptor());
        for i in 1..N {
            assert_eq!(
                descriptors[0].len(),
                descriptors[i].len(),
                "called zip_eq() with sources of different lengths"
            );
        }
        let len = if N == 0 { 0 } else { descriptors[0].len() };
        ZipEqSourceDescriptor { descriptors, len }
    }
}

impl<T, const N: usize> ParallelSource for ZipMax<[T; N]>
where
    T: ParallelSource,
{
    type Item = [Option<T::Item>; N];

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let array = self.0;
        let descriptors = array.map(|source| source.descriptor());
        let len = *descriptors
            .each_ref()
            .map(|desc| desc.len())
            .iter()
            .max()
            .unwrap_or(&0);
        ZipMaxSourceDescriptor { descriptors, len }
    }
}

impl<T, const N: usize> ParallelSource for ZipMin<[T; N]>
where
    T: ParallelSource,
{
    type Item = [T::Item; N];

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let array = self.0;
        let descriptors = array.map(|source| source.descriptor());
        let len = *descriptors
            .each_ref()
            .map(|desc| desc.len())
            .iter()
            .min()
            .unwrap_or(&0);
        ZipMinArraySourceDescriptor { descriptors, len }
    }
}

impl<T, const N: usize> SourceCleanup for ZipEqSourceDescriptor<[T; N]>
where
    T: SourceCleanup,
{
    const NEEDS_CLEANUP: bool = T::NEEDS_CLEANUP;

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            self.descriptors.each_ref().map(|desc| {
                // SAFETY: Given descriptors of equal lengths `len`, the `ZipEqSourceDescriptor`
                // implements a pass-through of indices in `0..len` to all of them.
                //
                // Therefore:
                // - if the caller passes ranges included in `0..len`, ranges passed to the
                //   downstream `cleanup_item_range()` functions are also included in the
                //   `0..len` range,
                // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                //   and `fetch_item()`, the zip-eq adaptor doesn't repeat indices passed to the
                //   downstream descriptors.
                unsafe { desc.cleanup_item_range(range.clone()) }
            });
        }
    }
}

impl<T, const N: usize> SourceDescriptor for ZipEqSourceDescriptor<[T; N]>
where
    T: SourceDescriptor,
{
    type Item = [T::Item; N];

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        self.descriptors.each_ref().map(|desc| {
            // SAFETY: Given descriptors of equal lengths `len`, the `ZipEqSourceDescriptor`
            // implements a pass-through of indices in `0..len` to all of them.
            //
            // Therefore:
            // - if the caller passes indices in `0..len`, indices passed to the downstream
            //   `fetch_item()` functions are also in the `0..len` range,
            // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
            //   and `fetch_item()`, the zip-eq adaptor doesn't repeat indices passed to the
            //   downstream descriptors.
            unsafe { desc.fetch_item(index) }
        })
    }
}

impl<T, const N: usize> SourceCleanup for ZipMaxSourceDescriptor<[T; N]>
where
    T: SourceDescriptor,
{
    const NEEDS_CLEANUP: bool = T::NEEDS_CLEANUP;

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            self.descriptors.each_ref().map(|desc| {
                let this_len = desc.len();
                let this_range = range.start.min(this_len)..range.end.min(this_len);
                // SAFETY: Given descriptors of maximal length `len`, the
                // `ZipMaxSourceDescriptor` implements a pass-through of indices in `0..len` to
                // all of them, padding with `None` values beyond each descriptor's length.
                //
                // Therefore:
                // - ranges passed to the downstream `cleanup_item_range()` functions are in
                //   their respective `0..len_i` ranges after clamping,
                // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                //   and `fetch_item()`, the zip-max adaptor doesn't repeat indices passed to
                //   the downstream descriptors.
                //
                // This line implements the pass-through function, clamped to the current
                // descriptor length.
                unsafe {
                    desc.cleanup_item_range(this_range);
                }
            });
        }
    }
}

impl<T, const N: usize> SourceDescriptor for ZipMaxSourceDescriptor<[T; N]>
where
    T: SourceDescriptor,
{
    type Item = [Option<T::Item>; N];

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        self.descriptors.each_ref().map(|desc| {
            if index < desc.len() {
                // SAFETY: Given descriptors of maximal length `len`, the
                // `ZipMaxSourceDescriptor` implements a pass-through of indices in `0..len` to
                // all of them, padding with `None` values beyond each descriptor's length.
                //
                // Therefore:
                // - indices passed to the downstream `fetch_item()` functions are in their
                //   respective `0..len_i` ranges in this branch,
                // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                //   and `fetch_item()`, the zip-max adaptor doesn't repeat indices passed to
                //   the downstream descriptors.
                //
                // This line implements the pass-through function, when the index is lower than
                // the current descriptor length.
                unsafe { Some(desc.fetch_item(index)) }
            } else {
                None
            }
        })
    }
}

struct ZipMinArraySourceDescriptor<T, const N: usize>
where
    T: SourceDescriptor,
{
    descriptors: [T; N],
    len: usize,
}

impl<T, const N: usize> SourceCleanup for ZipMinArraySourceDescriptor<T, N>
where
    T: SourceDescriptor,
{
    const NEEDS_CLEANUP: bool = T::NEEDS_CLEANUP;

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            self.descriptors.each_ref().map(|desc| {
                // SAFETY: Given descriptors of minimal length `len`, the
                // `ZipMinSourceDescriptor` implements a pass-through of indices in `0..len`
                // to all of them.
                //
                // Therefore:
                // - if the caller passes ranges included in `0..len`, ranges passed to the
                //   downstream `cleanup_item_range()` functions are also included in the
                //   `0..len` range, itself included in the respective `0..len_i` ranges
                //   (because `len <= len_i`),
                // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
                //   and `fetch_item()`, the zip-min adaptor doesn't repeat indices passed to
                //   the downstream descriptors.
                unsafe {
                    desc.cleanup_item_range(range.clone());
                }
            });
        }
    }
}

impl<T, const N: usize> SourceDescriptor for ZipMinArraySourceDescriptor<T, N>
where
    T: SourceDescriptor,
{
    type Item = [T::Item; N];

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        self.descriptors.each_ref().map(|desc| {
            // SAFETY: Given descriptors of minimal length `len`, the
            // `ZipMinSourceDescriptor` implements a pass-through of indices in `0..len`
            // to all of them.
            //
            // Therefore:
            // - if the caller passes indices in `0..len`, indices passed to the downstream
            //   `fetch_item()` functions are also in the `0..len` range, itself included in
            //   the respective `0..len_i` ranges (because `len <= len_i`),
            // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
            //   and `fetch_item()`, the zip-min adaptor doesn't repeat indices passed to
            //   the downstream descriptors.
            unsafe { desc.fetch_item(index) }
        })
    }
}

impl<T, const N: usize> Drop for ZipMinArraySourceDescriptor<T, N>
where
    T: SourceDescriptor,
{
    fn drop(&mut self) {
        if Self::NEEDS_CLEANUP {
            self.descriptors.each_ref().map(|desc| {
                // SAFETY: Given descriptors of minimal length `len`, the
                // `ZipMinSourceDescriptor` implements a pass-through of indices in `0..len`
                // to all of them.
                //
                // Therefore:
                // - the range `len..len_i` is included in the respective range `0..len_i`,
                // - items beyond the common `len` aren't passed to the inner descriptor other
                //   than in this drop implementation.
                unsafe {
                    desc.cleanup_item_range(self.len..desc.len());
                }
            });
        }
    }
}

#[cfg(test)]
mod test {
    #[test]
    fn assert_all_eq() {
        let len = assert_all_eq!(1, 1);
        assert_eq!(len, 1);
        let len = assert_all_eq!(1, 1, 1);
        assert_eq!(len, 1);
        let len = assert_all_eq!(1, 1, 1, 1);
        assert_eq!(len, 1);
    }

    #[test]
    #[should_panic(expected = "called zip_eq() with sources of different lengths")]
    fn assert_all_eq_unequal_2() {
        assert_all_eq!(1, 2);
    }

    #[test]
    #[should_panic(expected = "called zip_eq() with sources of different lengths")]
    fn assert_all_eq_unequal_3() {
        assert_all_eq!(1, 1, 2);
    }

    #[test]
    fn min_of() {
        assert_eq!(min_of!(1), 1);

        assert_eq!(min_of!(1, 2), 1);
        assert_eq!(min_of!(2, 1), 1);

        assert_eq!(min_of!(1, 2, 3), 1);
        assert_eq!(min_of!(1, 3, 2), 1);
        assert_eq!(min_of!(2, 1, 3), 1);
        assert_eq!(min_of!(2, 3, 1), 1);
        assert_eq!(min_of!(3, 1, 2), 1);
        assert_eq!(min_of!(3, 2, 1), 1);
    }

    #[test]
    fn max_of() {
        assert_eq!(max_of!(1), 1);

        assert_eq!(max_of!(1, 2), 2);
        assert_eq!(max_of!(2, 1), 2);

        assert_eq!(max_of!(1, 2, 3), 3);
        assert_eq!(max_of!(1, 3, 2), 3);
        assert_eq!(max_of!(2, 1, 3), 3);
        assert_eq!(max_of!(2, 3, 1), 3);
        assert_eq!(max_of!(3, 1, 2), 3);
        assert_eq!(max_of!(3, 2, 1), 3);
    }

    #[allow(clippy::bool_assert_comparison)]
    #[test]
    fn or_bools() {
        assert_eq!(or_bools!(true), true);
        assert_eq!(or_bools!(false), false);

        assert_eq!(or_bools!(true, true), true);
        assert_eq!(or_bools!(true, false), true);
        assert_eq!(or_bools!(false, true), true);
        assert_eq!(or_bools!(false, false), false);

        assert_eq!(or_bools!(true, true, true), true);
        assert_eq!(or_bools!(true, true, false), true);
        assert_eq!(or_bools!(true, false, true), true);
        assert_eq!(or_bools!(true, false, false), true);
        assert_eq!(or_bools!(false, true, true), true);
        assert_eq!(or_bools!(false, true, false), true);
        assert_eq!(or_bools!(false, false, true), true);
        assert_eq!(or_bools!(false, false, false), false);
    }
}
