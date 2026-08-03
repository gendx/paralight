// Copyright 2026 The Paralight Authors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{ExactParallelSink, FromExactParallelSink};
use crate::iter::source::zip::or_bools;
use std::array;
use std::mem::ManuallyDrop;

/// A parallel sink towards a tuple or array of sinks. This struct is consumed
/// by the [`finalize()`](FromExactParallelSink::finalize) method on
/// [`FromExactParallelSink`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSink`] trait, but it is nonetheless public as
/// it is an associated type of [`FromExactParallelSink`] for [`Unzip`].
#[must_use = "iterator adaptors are lazy"]
pub struct UnzipSink<T>(T);

/// A helper struct to collect into a tuple or array of sinks.
///
/// You most likely won't need to interact with this struct directly, but it is
/// nonetheless public as it is used in the signature of the
/// [`unzip()`](super::super::BaseExactParallelIterator::unzip) method on
/// [`BaseExactParallelIterator`](super::super::BaseExactParallelIterator).
#[must_use = "iterator adaptors are lazy"]
pub struct Unzip<T>(pub(crate) T);

/// # Safety
///
/// - In case of drop, calls `skip_item_range` on the `sink` with the given
///   `index`.
/// - Calling `into_inner` inhibits the drop.
struct SkipItemGuard<'a, S: ExactParallelSink> {
    sink: &'a S,
    index: usize,
}

impl<'a, S: ExactParallelSink> SkipItemGuard<'a, S> {
    fn into_inner(self) -> &'a S {
        // This inhibits the drop behavior as promised.
        let this = ManuallyDrop::new(self);
        this.sink
    }
}

impl<S: ExactParallelSink> Drop for SkipItemGuard<'_, S> {
    fn drop(&mut self) {
        // SAFETY: This does what the safety documentation describes on this struct.
        unsafe {
            self.sink.skip_item_range(self.index..self.index + 1);
        }
    }
}

macro_rules! unzipable_tuple {
    ( $($tuple:ident $i:tt),+ ) => {
        impl<$($tuple,)+> ExactParallelSink for UnzipSink<($($tuple,)+)>
        where $($tuple: ExactParallelSink,)+ {
            type Item = ( $($tuple::Item,)+ );
            const NEEDS_CLEANUP: bool = or_bools!( $($tuple::NEEDS_CLEANUP),+ );

            fn new(len: usize) -> Self {
                Self( ( $($tuple::new(len),)+ ) )
            }

            unsafe fn push_item(&self, index: usize, item: Self::Item) {
                // Build a guard around each sink to recover gracefully in case of panic in
                // `push_item` below.
                let guards = ( $(
                    SkipItemGuard {
                        sink: &self.0.$i,
                        index,
                    },
                )+ );

                // SAFETY:
                // - Indices are in the `0..len` range as they are forwarded from the caller.
                // - The current index is used at most once by the caller in calls to
                //   `push_item` and `skip_item_range`, so it's also passed at most once to the
                //   callees, either to `push_item` below or to `skip_item_range` via the
                //   `SkipItemGuard` in case of panic.
                $( unsafe {
                    guards.$i.into_inner().push_item(index, item.$i);
                } )+
            }

            unsafe fn skip_item_range(&self, range: std::ops::Range<usize>) {
                let _ = ( $(
                    // SAFETY:
                    // - Ranges are in the `0..len` range as they are forwarded from the caller.
                    // - The current range is used at most once by the caller in calls to
                    //   `push_item` and `skip_item_range`, so it's also passed at most once to the
                    //   callees, to `skip_item_range` below.
                    scopeguard::guard(&self.0.$i, |sink| unsafe {
                        sink.skip_item_range(range.clone());
                    }),
                )+ );
            }

            unsafe fn cancel(self) {
                let _ = ( $(
                    // SAFETY:
                    // - The caller has passed all indices exactly once to `push_item` and
                    //   `skip_item_range` calls, and we forwarded each index exactly once (possibly
                    //   via drop guards).
                    scopeguard::guard(self.0.$i, |sink| unsafe {
                        sink.cancel();
                    }),
                )+ );
            }
        }

        impl<$($tuple,)+> FromExactParallelSink for Unzip<($($tuple,)+)>
        where $($tuple: FromExactParallelSink,)+ {
            type Item = ( $($tuple::Item,)+ );
            type Sink = UnzipSink<( $($tuple::Sink,)+ )>;

            unsafe fn finalize(sink: Self::Sink) -> Self {
                // SAFETY:
                // - The caller has passed all indices exactly once to `push_item`, which we've
                //   forwarded.
                // - We may have forwarded an index to `skip_item_range` in case of panic, but
                //   in that case the caller would call `cancel` instead of `finalize` (see the
                //   implementations of `collect()` and `try_collect()`).
                Unzip(( $( unsafe {
                    $tuple::finalize(sink.0.$i)
                } ,)+ ))
            }
        }
    }
}

unzipable_tuple!(A 0);
unzipable_tuple!(A 0, B 1);
unzipable_tuple!(A 0, B 1, C 2);
unzipable_tuple!(A 0, B 1, C 2, D 3);
unzipable_tuple!(A 0, B 1, C 2, D 3, E 4);
unzipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5);
unzipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
unzipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
unzipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
unzipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
unzipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
unzipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);

impl<T, const N: usize> ExactParallelSink for UnzipSink<[T; N]>
where
    T: ExactParallelSink,
{
    type Item = [T::Item; N];
    const NEEDS_CLEANUP: bool = T::NEEDS_CLEANUP;

    fn new(len: usize) -> Self {
        Self(array::from_fn(|_| T::new(len)))
    }

    unsafe fn push_item(&self, index: usize, item: Self::Item) {
        // Build a guard around each sink to recover gracefully in case of panic in
        // `push_item` below.
        let mut i = 0;
        let guarded = item.map(|item| {
            let sink = &self.0[i];
            i += 1;
            (item, SkipItemGuard { sink, index })
        });

        let _: [(); N] = guarded.map(|(item, guard)| {
            unsafe {
                // SAFETY:
                // - Indices are in the `0..len` range as they are forwarded from the caller.
                // - The current index is used at most once by the caller in calls to
                //   `push_item` and `skip_item_range`, so it's also passed at most once to the
                //   callees, either to `push_item` below or to `skip_item_range` via the
                //   `SkipItemGuard` in case of panic.
                guard.into_inner().push_item(index, item);
            }
        });
    }

    unsafe fn skip_item_range(&self, range: std::ops::Range<usize>) {
        let _ = self.0.each_ref().map(|sink| {
            // SAFETY:
            // - Ranges are in the `0..len` range as they are forwarded from the caller.
            // - The current range is used at most once by the caller in calls to
            //   `push_item` and `skip_item_range`, so it's also passed at most once to the
            //   callees, to `skip_item_range` below.
            scopeguard::guard(sink, |sink| unsafe {
                sink.skip_item_range(range.clone());
            })
        });
    }

    unsafe fn cancel(self) {
        let _ = self.0.map(|sink| {
            // SAFETY:
            // - The caller has passed all indices exactly once to `push_item` and
            //   `skip_item_range` calls, and we forwarded each index exactly once (possibly
            //   via drop guards).
            scopeguard::guard(sink, |sink| unsafe {
                sink.cancel();
            })
        });
    }
}

impl<T, const N: usize> FromExactParallelSink for Unzip<[T; N]>
where
    T: FromExactParallelSink,
{
    type Item = [T::Item; N];
    type Sink = UnzipSink<[T::Sink; N]>;

    unsafe fn finalize(sink: Self::Sink) -> Self {
        // SAFETY:
        // - The caller has passed all indices exactly once to `push_item`, which we've
        //   forwarded.
        // - We may have forwarded an index to `skip_item_range` in case of panic, but
        //   in that case the caller would call `cancel` instead of `finalize` (see the
        //   implementations of `collect()` and `try_collect()`).
        Unzip(sink.0.map(|sink| unsafe { T::finalize(sink) }))
    }
}
