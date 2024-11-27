// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{IntoParallelSource, ParallelSource, SourceDescriptor};
#[cfg(feature = "nightly")]
use std::iter::Step;
use std::ops::{Range, RangeInclusive};

/// A parallel source over a [`Range`]. This struct is created by the
/// [`into_par_iter()`](IntoParallelSource::into_par_iter) method on
/// [`IntoParallelSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct RangeParallelSource<T> {
    range: Range<T>,
}

#[cfg(feature = "nightly")]
impl<T: Step + Copy + Send + Sync> IntoParallelSource for Range<T> {
    type Item = T;
    type Source = RangeParallelSource<T>;

    fn into_par_iter(self) -> Self::Source {
        RangeParallelSource { range: self }
    }
}

#[cfg(feature = "nightly")]
impl<T: Step + Copy + Send + Sync> ParallelSource for RangeParallelSource<T> {
    type Item = T;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let range = self.range;
        let (len_hint, len) = T::steps_between(&range.start, &range.end);
        let len = len.unwrap_or_else(|| {
            if len_hint == 0 {
                panic!("cannot iterate over a backward range");
            } else {
                panic!(
                    "cannot iterate over a range with more than usize::MAX items ({})",
                    usize::MAX
                );
            }
        });
        SourceDescriptor {
            len,
            fetch_item: move |index| T::forward(range.start, index),
        }
    }
}

#[cfg(not(feature = "nightly"))]
impl IntoParallelSource for Range<usize> {
    type Item = usize;
    type Source = RangeParallelSource<usize>;

    fn into_par_iter(self) -> Self::Source {
        RangeParallelSource { range: self }
    }
}

#[cfg(not(feature = "nightly"))]
impl ParallelSource for RangeParallelSource<usize> {
    type Item = usize;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let range = self.range;
        let len = range
            .end
            .checked_sub(range.start)
            .expect("cannot iterate over a backward range");
        SourceDescriptor {
            len,
            fetch_item: move |index| range.start + index,
        }
    }
}

/// A parallel source over a [`RangeInclusive`]. This struct is created by the
/// [`into_par_iter()`](IntoParallelSource::into_par_iter) method on
/// [`IntoParallelSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct RangeInclusiveParallelSource<T> {
    range: RangeInclusive<T>,
}

#[cfg(feature = "nightly")]
impl<T: Step + Copy + Send + Sync> IntoParallelSource for RangeInclusive<T> {
    type Item = T;
    type Source = RangeInclusiveParallelSource<T>;

    fn into_par_iter(self) -> Self::Source {
        RangeInclusiveParallelSource { range: self }
    }
}

#[cfg(feature = "nightly")]
impl<T: Step + Copy + Send + Sync> ParallelSource for RangeInclusiveParallelSource<T> {
    type Item = T;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let (start, end) = self.range.into_inner();
        let (len_hint, len) = T::steps_between(&start, &end);
        let len = len.unwrap_or_else(|| {
            if len_hint == 0 {
                panic!("cannot iterate over a backward range");
            } else {
                panic!(
                    "cannot iterate over a range with more than usize::MAX items ({})",
                    usize::MAX
                );
            }
        });
        let len = len.checked_add(1).unwrap_or_else(|| {
            panic!(
                "cannot iterate over a range with more than usize::MAX items ({})",
                usize::MAX
            );
        });
        SourceDescriptor {
            len,
            fetch_item: move |index| T::forward(start, index),
        }
    }
}

#[cfg(not(feature = "nightly"))]
impl IntoParallelSource for RangeInclusive<usize> {
    type Item = usize;
    type Source = RangeInclusiveParallelSource<usize>;

    fn into_par_iter(self) -> Self::Source {
        RangeInclusiveParallelSource { range: self }
    }
}

#[cfg(not(feature = "nightly"))]
impl ParallelSource for RangeInclusiveParallelSource<usize> {
    type Item = usize;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let (start, end) = self.range.into_inner();
        let len = end
            .checked_sub(start)
            .expect("cannot iterate over a backward range");
        let len = len.checked_add(1).unwrap_or_else(|| {
            panic!(
                "cannot iterate over a range with more than usize::MAX items ({})",
                usize::MAX
            );
        });
        SourceDescriptor {
            len,
            fetch_item: move |index| start + index,
        }
    }
}
