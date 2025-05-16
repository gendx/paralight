// Copyright 2024-2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{
    IntoParallelSource, ParallelSource, RewindableSource, SourceCleanup, SourceDescriptor,
};
#[cfg(feature = "nightly")]
use std::iter::Step;
use std::ops::{Range, RangeInclusive};

struct RangeSourceDescriptor<T> {
    start: T,
    len: usize,
}

impl<T> SourceCleanup for RangeSourceDescriptor<T> {
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: Range<usize>) {
        // Nothing to cleanup
    }
}

#[cfg(feature = "nightly")]
impl<T: Step + Copy + Send + Sync> SourceDescriptor for RangeSourceDescriptor<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        T::forward(self.start, index)
    }
}

#[cfg(not(feature = "nightly"))]
impl SourceDescriptor for RangeSourceDescriptor<usize> {
    type Item = usize;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        self.start + index
    }
}

/// A parallel source over a [`Range`]. This struct is created by the
/// [`into_par_iter()`](IntoParallelSource::into_par_iter) method on
/// [`IntoParallelSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
///
/// See also [`RangeInclusiveParallelSource`].
///
/// ### Stability blockers
///
/// On stable Rust, this struct is currently only implemented for ranges of
/// [`usize`]. Ranges over any [`Step`] type are only available on Rust nightly
/// with the `nightly` feature of Paralight enabled. This is because the
/// implementation depends on the
/// [`step_trait`](https://github.com/rust-lang/rust/issues/42168) nightly Rust
/// feature.
///
/// ```
/// # use paralight::iter::{
/// #     IntoParallelSource, ParallelIteratorExt, ParallelSourceExt, RangeParallelSource,
/// # };
/// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let iter: RangeParallelSource<usize> = (1..10).into_par_iter();
/// let sum = iter.with_thread_pool(&mut thread_pool).sum::<usize>();
/// assert_eq!(sum, 5 * 9);
/// ```
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

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
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
        RangeSourceDescriptor {
            start: range.start,
            len,
        }
    }
}

#[cfg(feature = "nightly")]
// SAFETY: TODO
unsafe impl<T: Step + Copy + Send + Sync> RewindableSource for RangeParallelSource<T> {}

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

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let range = self.range;
        let len = range
            .end
            .checked_sub(range.start)
            .expect("cannot iterate over a backward range");
        RangeSourceDescriptor {
            start: range.start,
            len,
        }
    }
}

#[cfg(not(feature = "nightly"))]
// SAFETY: TODO
unsafe impl RewindableSource for RangeParallelSource<usize> {}

/// A parallel source over a [`RangeInclusive`]. This struct is created by the
/// [`into_par_iter()`](IntoParallelSource::into_par_iter) method on
/// [`IntoParallelSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
///
/// See also [`RangeParallelSource`].
///
/// ### Stability blockers
///
/// On stable Rust, this struct is currently only implemented for ranges of
/// [`usize`]. Ranges over any [`Step`] type are only available on Rust nightly
/// with the `nightly` feature of Paralight enabled. This is because the
/// implementation depends on the
/// [`step_trait`](https://github.com/rust-lang/rust/issues/42168) nightly Rust
/// feature.
///
/// ```
/// # use paralight::iter::{
/// #     IntoParallelSource, ParallelIteratorExt, ParallelSourceExt, RangeInclusiveParallelSource,
/// # };
/// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let iter: RangeInclusiveParallelSource<usize> = (1..=10).into_par_iter();
/// let sum = iter.with_thread_pool(&mut thread_pool).sum::<usize>();
/// assert_eq!(sum, 5 * 11);
/// ```
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

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
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
        RangeSourceDescriptor { start, len }
    }
}

#[cfg(feature = "nightly")]
// SAFETY: TODO
unsafe impl<T: Step + Copy + Send + Sync> RewindableSource for RangeInclusiveParallelSource<T> {}

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

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
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
        RangeSourceDescriptor { start, len }
    }
}

#[cfg(not(feature = "nightly"))]
// SAFETY: TODO
unsafe impl RewindableSource for RangeInclusiveParallelSource<usize> {}
