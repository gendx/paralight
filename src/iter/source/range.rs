// Copyright 2024-2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{ExactParallelSource, ExactSourceDescriptor, IntoExactParallelSource, SourceCleanup};
#[cfg(feature = "nightly")]
use std::iter::Step;
use std::ops::{Range, RangeInclusive};

pub struct RangeSourceDescriptor<T> {
    start: T,
    len: usize,
}

impl<T> SourceCleanup for RangeSourceDescriptor<T> {
    const NEEDS_CLEANUP: bool = false;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn cleanup_item_range(&self, _range: Range<usize>) {
        // Nothing to cleanup
    }
}

#[cfg(feature = "nightly")]
impl<T: Step + Copy + Send + Sync> ExactSourceDescriptor for RangeSourceDescriptor<T> {
    type Item = T;

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        T::forward(self.start, index)
    }
}

#[cfg(not(feature = "nightly"))]
impl ExactSourceDescriptor for RangeSourceDescriptor<usize> {
    type Item = usize;

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        self.start + index
    }
}

#[cfg(feature = "nightly")]
/// ### Stability blockers
///
/// On stable Rust, this implementation is currently only available for ranges
/// of [`usize`]. Ranges over any [`Step`] type are only available on Rust
/// nightly with the `nightly` feature of Paralight enabled. This is because the
/// implementation depends on the
/// [`step_trait`](https://github.com/rust-lang/rust/issues/42168) nightly Rust
/// feature.
impl<T: Step + Copy + Send + Sync> IntoExactParallelSource for Range<T> {
    type Item = T;

    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let sum = (1..10)
    ///     .into_par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<usize>();
    /// assert_eq!(sum, 5 * 9);
    /// ```
    fn into_par_iter(self) -> impl ExactParallelSource<Item = Self::Item> {
        RangeParallelSource { range: self }
    }
}

#[cfg(not(feature = "nightly"))]
/// ### Stability blockers
///
/// On stable Rust, this implementation is currently only available for ranges
/// of [`usize`]. Ranges over any [`Step`] type are only available on Rust
/// nightly with the `nightly` feature of Paralight enabled. This is because the
/// implementation depends on the
/// [`step_trait`](https://github.com/rust-lang/rust/issues/42168) nightly Rust
/// feature.
impl IntoExactParallelSource for Range<usize> {
    type Item = usize;

    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let sum = (1..10)
    ///     .into_par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<usize>();
    /// assert_eq!(sum, 5 * 9);
    /// ```
    fn into_par_iter(self) -> impl ExactParallelSource<Item = Self::Item> {
        RangeParallelSource { range: self }
    }
}

struct RangeParallelSource<T> {
    range: Range<T>,
}

#[cfg(feature = "nightly")]
impl<T: Step + Copy + Send + Sync> ExactParallelSource for RangeParallelSource<T> {
    type Item = T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
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

#[cfg(not(feature = "nightly"))]
impl ExactParallelSource for RangeParallelSource<usize> {
    type Item = usize;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
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

#[cfg(feature = "nightly")]
/// ### Stability blockers
///
/// On stable Rust, this implementation is currently only available for ranges
/// of [`usize`]. Ranges over any [`Step`] type are only available on Rust
/// nightly with the `nightly` feature of Paralight enabled. This is because the
/// implementation depends on the
/// [`step_trait`](https://github.com/rust-lang/rust/issues/42168) nightly Rust
/// feature.
impl<T: Step + Copy + Send + Sync> IntoExactParallelSource for RangeInclusive<T> {
    type Item = T;

    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let sum = (1..=10)
    ///     .into_par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<usize>();
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn into_par_iter(self) -> impl ExactParallelSource<Item = Self::Item> {
        RangeInclusiveParallelSource { range: self }
    }
}

#[cfg(not(feature = "nightly"))]
/// ### Stability blockers
///
/// On stable Rust, this implementation is currently only available for ranges
/// of [`usize`]. Ranges over any [`Step`] type are only available on Rust
/// nightly with the `nightly` feature of Paralight enabled. This is because the
/// implementation depends on the
/// [`step_trait`](https://github.com/rust-lang/rust/issues/42168) nightly Rust
/// feature.
impl IntoExactParallelSource for RangeInclusive<usize> {
    type Item = usize;

    /// ```
    /// # use paralight::prelude::*;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let sum = (1..=10)
    ///     .into_par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<usize>();
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn into_par_iter(self) -> impl ExactParallelSource<Item = Self::Item> {
        RangeInclusiveParallelSource { range: self }
    }
}

struct RangeInclusiveParallelSource<T> {
    range: RangeInclusive<T>,
}

#[cfg(feature = "nightly")]
impl<T: Step + Copy + Send + Sync> ExactParallelSource for RangeInclusiveParallelSource<T> {
    type Item = T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
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

#[cfg(not(feature = "nightly"))]
impl ExactParallelSource for RangeInclusiveParallelSource<usize> {
    type Item = usize;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
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
