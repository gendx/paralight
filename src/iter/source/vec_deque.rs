// Copyright 2025-2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{
    ExactParallelSource, ExactParallelSourceExt, ExactSourceDescriptor,
    IntoExactParallelRefMutSource, IntoExactParallelRefSource,
};
use std::collections::VecDeque;

impl<'data, T: Sync + 'data> IntoExactParallelRefSource<'data> for VecDeque<T> {
    type Item = &'data T;

    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::VecDeque;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let input: VecDeque<_> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11);
    /// ```
    fn par_iter(&'data self) -> impl ExactParallelSource<Item = Self::Item> {
        VecDequeRefParallelSource { vec_deque: self }
    }
}

struct VecDequeRefParallelSource<'data, T> {
    vec_deque: &'data VecDeque<T>,
}

impl<'data, T: Sync> ExactParallelSource for VecDequeRefParallelSource<'data, T> {
    type Item = &'data T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let (first, second) = self.vec_deque.as_slices();
        first.par_iter().chain(second.par_iter()).exact_descriptor()
    }
}

impl<'data, T: Send + 'data> IntoExactParallelRefMutSource<'data> for VecDeque<T> {
    type Item = &'data mut T;

    /// ```
    /// # use paralight::prelude::*;
    /// # use std::collections::VecDeque;
    /// # let mut thread_pool = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # }
    /// # .build();
    /// let mut values: VecDeque<_> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();
    /// values
    ///     .par_iter_mut()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .for_each(|x| *x *= 2);
    /// assert_eq!(
    ///     values,
    ///     [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    ///         .into_iter()
    ///         .collect::<VecDeque<_>>()
    /// );
    /// ```
    fn par_iter_mut(&'data mut self) -> impl ExactParallelSource<Item = Self::Item> {
        VecDequeRefMutParallelSource { vec_deque: self }
    }
}

struct VecDequeRefMutParallelSource<'data, T> {
    vec_deque: &'data mut VecDeque<T>,
}

impl<'data, T: Send> ExactParallelSource for VecDequeRefMutParallelSource<'data, T> {
    type Item = &'data mut T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let (first, second) = self.vec_deque.as_mut_slices();
        first
            .par_iter_mut()
            .chain(second.par_iter_mut())
            .exact_descriptor()
    }
}
