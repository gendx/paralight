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

/// A parallel source over a reference to a [`VecDeque`]. This struct is created
/// by the [`par_iter()`](IntoExactParallelRefSource::par_iter) method on
/// [`IntoExactParallelRefSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSource`] and [`ExactParallelSourceExt`]
/// traits, but it is nonetheless public because of the `must_use` annotation.
///
/// See also [`VecDequeRefMutParallelSource`].
///
/// ```
/// # use paralight::iter::VecDequeRefParallelSource;
/// # use paralight::prelude::*;
/// # use std::collections::VecDeque;
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let input: VecDeque<_> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();
/// let iter: VecDequeRefParallelSource<_> = input.par_iter();
/// let sum = iter.with_thread_pool(&mut thread_pool).sum::<i32>();
/// assert_eq!(sum, 5 * 11);
/// ```
#[must_use = "iterator adaptors are lazy"]
pub struct VecDequeRefParallelSource<'data, T> {
    vec_deque: &'data VecDeque<T>,
}

impl<'data, T: Sync + 'data> IntoExactParallelRefSource<'data> for VecDeque<T> {
    type Item = &'data T;
    type Source = VecDequeRefParallelSource<'data, T>;

    fn par_iter(&'data self) -> Self::Source {
        VecDequeRefParallelSource { vec_deque: self }
    }
}

impl<'data, T: Sync> ExactParallelSource for VecDequeRefParallelSource<'data, T> {
    type Item = &'data T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let (first, second) = self.vec_deque.as_slices();
        first.par_iter().chain(second.par_iter()).exact_descriptor()
    }
}

/// A parallel source over a mutable reference to a [`VecDeque`]. This struct is
/// created by the
/// [`par_iter_mut()`](IntoExactParallelRefMutSource::par_iter_mut)
/// method on [`IntoExactParallelRefMutSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSource`] and [`ExactParallelSourceExt`]
/// traits, but it is nonetheless public because of the `must_use` annotation.
///
/// See also [`VecDequeRefParallelSource`].
///
/// ```
/// # use paralight::iter::VecDequeRefMutParallelSource;
/// # use paralight::prelude::*;
/// # use std::collections::VecDeque;
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// let mut values: VecDeque<_> = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_iter().collect();
/// let iter: VecDequeRefMutParallelSource<_> = values.par_iter_mut();
/// iter.with_thread_pool(&mut thread_pool)
///     .for_each(|x| *x *= 2);
/// assert_eq!(
///     values,
///     [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
///         .into_iter()
///         .collect::<VecDeque<_>>()
/// );
/// ```
#[must_use = "iterator adaptors are lazy"]
pub struct VecDequeRefMutParallelSource<'data, T> {
    vec_deque: &'data mut VecDeque<T>,
}

impl<'data, T: Send + 'data> IntoExactParallelRefMutSource<'data> for VecDeque<T> {
    type Item = &'data mut T;
    type Source = VecDequeRefMutParallelSource<'data, T>;

    fn par_iter_mut(&'data mut self) -> Self::Source {
        VecDequeRefMutParallelSource { vec_deque: self }
    }
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
