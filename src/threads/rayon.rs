// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Adaptors over Rayon thread pools.

use super::{RangeStrategy, ThreadCount};
use crate::core::pipeline::{IterPipelineImpl, Pipeline, UpperBoundedPipelineImpl};
use crate::core::range::{
    FixedRangeFactory, RangeFactory, RangeOrchestrator, WorkStealingRangeFactory,
};
use crate::iter::{Accumulator, GenericThreadPool, SourceCleanup};
use crossbeam_utils::CachePadded;
use rayon_core::{Scope, ThreadPool};
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

/// Adaptor to execute Paralight iterators over a thread pool provided by the [Rayon](https://docs.rs/rayon) crate.
///
/// This type implements the [`GenericThreadPool`] trait, allowing to use the
/// thread pool with Paralight iterators via the
/// [`with_thread_pool()`](crate::iter::ParallelSourceExt::with_thread_pool)
/// adaptor.
///
/// ```
/// # // TODO: Enable Miri once supported by Rayon and its dependencies: https://github.com/crossbeam-rs/crossbeam/issues/1181.
/// # #[cfg(not(miri))]
/// # {
/// # use paralight::prelude::*;
/// let thread_pool = RayonThreadPool::new_global(
///     ThreadCount::try_from(rayon_core::current_num_threads())
///         .expect("Paralight cannot operate with 0 threads"),
///     RangeStrategy::WorkStealing,
/// );
///
/// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let sum = input.par_iter().with_thread_pool(&thread_pool).sum::<i32>();
/// assert_eq!(sum, 5 * 11);
///
/// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
/// let product = input.par_iter().with_thread_pool(&thread_pool).product::<i32>();
/// assert_eq!(product, 3_628_800);
/// # }
/// ```
pub struct RayonThreadPool<'a> {
    inner: RayonThreadPoolEnum<'a>,
}

impl RayonThreadPool<'static> {
    /// Wraps [Rayon](https://docs.rs/rayon)'s global thread pool for use with
    /// Paralight iterators, spawning the given number of Paralight tasks and
    /// using the given parallelism strategy.
    ///
    /// As a guiding principle, the number of tasks should match the number of
    /// threads in Rayon's pool, but other choices are possible. Note that
    /// each task may be executed by any thread in the pool, so there is no
    /// guarantee of a one-to-one match between Paralight tasks and Rayon
    /// threads (for a one-to-one match, use Paralight's built-in
    /// [`ThreadPool`](crate::threads::ThreadPool)). This is especially true if
    /// the Rayon thread pool is also executing other tasks.
    ///
    /// Spawning fewer tasks limits the amount of parallelism, which might be
    /// desirable if other work is being executed in parallel (on the Rayon
    /// thread pool, on other threads in the progream, or on other processes
    /// on your system).
    ///
    /// Spawning more tasks than available threads is likely not beneficial as
    /// it adds more overhead, especially with [`RangeStrategy::WorkStealing`].
    /// It might make sense with [`RangeStrategy::Fixed`] to introduce a bit of
    /// work-stealing in the mix. As always: benchmark your code for your use
    /// case.
    ///
    /// ```
    /// # // TODO: Enable Miri once supported by Rayon and its dependencies: https://github.com/crossbeam-rs/crossbeam/issues/1181.
    /// # #[cfg(not(miri))]
    /// # {
    /// # use paralight::prelude::*;
    /// let thread_pool = RayonThreadPool::new_global(
    ///     ThreadCount::try_from(rayon_core::current_num_threads())
    ///         .expect("Paralight cannot operate with 0 threads"),
    ///     RangeStrategy::WorkStealing,
    /// );
    ///
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input.par_iter().with_thread_pool(&thread_pool).sum::<i32>();
    /// assert_eq!(sum, 5 * 11);
    /// # }
    /// ```
    pub fn new_global(num_tasks: ThreadCount, range_strategy: RangeStrategy) -> Self {
        Self {
            inner: RayonThreadPoolEnum::new(None, num_tasks, range_strategy),
        }
    }
}

impl<'a> RayonThreadPool<'a> {
    /// Wrap a user-created [Rayon](https://docs.rs/rayon) thread pool for use
    /// with Paralight iterators, spawning the given number of Paralight tasks
    /// and using the given parallelism strategy.
    ///
    /// As a guiding principle, the number of tasks should match the number of
    /// threads in Rayon's pool, but other choices are possible. Note that
    /// each task may be executed by any thread in the pool, so there is no
    /// guarantee of a one-to-one match between Paralight tasks and Rayon
    /// threads (for a one-to-one match, use Paralight's built-in
    /// [`ThreadPool`](crate::threads::ThreadPool)). This is especially true if
    /// the Rayon thread pool is also executing other tasks.
    ///
    /// Spawning fewer tasks limits the amount of parallelism, which might be
    /// desirable if other work is being executed in parallel (on the Rayon
    /// thread pool, on other threads in the progream, or on other processes
    /// on your system).
    ///
    /// Spawning more tasks than available threads is likely not beneficial as
    /// it adds more overhead, especially with [`RangeStrategy::WorkStealing`].
    /// It might make sense with [`RangeStrategy::Fixed`] to introduce a bit of
    /// work-stealing in the mix. As always: benchmark your code for your use
    /// case.
    ///
    /// ```
    /// # // TODO: Enable Miri once supported by Rayon and its dependencies: https://github.com/crossbeam-rs/crossbeam/issues/1181.
    /// # #[cfg(not(miri))]
    /// # {
    /// # use paralight::prelude::*;
    /// // Create a custom Rayon thread pool.
    /// let thread_pool = rayon_core::ThreadPoolBuilder::new()
    ///     .num_threads(std::thread::available_parallelism().unwrap().into())
    ///     .build()
    ///     .unwrap();
    ///
    /// // Wrap it for use with Paralight.
    /// let thread_pool_wrapper = RayonThreadPool::new(
    ///     &thread_pool,
    ///     ThreadCount::AvailableParallelism,
    ///     RangeStrategy::WorkStealing,
    /// );
    ///
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&thread_pool_wrapper)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11);
    /// # }
    /// ```
    pub fn new(
        thread_pool: &'a ThreadPool,
        num_tasks: ThreadCount,
        range_strategy: RangeStrategy,
    ) -> Self {
        Self {
            inner: RayonThreadPoolEnum::new(Some(thread_pool), num_tasks, range_strategy),
        }
    }
}

impl RayonThreadPool<'_> {
    /// Returns the number of Paralight tasks that are spawned by this thread
    /// pool wrapper.
    pub fn num_tasks(&self) -> NonZeroUsize {
        self.inner.num_tasks()
    }
}

// SAFETY: Proof of the safety guarantees is deferred to the inner calls.
unsafe impl GenericThreadPool for &RayonThreadPool<'_> {
    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        input_len: usize,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // Proof of the safety guarantees is deferred to the inner function.
        self.inner
            .upper_bounded_pipeline(input_len, init, process_item, finalize, reduce, cleanup)
    }

    fn iter_pipeline<Output: Send>(
        self,
        input_len: usize,
        accum: impl Accumulator<usize, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // Proof of the safety guarantees is deferred to the inner function.
        self.inner.iter_pipeline(input_len, accum, reduce, cleanup)
    }
}

/// Underlying [`RayonThreadPool`] implementation, dispatching over the
/// [`RangeStrategy`].
enum RayonThreadPoolEnum<'a> {
    Fixed(RayonThreadPoolImpl<'a, FixedRangeFactory>),
    WorkStealing(RayonThreadPoolImpl<'a, WorkStealingRangeFactory>),
}

impl<'a> RayonThreadPoolEnum<'a> {
    /// Creates a new thread pool using the given parameters.
    fn new(
        thread_pool: Option<&'a ThreadPool>,
        num_tasks: ThreadCount,
        range_strategy: RangeStrategy,
    ) -> Self {
        let num_tasks: NonZeroUsize = num_tasks.count();
        let num_tasks: usize = num_tasks.into();
        match range_strategy {
            RangeStrategy::Fixed => RayonThreadPoolEnum::Fixed(RayonThreadPoolImpl::new(
                thread_pool,
                num_tasks,
                FixedRangeFactory::new(num_tasks),
            )),
            RangeStrategy::WorkStealing => {
                RayonThreadPoolEnum::WorkStealing(RayonThreadPoolImpl::new(
                    thread_pool,
                    num_tasks,
                    WorkStealingRangeFactory::new(num_tasks),
                ))
            }
        }
    }

    /// Returns the number of Paralight tasks that are spawned by this thread
    /// pool wrapper.
    fn num_tasks(&self) -> NonZeroUsize {
        match self {
            RayonThreadPoolEnum::Fixed(inner) => inner.num_tasks(),
            RayonThreadPoolEnum::WorkStealing(inner) => inner.num_tasks(),
        }
    }

    /// Processes an input of the given length in parallel and returns the
    /// aggregated output.
    ///
    /// With this variant, the pipeline may skip processing items at larger
    /// indices whenever a call to `process_item` returns
    /// [`ControlFlow::Break`].
    ///
    /// # Safety guarantees
    ///
    /// This function guarantees that:
    /// - the indices passed to `process_item()` are in `0..input_len`,
    /// - the ranges passed to `cleanup.cleanup_item_range()` are included in
    ///   `0..input_len`,
    /// - each index in `0..inner_len` is passed exactly once in calls to
    ///   `process_item()` and `cleanup.cleanup_item_range()`.
    fn upper_bounded_pipeline<Output: Send, Accum>(
        &self,
        input_len: usize,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // Proof of the safety guarantees is deferred to the inner function.
        match self {
            RayonThreadPoolEnum::Fixed(inner) => inner.upper_bounded_pipeline(
                input_len,
                init,
                process_item,
                finalize,
                reduce,
                cleanup,
            ),
            RayonThreadPoolEnum::WorkStealing(inner) => inner.upper_bounded_pipeline(
                input_len,
                init,
                process_item,
                finalize,
                reduce,
                cleanup,
            ),
        }
    }

    /// Processes an input of the given length in parallel and returns the
    /// aggregated output.
    ///
    /// # Safety guarantees
    ///
    /// This function guarantees that:
    /// - the indices passed to `accum.accumulate()` are in `0..input_len`,
    /// - the ranges passed to `cleanup.cleanup_item_range()` are included in
    ///   `0..input_len`,
    /// - each index in `0..inner_len` is passed exactly once in calls to
    ///   `accum.accumulate()` and `cleanup.cleanup_item_range()`.
    fn iter_pipeline<Output: Send>(
        &self,
        input_len: usize,
        accum: impl Accumulator<usize, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // Proof of the safety guarantees is deferred to the inner function.
        match self {
            RayonThreadPoolEnum::Fixed(inner) => {
                inner.iter_pipeline(input_len, accum, reduce, cleanup)
            }
            RayonThreadPoolEnum::WorkStealing(inner) => {
                inner.iter_pipeline(input_len, accum, reduce, cleanup)
            }
        }
    }
}

/// Underlying [`RayonThreadPool`] implementation, specialized to a
/// [`RangeStrategy`].
struct RayonThreadPoolImpl<'a, F: RangeFactory> {
    /// Handle to the Rayon thread pool, or [`None`] if using the global Rayon
    /// thread pool.
    thread_pool: Option<&'a ThreadPool>,
    /// Orchestrator for the work ranges distributed to the tasks.
    range_orchestrator: F::Orchestrator,
    /// Range of items that each task needs to process.
    ranges: Box<[F::Range]>,
}

impl<'a, F: RangeFactory> RayonThreadPoolImpl<'a, F> {
    /// Creates a new thread pool using the given parameters.
    fn new(thread_pool: Option<&'a ThreadPool>, num_tasks: usize, range_factory: F) -> Self {
        let ranges = (0..num_tasks).map(|id| range_factory.range(id)).collect();
        Self {
            thread_pool,
            range_orchestrator: range_factory.orchestrator(),
            ranges,
        }
    }

    /// Returns the number of Paralight tasks that are spawned by this thread
    /// pool wrapper.
    fn num_tasks(&self) -> NonZeroUsize {
        self.ranges.len().try_into().unwrap()
    }
}

impl<F: RangeFactory> RayonThreadPoolImpl<'_, F> {
    /// Creates a fork-join scope on the underlying Rayon thread pool and
    /// invokes the closure with a reference to the scope.
    fn scope<'scope, OP, R>(&self, op: OP) -> R
    where
        OP: FnOnce(&Scope<'scope>) -> R + Send,
        R: Send,
    {
        match self.thread_pool {
            None => rayon_core::scope(op),
            Some(thread_pool) => thread_pool.scope(op),
        }
    }
}

impl<F: RangeFactory> RayonThreadPoolImpl<'_, F>
where
    F::Range: Sync,
{
    /// Processes an input of the given length in parallel and returns the
    /// aggregated output.
    ///
    /// With this variant, the pipeline may skip processing items at larger
    /// indices whenever a call to `process_item` returns
    /// [`ControlFlow::Break`].
    ///
    /// # Safety guarantees
    ///
    /// This function guarantees that:
    /// - the indices passed to `process_item()` are in `0..input_len`,
    /// - the ranges passed to `cleanup.cleanup_item_range()` are included in
    ///   `0..input_len`,
    /// - each index in `0..inner_len` is passed exactly once in calls to
    ///   `process_item()` and `cleanup.cleanup_item_range()`.
    fn upper_bounded_pipeline<Output: Send, Accum>(
        &self,
        input_len: usize,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // The safety guarantees derive from this call as well as how the
        // `UpperBoundedPipelineImpl` uses a `SkipIteratorWrapper` on each worker
        // thread's range.
        self.range_orchestrator.reset_ranges(input_len);

        let num_tasks = self.ranges.len();
        let outputs = (0..num_tasks)
            .map(|_| Mutex::new(None))
            .collect::<Arc<[_]>>();
        let bound = AtomicUsize::new(usize::MAX);

        let pipeline = &UpperBoundedPipelineImpl {
            bound: CachePadded::new(bound),
            outputs: outputs.clone(),
            init,
            process_item,
            finalize,
            cleanup,
        };

        let ranges = &self.ranges;
        self.scope({
            |scope| {
                for (id, range) in ranges.iter().enumerate() {
                    scope.spawn(move |_| {
                        pipeline.run(id, range);
                    });
                }
            }
        });

        outputs
            .iter()
            .map(move |output| output.lock().unwrap().take().unwrap())
            .reduce(reduce)
            .unwrap()
    }

    /// Processes an input of the given length in parallel and returns the
    /// aggregated output.
    ///
    /// # Safety guarantees
    ///
    /// This function guarantees that:
    /// - the indices passed to `accum.accumulate()` are in `0..input_len`,
    /// - the ranges passed to `cleanup.cleanup_item_range()` are included in
    ///   `0..input_len`,
    /// - each index in `0..inner_len` is passed exactly once in calls to
    ///   `accum.accumulate()` and `cleanup.cleanup_item_range()`.
    fn iter_pipeline<Output: Send>(
        &self,
        input_len: usize,
        accum: impl Accumulator<usize, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // The safety guarantees derive from this call as well as how the
        // `IterPipelineImpl` uses a `SkipIteratorWrapper` on each worker thread's
        // range.
        self.range_orchestrator.reset_ranges(input_len);

        let num_tasks = self.ranges.len();
        let outputs = (0..num_tasks)
            .map(|_| Mutex::new(None))
            .collect::<Arc<[_]>>();

        let pipeline = &IterPipelineImpl {
            outputs: outputs.clone(),
            accum,
            cleanup,
        };

        let ranges = &self.ranges;
        self.scope({
            |scope| {
                for (id, range) in ranges.iter().enumerate() {
                    scope.spawn(move |_| {
                        pipeline.run(id, range);
                    });
                }
            }
        });

        reduce.accumulate(
            outputs
                .iter()
                .map(move |output| output.lock().unwrap().take().unwrap()),
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_num_tasks() {
        for range_strategy in [RangeStrategy::Fixed, RangeStrategy::WorkStealing] {
            let thread_pool =
                RayonThreadPool::new_global(ThreadCount::AvailableParallelism, range_strategy);
            assert_eq!(
                thread_pool.num_tasks(),
                std::thread::available_parallelism().unwrap()
            );

            let thread_pool =
                RayonThreadPool::new_global(ThreadCount::try_from(4).unwrap(), range_strategy);
            assert_eq!(thread_pool.num_tasks(), NonZeroUsize::try_from(4).unwrap());
        }
    }
}
