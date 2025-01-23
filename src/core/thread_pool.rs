// Copyright 2024-2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A thread pool implementing parallelism at a lightweight cost.

use super::range::{
    FixedRangeFactory, Range, RangeFactory, RangeOrchestrator, SkipIterator,
    WorkStealingRangeFactory,
};
use super::sync::{make_lending_group, Borrower, Lender, WorkerState};
use super::util::LifetimeParameterized;
use crate::iter::{Accumulator, SourceCleanup};
use crate::macros::{log_debug, log_error, log_warn};
use crossbeam_utils::CachePadded;
// Platforms that support `libc::sched_setaffinity()`.
#[cfg(all(
    not(miri),
    any(
        target_os = "android",
        target_os = "dragonfly",
        target_os = "freebsd",
        target_os = "linux"
    )
))]
use nix::{
    sched::{sched_setaffinity, CpuSet},
    unistd::Pid,
};
use std::convert::TryFrom;
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

/// Number of threads to spawn in a thread pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadCount {
    /// Spawn the number of threads returned by
    /// [`std::thread::available_parallelism()`].
    AvailableParallelism,
    /// Spawn the given number of threads.
    Count(NonZeroUsize),
}

impl TryFrom<usize> for ThreadCount {
    type Error = <NonZeroUsize as TryFrom<usize>>::Error;

    fn try_from(thread_count: usize) -> Result<Self, Self::Error> {
        let count = NonZeroUsize::try_from(thread_count)?;
        Ok(ThreadCount::Count(count))
    }
}

/// Strategy to distribute ranges of work items among threads.
#[derive(Clone, Copy)]
pub enum RangeStrategy {
    /// Each thread processes a fixed range of items.
    Fixed,
    /// Threads can steal work from each other.
    WorkStealing,
}

/// Policy to pin worker threads to CPUs.
#[derive(Clone, Copy)]
pub enum CpuPinningPolicy {
    /// Don't pin worker threads to CPUs.
    No,
    /// Pin each worker thread to a CPU, if CPU pinning is supported and
    /// implemented on this platform.
    IfSupported,
    /// Pin each worker thread to a CPU. If CPU pinning isn't supported on this
    /// platform (or not implemented), building a thread pool will panic.
    Always,
}

/// A builder for [`ThreadPool`].
pub struct ThreadPoolBuilder {
    /// Number of worker threads to spawn in the pool.
    pub num_threads: ThreadCount,
    /// Strategy to distribute ranges of work items among threads.
    pub range_strategy: RangeStrategy,
    /// Policy to pin worker threads to CPUs.
    pub cpu_pinning: CpuPinningPolicy,
}

impl ThreadPoolBuilder {
    /// Spawns a thread pool.
    ///
    /// ```
    /// # use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
    /// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
    /// let pool_builder = ThreadPoolBuilder {
    ///     num_threads: ThreadCount::AvailableParallelism,
    ///     range_strategy: RangeStrategy::WorkStealing,
    ///     cpu_pinning: CpuPinningPolicy::No,
    /// };
    /// let mut thread_pool = pool_builder.build();
    ///
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter()
    ///     .with_thread_pool(&mut thread_pool)
    ///     .sum::<i32>();
    /// assert_eq!(sum, 5 * 11);
    /// ```
    pub fn build(&self) -> ThreadPool {
        ThreadPool::new(self)
    }
}

/// A thread pool that can execute parallel pipelines.
///
/// This type doesn't expose any public methods other than
/// [`num_threads()`](Self::num_threads). You can interact with it via
/// the [`ThreadPoolBuilder::build()`] function to create a thread pool, and the
/// [`with_thread_pool()`](crate::iter::ParallelSourceExt::with_thread_pool)
/// method to attach a thread pool to a parallel iterator.
pub struct ThreadPool {
    inner: ThreadPoolEnum,
}

impl ThreadPool {
    /// Creates a new thread pool using the given parameters.
    fn new(builder: &ThreadPoolBuilder) -> Self {
        Self {
            inner: ThreadPoolEnum::new(builder),
        }
    }

    /// Returns the number of worker threads that have been spawned in this
    /// thread pool.
    pub fn num_threads(&self) -> NonZeroUsize {
        self.inner.num_threads()
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
    pub(crate) fn upper_bounded_pipeline<Output: Send, Accum>(
        &mut self,
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
    pub(crate) fn iter_pipeline<Output: Send>(
        &mut self,
        input_len: usize,
        accum: impl Accumulator<usize, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // Proof of the safety guarantees is deferred to the inner function.
        self.inner.iter_pipeline(input_len, accum, reduce, cleanup)
    }
}

/// Underlying [`ThreadPool`] implementation, dispatching over the
/// [`RangeStrategy`].
enum ThreadPoolEnum {
    Fixed(ThreadPoolImpl<FixedRangeFactory>),
    WorkStealing(ThreadPoolImpl<WorkStealingRangeFactory>),
}

impl ThreadPoolEnum {
    /// Creates a new thread pool using the given parameters.
    fn new(builder: &ThreadPoolBuilder) -> Self {
        let num_threads: NonZeroUsize = match builder.num_threads {
            ThreadCount::AvailableParallelism => std::thread::available_parallelism()
                .expect("Getting the available parallelism failed"),
            ThreadCount::Count(count) => count,
        };
        let num_threads: usize = num_threads.into();
        match builder.range_strategy {
            RangeStrategy::Fixed => ThreadPoolEnum::Fixed(ThreadPoolImpl::new(
                num_threads,
                FixedRangeFactory::new(num_threads),
                builder.cpu_pinning,
            )),
            RangeStrategy::WorkStealing => ThreadPoolEnum::WorkStealing(ThreadPoolImpl::new(
                num_threads,
                WorkStealingRangeFactory::new(num_threads),
                builder.cpu_pinning,
            )),
        }
    }

    /// Returns the number of worker threads that have been spawned in this
    /// thread pool.
    fn num_threads(&self) -> NonZeroUsize {
        match self {
            ThreadPoolEnum::Fixed(inner) => inner.num_threads(),
            ThreadPoolEnum::WorkStealing(inner) => inner.num_threads(),
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
        &mut self,
        input_len: usize,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // Proof of the safety guarantees is deferred to the inner function.
        match self {
            ThreadPoolEnum::Fixed(inner) => inner.upper_bounded_pipeline(
                input_len,
                init,
                process_item,
                finalize,
                reduce,
                cleanup,
            ),
            ThreadPoolEnum::WorkStealing(inner) => inner.upper_bounded_pipeline(
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
        &mut self,
        input_len: usize,
        accum: impl Accumulator<usize, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // Proof of the safety guarantees is deferred to the inner function.
        match self {
            ThreadPoolEnum::Fixed(inner) => inner.iter_pipeline(input_len, accum, reduce, cleanup),
            ThreadPoolEnum::WorkStealing(inner) => {
                inner.iter_pipeline(input_len, accum, reduce, cleanup)
            }
        }
    }
}

/// Underlying [`ThreadPool`] implementation, specialized to a
/// [`RangeStrategy`].
struct ThreadPoolImpl<F: RangeFactory> {
    /// Handles to all the worker threads in the pool.
    threads: Vec<WorkerThreadHandle>,
    /// Orchestrator for the work ranges distributed to the threads.
    range_orchestrator: F::Orchestrator,
    /// Pipeline to map and reduce inputs into an output.
    pipeline: Lender<DynLifetimeSyncPipeline<F::Range>>,
}

/// Handle to a worker thread in a thread pool.
struct WorkerThreadHandle {
    /// Thread handle object.
    handle: JoinHandle<()>,
}

impl<F: RangeFactory> ThreadPoolImpl<F> {
    /// Creates a new thread pool using the given parameters.
    fn new(num_threads: usize, range_factory: F, cpu_pinning: CpuPinningPolicy) -> Self
    where
        F::Range: Send + 'static,
    {
        let (lender, borrowers) = make_lending_group(num_threads);

        #[cfg(any(
            miri,
            not(any(
                target_os = "android",
                target_os = "dragonfly",
                target_os = "freebsd",
                target_os = "linux"
            ))
        ))]
        match cpu_pinning {
            CpuPinningPolicy::No => (),
            CpuPinningPolicy::IfSupported => {
                log_warn!("Pinning threads to CPUs is not implemented on this platform.")
            }
            CpuPinningPolicy::Always => {
                panic!("Pinning threads to CPUs is not implemented on this platform.")
            }
        }

        let threads = borrowers
            .into_iter()
            .enumerate()
            .map(|(id, borrower)| {
                let mut context = ThreadContext {
                    id,
                    range: range_factory.range(id),
                    pipeline: borrower,
                };
                WorkerThreadHandle {
                    handle: std::thread::spawn(move || {
                        #[cfg(all(
                            not(miri),
                            any(
                                target_os = "android",
                                target_os = "dragonfly",
                                target_os = "freebsd",
                                target_os = "linux"
                            )
                        ))]
                        match cpu_pinning {
                            CpuPinningPolicy::No => (),
                            CpuPinningPolicy::IfSupported => {
                                let mut cpu_set = CpuSet::new();
                                if let Err(_e) = cpu_set.set(id) {
                                    log_warn!("Failed to set CPU affinity for thread #{id}: {_e}");
                                } else if let Err(_e) =
                                    sched_setaffinity(Pid::from_raw(0), &cpu_set)
                                {
                                    log_warn!("Failed to set CPU affinity for thread #{id}: {_e}");
                                } else {
                                    log_debug!("Pinned thread #{id} to CPU #{id}");
                                }
                            }
                            CpuPinningPolicy::Always => {
                                let mut cpu_set = CpuSet::new();
                                if let Err(e) = cpu_set.set(id) {
                                    panic!("Failed to set CPU affinity for thread #{id}: {e}");
                                } else if let Err(e) = sched_setaffinity(Pid::from_raw(0), &cpu_set)
                                {
                                    panic!("Failed to set CPU affinity for thread #{id}: {e}");
                                } else {
                                    log_debug!("Pinned thread #{id} to CPU #{id}");
                                }
                            }
                        }
                        context.run()
                    }),
                }
            })
            .collect();
        log_debug!("[main thread] Spawned threads");

        Self {
            threads,
            range_orchestrator: range_factory.orchestrator(),
            pipeline: lender,
        }
    }

    /// Returns the number of worker threads that have been spawned in this
    /// thread pool.
    fn num_threads(&self) -> NonZeroUsize {
        self.threads.len().try_into().unwrap()
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
        &mut self,
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

        let num_threads = self.threads.len();
        let outputs = (0..num_threads)
            .map(|_| Mutex::new(None))
            .collect::<Arc<[_]>>();
        let bound = AtomicUsize::new(usize::MAX);

        self.pipeline.lend(&UpperBoundedPipelineImpl {
            bound: CachePadded::new(bound),
            outputs: outputs.clone(),
            init,
            process_item,
            finalize,
            cleanup,
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
        &mut self,
        input_len: usize,
        accum: impl Accumulator<usize, Output> + Sync,
        reduce: impl Accumulator<Output, Output>,
        cleanup: &(impl SourceCleanup + Sync),
    ) -> Output {
        // The safety guarantees derive from this call as well as how the
        // `IterPipelineImpl` uses a `SkipIteratorWrapper` on each worker thread's
        // range.
        self.range_orchestrator.reset_ranges(input_len);

        let num_threads = self.threads.len();
        let outputs = (0..num_threads)
            .map(|_| Mutex::new(None))
            .collect::<Arc<[_]>>();

        self.pipeline.lend(&IterPipelineImpl {
            outputs: outputs.clone(),
            accum,
            cleanup,
        });

        reduce.accumulate(
            outputs
                .iter()
                .map(move |output| output.lock().unwrap().take().unwrap()),
        )
    }
}

impl<F: RangeFactory> Drop for ThreadPoolImpl<F> {
    /// Joins all the threads in the pool.
    #[allow(clippy::single_match, clippy::unused_enumerate_index)]
    fn drop(&mut self) {
        self.pipeline.finish_workers();

        log_debug!("[main thread] Joining threads in the pool...");
        for (_i, t) in self.threads.drain(..).enumerate() {
            let result = t.handle.join();
            match result {
                Ok(_) => log_debug!("[main thread] Thread {_i} joined with result: {result:?}"),
                Err(_) => log_error!("[main thread] Thread {_i} joined with result: {result:?}"),
            }
        }
        log_debug!("[main thread] Joined threads.");

        #[cfg(feature = "log_parallelism")]
        self.range_orchestrator.print_statistics();
    }
}

trait Pipeline<R: Range> {
    fn run(&self, worker_id: usize, range: &R);
}

/// An intermediate struct representing a `dyn Pipeline<R> + Sync` with variable
/// lifetime. Because Rust doesn't directly support higher-kinded types, we use
/// the generic associated type of the [`LifetimeParameterized`] trait as a
/// proxy.
struct DynLifetimeSyncPipeline<R: Range>(PhantomData<R>);

impl<R: Range> LifetimeParameterized for DynLifetimeSyncPipeline<R> {
    type T<'a> = dyn Pipeline<R> + Sync + 'a;
}

struct UpperBoundedPipelineImpl<
    'a,
    Output,
    Accum,
    Init: Fn() -> Accum,
    ProcessItem: Fn(Accum, usize) -> ControlFlow<Accum, Accum>,
    Finalize: Fn(Accum) -> Output,
    Cleanup: SourceCleanup,
> {
    bound: CachePadded<AtomicUsize>,
    outputs: Arc<[Mutex<Option<Output>>]>,
    init: Init,
    process_item: ProcessItem,
    finalize: Finalize,
    cleanup: &'a Cleanup,
}

impl<R, Output, Accum, Init, ProcessItem, Finalize, Cleanup> Pipeline<R>
    for UpperBoundedPipelineImpl<'_, Output, Accum, Init, ProcessItem, Finalize, Cleanup>
where
    R: Range,
    Init: Fn() -> Accum,
    ProcessItem: Fn(Accum, usize) -> ControlFlow<Accum, Accum>,
    Finalize: Fn(Accum) -> Output,
    Cleanup: SourceCleanup,
{
    fn run(&self, worker_id: usize, range: &R) {
        let mut accumulator = (self.init)();
        let iter = SkipIteratorWrapper {
            iter: range.upper_bounded_iter(&self.bound),
            cleanup: self.cleanup,
        };
        for i in iter {
            let acc = (self.process_item)(accumulator, i);
            accumulator = match acc {
                ControlFlow::Continue(acc) => acc,
                ControlFlow::Break(acc) => {
                    self.bound.fetch_min(i, Ordering::Relaxed);
                    acc
                }
            };
        }
        let output = (self.finalize)(accumulator);
        *self.outputs[worker_id].lock().unwrap() = Some(output);
    }
}

struct IterPipelineImpl<'a, Output, Accum: Accumulator<usize, Output>, Cleanup: SourceCleanup> {
    outputs: Arc<[Mutex<Option<Output>>]>,
    accum: Accum,
    cleanup: &'a Cleanup,
}

impl<R, Output, Accum, Cleanup> Pipeline<R> for IterPipelineImpl<'_, Output, Accum, Cleanup>
where
    R: Range,
    Accum: Accumulator<usize, Output>,
    Cleanup: SourceCleanup,
{
    fn run(&self, worker_id: usize, range: &R) {
        let iter = SkipIteratorWrapper {
            iter: range.iter(),
            cleanup: self.cleanup,
        };
        let output = self.accum.accumulate(iter);
        *self.outputs[worker_id].lock().unwrap() = Some(output);
    }
}

struct SkipIteratorWrapper<'a, I: SkipIterator, Cleanup: SourceCleanup> {
    iter: I,
    cleanup: &'a Cleanup,
}

impl<I: SkipIterator, Cleanup: SourceCleanup> Iterator for SkipIteratorWrapper<'_, I, Cleanup> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter.next() {
                (index, None) => return index,
                (index, Some(skipped_range)) => {
                    // SAFETY: Due to the safety guarantees of `RangeFactory`:
                    // - `skipped_range` is included in the range `0..input_len` (where `input_len`
                    //   is the parameter to the `ThreadPool::*_pipeline()` call),
                    // - `skipped_range` doesn't overlap with any other range passed to
                    //   `cleanup_item_range()` (here or in the `Drop` implementation) nor index
                    //   passed to `fetch_item()` (any index returned by this `next()` function).
                    unsafe {
                        self.cleanup.cleanup_item_range(skipped_range);
                    }
                    if index.is_some() {
                        return index;
                    }
                }
            }
        }
    }
}

impl<I: SkipIterator, Cleanup: SourceCleanup> Drop for SkipIteratorWrapper<'_, I, Cleanup> {
    fn drop(&mut self) {
        if let Some(range) = self.iter.remaining_range() {
            // SAFETY: Due to the safety guarantees of `RangeFactory`:
            // - `range` is included in the range `0..input_len` (where `input_len` is the
            //   parameter to the `ThreadPool::*_pipeline()` call),
            // - `range` doesn't overlap with any other range passed to
            //   `cleanup_item_range()` (here or in the `next()` implementation) nor index
            //   passed to `fetch_item()` (any index returned by the `next()` function).
            unsafe {
                self.cleanup.cleanup_item_range(range);
            }
        }
    }
}

/// Context object owned by a worker thread.
struct ThreadContext<R: Range> {
    /// Thread index.
    id: usize,
    /// Range of items that this worker thread needs to process.
    range: R,
    /// Pipeline to map and reduce inputs into the output.
    pipeline: Borrower<DynLifetimeSyncPipeline<R>>,
}

impl<R: Range> ThreadContext<R> {
    /// Main function run by this thread.
    fn run(&mut self) {
        loop {
            match self.pipeline.borrow(|pipeline| {
                pipeline.run(self.id, &self.range);
            }) {
                WorkerState::Finished => break,
                WorkerState::Ready => continue,
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};

    #[test]
    fn test_thread_count_try_from_usize() {
        assert!(ThreadCount::try_from(0).is_err());
        assert_eq!(
            ThreadCount::try_from(1),
            Ok(ThreadCount::Count(NonZeroUsize::try_from(1).unwrap()))
        );
    }

    #[test]
    fn test_build_thread_pool_available_parallelism() {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<i32>();

        assert_eq!(sum, 5 * 11);
    }

    #[test]
    fn test_build_thread_pool_fixed_thread_count() {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::try_from(4).unwrap(),
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<i32>();

        assert_eq!(sum, 5 * 11);
    }

    #[test]
    fn test_build_thread_pool_cpu_pinning_if_supported() {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        }
        .build();

        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<i32>();

        assert_eq!(sum, 5 * 11);
    }

    #[cfg(all(
        not(miri),
        any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "linux"
        )
    ))]
    #[test]
    fn test_build_thread_pool_cpu_pinning_always() {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::Always,
        }
        .build();

        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<i32>();

        assert_eq!(sum, 5 * 11);
    }

    #[cfg(any(
        miri,
        not(any(
            target_os = "android",
            target_os = "dragonfly",
            target_os = "freebsd",
            target_os = "linux"
        ))
    ))]
    #[test]
    #[should_panic = "Pinning threads to CPUs is not implemented on this platform."]
    fn test_build_thread_pool_cpu_pinning_always_not_supported() {
        ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::Always,
        }
        .build();
    }

    #[test]
    fn test_num_threads() {
        for range_strategy in [RangeStrategy::Fixed, RangeStrategy::WorkStealing] {
            let thread_pool = ThreadPoolBuilder {
                num_threads: ThreadCount::AvailableParallelism,
                range_strategy,
                cpu_pinning: CpuPinningPolicy::No,
            }
            .build();
            assert_eq!(
                thread_pool.num_threads(),
                std::thread::available_parallelism().unwrap()
            );

            let thread_pool = ThreadPoolBuilder {
                num_threads: ThreadCount::try_from(4).unwrap(),
                range_strategy,
                cpu_pinning: CpuPinningPolicy::No,
            }
            .build();
            assert_eq!(
                thread_pool.num_threads(),
                NonZeroUsize::try_from(4).unwrap()
            );
        }
    }
}
