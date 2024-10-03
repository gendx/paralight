// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A thread pool implementing parallelism at a lightweight cost.

use super::range::{
    FixedRangeFactory, Range, RangeFactory, RangeOrchestrator, WorkStealingRangeFactory,
};
use super::sync::{make_lending_group, Borrower, Lender, WorkerState};
use super::util::LifetimeParameterized;
use crate::macros::{log_debug, log_error, log_warn};
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
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::thread::{Scope, ScopedJoinHandle};

/// A builder for [`ThreadPool`].
pub struct ThreadPoolBuilder {
    /// Number of worker threads to spawn in the pool.
    pub num_threads: NonZeroUsize,
    /// Strategy to distribute ranges of work items among threads.
    pub range_strategy: RangeStrategy,
}

impl ThreadPoolBuilder {
    /// Spawn a scoped thread pool.
    ///
    /// ```rust
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// let pool_builder = ThreadPoolBuilder {
    ///     num_threads: NonZeroUsize::try_from(4).unwrap(),
    ///     range_strategy: RangeStrategy::WorkStealing,
    /// };
    ///
    /// let sum = pool_builder.scope(|mut thread_pool| {
    ///     let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    ///     thread_pool.pipeline(
    ///         &input,
    ///         || 0u64,
    ///         |acc, _, x| *acc += *x,
    ///         |acc| acc,
    ///         |a, b| a + b,
    ///     )
    /// });
    /// assert_eq!(sum, 5 * 11);
    /// ```
    pub fn scope<R>(&self, f: impl FnOnce(ThreadPool) -> R) -> R {
        std::thread::scope(|scope| {
            let thread_pool = ThreadPool::new(scope, self.num_threads, self.range_strategy);
            f(thread_pool)
        })
    }
}

/// A thread pool tied to a scope, that can process inputs into outputs of the
/// given types.
///
/// See [`std::thread::scope()`] for what scoped threads mean and what the
/// `'scope` lifetime refers to.
pub struct ThreadPool<'scope> {
    inner: ThreadPoolEnum<'scope>,
}

impl<'scope> ThreadPool<'scope> {
    /// Creates a new pool tied to the given scope, spawning the given number of
    /// worker threads.
    fn new(
        thread_scope: &'scope Scope<'scope, '_>,
        num_threads: NonZeroUsize,
        range_strategy: RangeStrategy,
    ) -> Self {
        Self {
            inner: ThreadPoolEnum::new(thread_scope, num_threads, range_strategy),
        }
    }

    /// Processes an input slice in parallel and returns the aggregated output.
    ///
    /// # Parameters
    ///
    /// - `input` slice to process in parallel,
    /// - `init` function to create a new (per-thread) accumulator,
    /// - `process_item` function to accumulate an item from the slice into the
    ///   accumulator,
    /// - `finalize` function to transform an accumulator into an output,
    /// - `reduce` function to reduce a pair of outputs into one output.
    ///
    /// ```rust
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = thread_pool.pipeline(
    ///     &input,
    ///     || 0u64,
    ///     |acc, _, x| *acc += *x,
    ///     |acc| acc,
    ///     |a, b| a + b,
    /// );
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    pub fn pipeline<Input: Sync, Output: Send, Accum>(
        &mut self,
        input: &[Input],
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(&mut Accum, usize, &Input) + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.inner
            .pipeline(input, init, process_item, finalize, reduce)
    }
}

enum ThreadPoolEnum<'scope> {
    Fixed(ThreadPoolImpl<'scope, FixedRangeFactory>),
    WorkStealing(ThreadPoolImpl<'scope, WorkStealingRangeFactory>),
}

impl<'scope> ThreadPoolEnum<'scope> {
    /// Creates a new pool tied to the given scope, spawning the given number of
    /// worker threads.
    fn new(
        thread_scope: &'scope Scope<'scope, '_>,
        num_threads: NonZeroUsize,
        range_strategy: RangeStrategy,
    ) -> Self {
        let num_threads: usize = num_threads.into();
        match range_strategy {
            RangeStrategy::Fixed => ThreadPoolEnum::Fixed(ThreadPoolImpl::new(
                thread_scope,
                num_threads,
                FixedRangeFactory::new(num_threads),
            )),
            RangeStrategy::WorkStealing => ThreadPoolEnum::WorkStealing(ThreadPoolImpl::new(
                thread_scope,
                num_threads,
                WorkStealingRangeFactory::new(num_threads),
            )),
        }
    }

    /// Processes an input slice in parallel and returns the aggregated output.
    fn pipeline<Input: Sync, Output: Send, Accum>(
        &mut self,
        input: &[Input],
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(&mut Accum, usize, &Input) + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        match self {
            ThreadPoolEnum::Fixed(inner) => {
                inner.pipeline(input, init, process_item, finalize, reduce)
            }
            ThreadPoolEnum::WorkStealing(inner) => {
                inner.pipeline(input, init, process_item, finalize, reduce)
            }
        }
    }
}

struct ThreadPoolImpl<'scope, F: RangeFactory> {
    /// Handles to all the worker threads in the pool.
    threads: Vec<WorkerThreadHandle<'scope>>,
    /// Orchestrator for the work ranges distributed to the threads.
    range_orchestrator: F::Orchestrator,
    /// Pipeline to map and reduce inputs into the output.
    pipeline: Lender<DynLifetimeSyncPipeline<F::Range>>,
}

/// Handle to a worker thread in the pool.
struct WorkerThreadHandle<'scope> {
    /// Thread handle object.
    handle: ScopedJoinHandle<'scope, ()>,
}

/// Strategy to distribute ranges of work items among threads.
#[derive(Clone, Copy)]
pub enum RangeStrategy {
    /// Each thread processes a fixed range of items.
    Fixed,
    /// Threads can steal work from each other.
    WorkStealing,
}

impl<'scope, F: RangeFactory> ThreadPoolImpl<'scope, F> {
    /// Creates a new pool tied to the given scope, spawning the given number of
    /// worker threads.
    fn new(thread_scope: &'scope Scope<'scope, '_>, num_threads: usize, range_factory: F) -> Self
    where
        F::Range: Send + 'scope,
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
        log_warn!("Pinning threads to CPUs is not implemented on this platform.");
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
                    handle: thread_scope.spawn(move || {
                        #[cfg(all(
                            not(miri),
                            any(
                                target_os = "android",
                                target_os = "dragonfly",
                                target_os = "freebsd",
                                target_os = "linux"
                            )
                        ))]
                        {
                            let mut cpu_set = CpuSet::new();
                            if let Err(_e) = cpu_set.set(id) {
                                log_warn!("Failed to set CPU affinity for thread #{id}: {_e}");
                            } else if let Err(_e) = sched_setaffinity(Pid::from_raw(0), &cpu_set) {
                                log_warn!("Failed to set CPU affinity for thread #{id}: {_e}");
                            } else {
                                log_debug!("Pinned thread #{id} to CPU #{id}");
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

    /// Processes an input slice in parallel and returns the aggregated output.
    fn pipeline<Input: Sync, Output: Send, Accum>(
        &mut self,
        input: &[Input],
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(&mut Accum, usize, &Input) + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.range_orchestrator.reset_ranges(input.len());

        let num_threads = self.threads.len();
        let outputs = (0..num_threads)
            .map(|_| Mutex::new(None))
            .collect::<Arc<[_]>>();

        self.pipeline.lend(&PipelineImpl {
            input,
            outputs: outputs.clone(),
            init,
            process_item,
            finalize,
        });

        outputs
            .iter()
            .map(move |output| output.lock().unwrap().take().unwrap())
            .reduce(reduce)
            .unwrap()
    }
}

impl<F: RangeFactory> Drop for ThreadPoolImpl<'_, F> {
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

struct PipelineImpl<
    'data,
    Input,
    Output,
    Accum,
    Init: Fn() -> Accum,
    ProcessItem: Fn(&mut Accum, usize, &Input),
    Finalize: Fn(Accum) -> Output,
> {
    input: &'data [Input],
    outputs: Arc<[Mutex<Option<Output>>]>,
    init: Init,
    process_item: ProcessItem,
    finalize: Finalize,
}

impl<R, Input, Output, Accum, Init, ProcessItem, Finalize> Pipeline<R>
    for PipelineImpl<'_, Input, Output, Accum, Init, ProcessItem, Finalize>
where
    R: Range,
    Init: Fn() -> Accum,
    ProcessItem: Fn(&mut Accum, usize, &Input),
    Finalize: Fn(Accum) -> Output,
{
    fn run(&self, worker_id: usize, range: &R) {
        // SAFETY: The underlying input slice is valid and not mutated for the whole
        // lifetime of this block.
        let mut accumulator = (self.init)();
        for i in range.iter() {
            (self.process_item)(&mut accumulator, i, &self.input[i]);
        }
        let output = (self.finalize)(accumulator);
        *self.outputs[worker_id].lock().unwrap() = Some(output);
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
