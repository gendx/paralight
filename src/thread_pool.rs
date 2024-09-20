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
use crate::macros::{log_debug, log_error, log_warn};
// Platforms that support `libc::sched_setaffinity()`.
use super::util::{SliceView, Status};
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
use std::cell::Cell;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
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
    /// let sum = pool_builder.scope(|thread_pool| {
    ///     let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    ///     thread_pool
    ///         .process_inputs(&input, || 0u64, |acc, _, x| *acc += *x, |acc| acc)
    ///         .reduce(|a, b| a + b)
    ///         .unwrap()
    /// });
    /// assert_eq!(sum, 5 * 11);
    /// ```
    pub fn scope<Input: Sync, Output: Send, Accum, R>(
        &self,
        f: impl FnOnce(ThreadPool<Input, Output, Accum>) -> R,
    ) -> R {
        std::thread::scope(|scope| {
            let thread_pool = ThreadPool::new(scope, self.num_threads, self.range_strategy);
            f(thread_pool)
        })
    }
}

/// Status of the main thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MainStatus {
    /// The main thread is waiting for the worker threads to finish a round.
    Waiting,
    /// The main thread is ready to prepare the next round.
    Ready,
}

/// Status sent to the worker threads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerStatus {
    /// The worker threads need to compute a pipeline round of the given color.
    Round(RoundColor),
    /// There is nothing more to do and the worker threads must exit.
    Finished,
}

/// An 2-element enumeration to distinguish successive rounds. The "colors" are
/// only illustrative.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundColor {
    Blue,
    Red,
}

impl RoundColor {
    /// Flips to the other color.
    fn toggle(&mut self) {
        *self = match self {
            RoundColor::Blue => RoundColor::Red,
            RoundColor::Red => RoundColor::Blue,
        }
    }
}

/// A thread pool tied to a scope, that can process inputs into outputs of the
/// given types.
pub struct ThreadPool<'scope, Input, Output, Accum> {
    /// Handles to all the worker threads in the pool.
    threads: Vec<WorkerThreadHandle<'scope, Output>>,
    /// Number of worker threads active in the current round.
    num_active_threads: Arc<AtomicUsize>,
    /// Number of worker threads that panicked in the current round.
    num_panicking_threads: Arc<AtomicUsize>,
    /// Color of the current round.
    round: Cell<RoundColor>,
    /// Status of the worker threads.
    worker_status: Arc<Status<WorkerStatus>>,
    /// Status of the main thread.
    main_status: Arc<Status<MainStatus>>,
    /// Orchestrator for the work ranges distributed to the threads. This is a
    /// dynamic object to avoid making the range type a parameter of
    /// everything.
    range_orchestrator: Box<dyn RangeOrchestrator>,
    /// Reference to the inputs to process.
    input: Arc<RwLock<SliceView<Input>>>,
    /// Pipeline to map and reduce inputs into the output.
    pipeline: Arc<RwLock<Option<Pipeline<Input, Output, Accum>>>>,
}

/// Handle to a worker thread in the pool.
struct WorkerThreadHandle<'scope, Output> {
    /// Thread handle object.
    handle: ScopedJoinHandle<'scope, ()>,
    /// Storage for this thread's computation output.
    output: Arc<Mutex<Option<Output>>>,
}

/// Strategy to distribute ranges of work items among threads.
#[derive(Clone, Copy)]
pub enum RangeStrategy {
    /// Each thread processes a fixed range of items.
    Fixed,
    /// Threads can steal work from each other.
    WorkStealing,
}

impl<'scope, Input: Sync + 'scope, Output: Send + 'scope, Accum: 'scope>
    ThreadPool<'scope, Input, Output, Accum>
{
    /// Creates a new pool tied to the given scope, spawning the given number of
    /// worker threads.
    fn new<'env>(
        thread_scope: &'scope Scope<'scope, 'env>,
        num_threads: NonZeroUsize,
        range_strategy: RangeStrategy,
    ) -> Self {
        let num_threads: usize = num_threads.into();
        match range_strategy {
            RangeStrategy::Fixed => Self::new_with_factory(
                thread_scope,
                num_threads,
                FixedRangeFactory::new(num_threads),
            ),
            RangeStrategy::WorkStealing => Self::new_with_factory(
                thread_scope,
                num_threads,
                WorkStealingRangeFactory::new(num_threads),
            ),
        }
    }

    fn new_with_factory<'env, RnFactory: RangeFactory>(
        thread_scope: &'scope Scope<'scope, 'env>,
        num_threads: usize,
        range_factory: RnFactory,
    ) -> Self
    where
        RnFactory::Rn: 'scope + Send,
        RnFactory::Orchestrator: 'static,
    {
        let color = RoundColor::Blue;
        let num_active_threads = Arc::new(AtomicUsize::new(0));
        let num_panicking_threads = Arc::new(AtomicUsize::new(0));
        let worker_status = Arc::new(Status::new(WorkerStatus::Round(color)));
        let main_status = Arc::new(Status::new(MainStatus::Waiting));

        let input = Arc::new(RwLock::new(SliceView::new()));
        let pipeline = Arc::new(RwLock::new(None));

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
        let threads = (0..num_threads)
            .map(|id| {
                let output = Arc::new(Mutex::new(None));
                let context = ThreadContext {
                    #[cfg(feature = "log")]
                    id,
                    num_active_threads: num_active_threads.clone(),
                    num_panicking_threads: num_panicking_threads.clone(),
                    worker_status: worker_status.clone(),
                    main_status: main_status.clone(),
                    range: range_factory.range(id),
                    input: input.clone(),
                    output: output.clone(),
                    pipeline: pipeline.clone(),
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
                    output,
                }
            })
            .collect();
        log_debug!("[main thread] Spawned threads");

        Self {
            threads,
            num_active_threads,
            num_panicking_threads,
            round: Cell::new(color),
            worker_status,
            main_status,
            range_orchestrator: Box::new(range_factory.orchestrator()),
            input,
            pipeline,
        }
    }

    /// Processes an input slice in parallel and returns an iterator over the
    /// threads' outputs.
    ///
    /// # Parameters
    ///
    /// - `input` slice to process in parallel,
    /// - `init` function to create a new (per-thread) accumulator,
    /// - `process_item` function to accumulate an item from the slice into the
    ///   accumulator,
    /// - `finalize` function to transform an accumulator into an output.
    ///
    /// ```rust
    /// # use paralight::{RangeStrategy, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: NonZeroUsize::try_from(4).unwrap(),
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// # };
    /// # pool_builder.scope(|thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = thread_pool
    ///     .process_inputs(&input, || 0u64, |acc, _, x| *acc += *x, |acc| acc)
    ///     .reduce(|a, b| a + b)
    ///     .unwrap();
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    pub fn process_inputs(
        &self,
        input: &[Input],
        init: impl Fn() -> Accum + Send + Sync + 'static,
        process_item: impl Fn(&mut Accum, usize, &Input) + Send + Sync + 'static,
        finalize: impl Fn(Accum) -> Output + Send + Sync + 'static,
    ) -> impl Iterator<Item = Output> + '_ {
        self.range_orchestrator.reset_ranges(input.len());

        let num_threads = self.threads.len();
        self.num_active_threads.store(num_threads, Ordering::SeqCst);

        let mut round = self.round.get();
        round.toggle();
        self.round.set(round);

        self.input.write().unwrap().set(input);
        *self.pipeline.write().unwrap() = Some(Pipeline {
            init: Box::new(init),
            process_item: Box::new(process_item),
            finalize: Box::new(finalize),
        });
        log_debug!("[main thread, round {round:?}] Ready to compute a parallel pipeline.");

        self.worker_status.notify_all(WorkerStatus::Round(round));

        log_debug!("[main thread, round {round:?}] Waiting for all threads to finish computing this pipeline.");

        let mut guard = self
            .main_status
            .wait_while(|status| *status == MainStatus::Waiting);
        assert_eq!(*guard, MainStatus::Ready);

        let num_panicking_threads = self.num_panicking_threads.load(Ordering::SeqCst);
        if num_panicking_threads != 0 {
            log_error!(
                "[main thread, round {round:?}] {num_panicking_threads} worker thread(s) panicked!"
            );
            panic!("{num_panicking_threads} worker thread(s) panicked!");
        }

        *guard = MainStatus::Waiting;
        drop(guard);

        log_debug!(
            "[main thread, round {round:?}] All threads have now finished computing this pipeline."
        );
        *self.pipeline.write().unwrap() = None;
        self.input.write().unwrap().clear();

        self.threads
            .iter()
            .map(move |t| t.output.lock().unwrap().take().unwrap())
    }
}

impl<Input, Output, Accum> Drop for ThreadPool<'_, Input, Output, Accum> {
    /// Joins all the threads in the pool.
    #[allow(clippy::single_match, clippy::unused_enumerate_index)]
    fn drop(&mut self) {
        log_debug!("[main thread] Notifying threads to finish...");
        self.worker_status.notify_all(WorkerStatus::Finished);

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

#[allow(clippy::type_complexity)]
struct Pipeline<Input, Output, Accum> {
    init: Box<dyn Fn() -> Accum + Send + Sync>,
    process_item: Box<dyn Fn(&mut Accum, usize, &Input) + Send + Sync>,
    finalize: Box<dyn Fn(Accum) -> Output + Send + Sync>,
}

/// Context object owned by a worker thread.
struct ThreadContext<Rn: Range, Input, Output, Accum> {
    /// Thread index.
    #[cfg(feature = "log")]
    id: usize,
    /// Number of worker threads active in the current round.
    num_active_threads: Arc<AtomicUsize>,
    /// Number of worker threads that panicked in the current round.
    num_panicking_threads: Arc<AtomicUsize>,
    /// Status of the worker threads.
    worker_status: Arc<Status<WorkerStatus>>,
    /// Status of the main thread.
    main_status: Arc<Status<MainStatus>>,
    /// Range of items that this worker thread needs to process.
    range: Rn,
    /// Reference to the inputs to process.
    input: Arc<RwLock<SliceView<Input>>>,
    /// Output that this thread writes to.
    output: Arc<Mutex<Option<Output>>>,
    /// Pipeline to map and reduce inputs into the output.
    pipeline: Arc<RwLock<Option<Pipeline<Input, Output, Accum>>>>,
}

impl<Rn: Range, Input, Output, Accum> ThreadContext<Rn, Input, Output, Accum> {
    /// Main function run by this thread.
    fn run(&self) {
        let mut round = RoundColor::Blue;
        loop {
            round.toggle();
            log_debug!(
                "[thread {}, round {round:?}] Waiting for start signal",
                self.id
            );

            let worker_status: WorkerStatus =
                *self.worker_status.wait_while(|status| match status {
                    WorkerStatus::Finished => false,
                    WorkerStatus::Round(r) => *r != round,
                });
            match worker_status {
                WorkerStatus::Finished => {
                    log_debug!(
                        "[thread {}, round {round:?}] Received finish signal",
                        self.id
                    );
                    break;
                }
                WorkerStatus::Round(r) => {
                    assert_eq!(round, r);
                    log_debug!(
                        "[thread {}, round {round:?}] Received start signal. Processing...",
                        self.id
                    );

                    // Regardless of the pipeline computation status (success or panic), we want to
                    // notify the main thread that this thread has finished working with the
                    // pipeline object. This happens when the notifier is dropped (whether at the
                    // end of this scope or when a panic is unwound).
                    let notifier = Notifier {
                        #[cfg(feature = "log")]
                        id: self.id,
                        #[cfg(feature = "log")]
                        round,
                        num_active_threads: &self.num_active_threads,
                        num_panicking_threads: &self.num_panicking_threads,
                        main_status: &self.main_status,
                    };

                    {
                        let pipeline_guard = self.pipeline.read().unwrap();
                        let pipeline = pipeline_guard.as_ref().unwrap();
                        let mut accumulator = (pipeline.init)();

                        let guard = self.input.read().unwrap();
                        // SAFETY: the underlying input slice is valid and not mutated for the whole
                        // lifetime of this block.
                        let input = unsafe { guard.get().unwrap() };
                        for i in self.range.iter() {
                            (pipeline.process_item)(&mut accumulator, i, &input[i]);
                        }
                        drop(guard);
                        *self.output.lock().unwrap() = Some((pipeline.finalize)(accumulator));
                    }

                    // Explicit drop for clarity.
                    drop(notifier);
                }
            }
        }
    }
}

/// Object whose destructor notifies the main thread that a worker thread has
/// finished its computing round (or has panicked).
struct Notifier<'a> {
    /// Thread index.
    #[cfg(feature = "log")]
    id: usize,
    /// Color of the current round.
    #[cfg(feature = "log")]
    round: RoundColor,
    /// Number of worker threads active in the current round.
    num_active_threads: &'a AtomicUsize,
    /// Number of worker threads that panicked in the current round.
    num_panicking_threads: &'a AtomicUsize,
    /// Status of the main thread.
    main_status: &'a Status<MainStatus>,
}

impl Drop for Notifier<'_> {
    fn drop(&mut self) {
        #[cfg(feature = "log")]
        let round = self.round;

        // Computing a pipeline may panic, and we want to notify the main thread in that
        // case to avoid using garbage output.
        if std::thread::panicking() {
            log_error!(
                "[thread {}] Detected panic in this thread, notifying the main thread",
                self.id
            );
            self.num_panicking_threads.fetch_add(1, Ordering::SeqCst);
        }

        let thread_count = self.num_active_threads.fetch_sub(1, Ordering::SeqCst);
        assert!(thread_count > 0);
        log_debug!(
            "[thread {}, round {round:?}] Decremented the number of active threads: {}.",
            self.id,
            thread_count - 1
        );

        if thread_count == 1 {
            // We're the last thread.
            log_debug!(
                "[thread {}, round {round:?}] We're the last thread. Waking up the main thread.",
                self.id
            );

            match self.main_status.try_notify_one(MainStatus::Ready) {
                Ok(_) => log_debug!(
                    "[thread {}, round {round:?}] Notified the main thread.",
                    self.id
                ),
                Err(_e) => {
                    log_error!(
                        "[thread {}] Failed to notify the main thread, the mutex was poisoned: {_e:?}",
                        self.id
                    );
                    panic!("Failed to notify the main thread, the mutex was poisoned: {_e:?}");
                }
            }
        } else {
            log_debug!(
                "[thread {}, round {round:?}] Waiting for other threads to finish.",
                self.id
            );
        }
    }
}
