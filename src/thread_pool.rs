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
use super::util::SliceView;
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
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError, RwLock};
use std::thread::{Scope, ScopedJoinHandle};

/// A builder for [`ThreadPool`].
pub struct ThreadPoolBuilder {
    /// Number of worker threads to spawn in the pool.
    pub num_threads: NonZeroUsize,
    /// Strategy to distribute ranges of work items among threads.
    pub range_strategy: RangeStrategy,
}

impl ThreadPoolBuilder {
    /// Spawn a scoped thread pool using the given input and accumulator.
    ///
    /// ```rust
    /// # use paralight::{RangeStrategy, ThreadAccumulator, ThreadPool, ThreadPoolBuilder};
    /// # use std::num::NonZeroUsize;
    /// let pool_builder = ThreadPoolBuilder {
    ///     num_threads: NonZeroUsize::try_from(4).unwrap(),
    ///     range_strategy: RangeStrategy::WorkStealing,
    /// };
    ///
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = pool_builder.scope(
    ///     || SumAccumulator,
    ///     |thread_pool| {
    ///         thread_pool
    ///             .process_inputs(&input)
    ///             .reduce(|a, b| a + b)
    ///             .unwrap()
    ///     },
    /// );
    /// assert_eq!(sum, 5 * 11);
    ///
    /// // Example of accumulator that computes a sum of integers.
    /// struct SumAccumulator;
    ///
    /// impl ThreadAccumulator<u64, u64> for SumAccumulator {
    ///     type Accumulator<'a> = u64;
    ///
    ///     fn init(&self) -> u64 {
    ///         0
    ///     }
    ///
    ///     fn process_item(&self, accumulator: &mut u64, _index: usize, x: &u64) {
    ///         *accumulator += *x;
    ///     }
    ///
    ///     fn finalize(&self, accumulator: u64) -> u64 {
    ///         accumulator
    ///     }
    /// }
    /// ```
    pub fn scope<Input: Sync, Output: Send, Accum: ThreadAccumulator<Input, Output> + Send, R>(
        &self,
        new_accumulator: impl Fn() -> Accum,
        f: impl FnOnce(ThreadPool<Input, Output>) -> R,
    ) -> R {
        std::thread::scope(|scope| {
            let thread_pool = ThreadPool::new(
                scope,
                self.num_threads,
                self.range_strategy,
                new_accumulator,
            );
            f(thread_pool)
        })
    }
}

/// Status of the main thread.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MainStatus {
    /// The main thread is waiting for the worker threads to finish a round.
    Waiting,
    /// The main thread is ready to prepare the next round.
    Ready,
    /// One of the worker threads panicked.
    WorkerPanic,
}

/// Status sent to the worker threads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkerStatus {
    /// The threads need to compute a round of the given color.
    Round(RoundColor),
    /// There is nothing more to do and the threads must exit.
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

/// An ergonomic wrapper around a [`Mutex`]-[`Condvar`] pair.
struct Status<T> {
    mutex: Mutex<T>,
    condvar: Condvar,
}

impl<T> Status<T> {
    /// Creates a new status initialized with the given value.
    fn new(t: T) -> Self {
        Self {
            mutex: Mutex::new(t),
            condvar: Condvar::new(),
        }
    }

    /// Attempts to set the status to the given value and notifies one waiting
    /// thread.
    ///
    /// Fails if the [`Mutex`] is poisoned.
    fn try_notify_one(&self, t: T) -> Result<(), PoisonError<MutexGuard<'_, T>>> {
        *self.mutex.lock()? = t;
        self.condvar.notify_one();
        Ok(())
    }

    /// If the predicate is true on this status, sets the status to the given
    /// value and notifies one waiting thread.
    fn notify_one_if(&self, predicate: impl Fn(&T) -> bool, t: T) {
        let mut locked = self.mutex.lock().unwrap();
        if predicate(&*locked) {
            *locked = t;
            self.condvar.notify_one();
        }
    }

    /// Sets the status to the given value and notifies all waiting threads.
    fn notify_all(&self, t: T) {
        *self.mutex.lock().unwrap() = t;
        self.condvar.notify_all();
    }

    /// Waits until the predicate is true on this status.
    ///
    /// This returns a [`MutexGuard`], allowing to further inspect or modify the
    /// status.
    fn wait_while(&self, predicate: impl FnMut(&mut T) -> bool) -> MutexGuard<T> {
        self.condvar
            .wait_while(self.mutex.lock().unwrap(), predicate)
            .unwrap()
    }
}

/// A thread pool tied to a scope, that can process inputs into outputs of the
/// given types.
pub struct ThreadPool<'scope, Input, Output> {
    /// Handles to all the worker threads in the pool.
    threads: Vec<WorkerThreadHandle<'scope, Output>>,
    /// Number of worker threads active in the current round.
    num_active_threads: Arc<AtomicUsize>,
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

impl<'scope, Input: Sync + 'scope, Output: Send + 'scope> ThreadPool<'scope, Input, Output> {
    /// Creates a new pool tied to the given scope, spawning the given number of
    /// threads and using the given input slice.
    fn new<'env, Accum: ThreadAccumulator<Input, Output> + Send + 'scope>(
        thread_scope: &'scope Scope<'scope, 'env>,
        num_threads: NonZeroUsize,
        range_strategy: RangeStrategy,
        new_accumulator: impl Fn() -> Accum,
    ) -> Self {
        let num_threads: usize = num_threads.into();
        match range_strategy {
            RangeStrategy::Fixed => Self::new_with_factory(
                thread_scope,
                num_threads,
                FixedRangeFactory::new(num_threads),
                new_accumulator,
            ),
            RangeStrategy::WorkStealing => Self::new_with_factory(
                thread_scope,
                num_threads,
                WorkStealingRangeFactory::new(num_threads),
                new_accumulator,
            ),
        }
    }

    fn new_with_factory<
        'env,
        RnFactory: RangeFactory,
        Accum: ThreadAccumulator<Input, Output> + Send + 'scope,
    >(
        thread_scope: &'scope Scope<'scope, 'env>,
        num_threads: usize,
        range_factory: RnFactory,
        new_accumulator: impl Fn() -> Accum,
    ) -> Self
    where
        RnFactory::Rn: 'scope + Send,
        RnFactory::Orchestrator: 'static,
    {
        let color = RoundColor::Blue;
        let num_active_threads = Arc::new(AtomicUsize::new(0));
        let worker_status = Arc::new(Status::new(WorkerStatus::Round(color)));
        let main_status = Arc::new(Status::new(MainStatus::Waiting));

        let input = Arc::new(RwLock::new(SliceView::new()));

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
                    worker_status: worker_status.clone(),
                    main_status: main_status.clone(),
                    range: range_factory.range(id),
                    input: input.clone(),
                    output: output.clone(),
                    accumulator: new_accumulator(),
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
            round: Cell::new(color),
            worker_status,
            main_status,
            range_orchestrator: Box::new(range_factory.orchestrator()),
            input,
        }
    }

    /// Performs a computation round, processing the input slice in parallel and
    /// returning an iterator over the threads' outputs.
    pub fn process_inputs(&self, input: &[Input]) -> impl Iterator<Item = Output> + '_ {
        self.range_orchestrator.reset_ranges(input.len());

        let num_threads = self.threads.len();
        self.num_active_threads.store(num_threads, Ordering::SeqCst);

        let mut round = self.round.get();
        round.toggle();
        self.round.set(round);

        self.input.write().unwrap().set(input);
        log_debug!("[main thread, round {round:?}] Ready to compute a round.");

        self.worker_status.notify_all(WorkerStatus::Round(round));

        log_debug!("[main thread, round {round:?}] Waiting for all threads to finish this round.");

        let mut guard = self
            .main_status
            .wait_while(|status| *status == MainStatus::Waiting);
        if *guard == MainStatus::WorkerPanic {
            log_error!("[main thread] A worker thread panicked!");
            panic!("A worker thread panicked!");
        }
        *guard = MainStatus::Waiting;
        drop(guard);

        log_debug!("[main thread, round {round:?}] All threads have now finished this round.");
        self.input.write().unwrap().clear();

        self.threads
            .iter()
            .map(move |t| t.output.lock().unwrap().take().unwrap())
    }
}

impl<Input, Output> Drop for ThreadPool<'_, Input, Output> {
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

/// Trait representing a function to map and reduce inputs into an output.
pub trait ThreadAccumulator<Input, Output> {
    /// Type to accumulate inputs into.
    type Accumulator<'a>
    where
        Self: 'a;

    /// Creates a new accumulator to process inputs.
    fn init(&self) -> Self::Accumulator<'_>;

    /// Accumulates the given input item.
    fn process_item<'a>(
        &'a self,
        accumulator: &mut Self::Accumulator<'a>,
        index: usize,
        item: &Input,
    );

    /// Converts the given accumulator into an output.
    fn finalize<'a>(&'a self, accumulator: Self::Accumulator<'a>) -> Output;
}

/// Context object owned by a worker thread.
struct ThreadContext<Rn: Range, Input, Output, Accum: ThreadAccumulator<Input, Output>> {
    /// Thread index.
    #[cfg(feature = "log")]
    id: usize,
    /// Number of worker threads active in the current round.
    num_active_threads: Arc<AtomicUsize>,
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
    /// Function to map and reduce inputs into the output.
    accumulator: Accum,
}

impl<Rn: Range, Input, Output, Accum: ThreadAccumulator<Input, Output>>
    ThreadContext<Rn, Input, Output, Accum>
{
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

                    // Computing a round may panic, and we want to notify the main thread in that
                    // case to avoid a deadlock.
                    let panic_notifier = PanicNotifier {
                        #[cfg(feature = "log")]
                        id: self.id,
                        main_status: &self.main_status,
                    };
                    {
                        let mut accumulator = self.accumulator.init();
                        let guard = self.input.read().unwrap();
                        // SAFETY: the underlying input slice is valid and not mutated for the whole
                        // lifetime of this block.
                        let input = unsafe { guard.get().unwrap() };
                        for i in self.range.iter() {
                            self.accumulator
                                .process_item(&mut accumulator, i, &input[i]);
                        }
                        drop(guard);
                        *self.output.lock().unwrap() = Some(self.accumulator.finalize(accumulator));
                    }
                    std::mem::forget(panic_notifier);

                    let thread_count = self.num_active_threads.fetch_sub(1, Ordering::SeqCst);
                    assert!(thread_count > 0);
                    log_debug!(
                        "[thread {}, round {round:?}] Decremented the counter: {}.",
                        self.id,
                        thread_count - 1
                    );
                    if thread_count == 1 {
                        // We're the last thread.
                        log_debug!(
                            "[thread {}, round {round:?}] We're the last thread. Notifying the main thread.",
                            self.id
                        );

                        self.main_status.notify_one_if(
                            |&status| status == MainStatus::Waiting,
                            MainStatus::Ready,
                        );

                        log_debug!(
                            "[thread {}, round {round:?}] Notified the main thread.",
                            self.id
                        );
                    } else {
                        log_debug!(
                            "[thread {}, round {round:?}] Waiting for other threads to finish.",
                            self.id
                        );
                    }
                }
            }
        }
    }
}

/// Object whose destructor notifies the main thread that a panic happened.
///
/// The way to use this is to create an instance before a section that may
/// panic, and to [`std::mem::forget()`] it at the end of the section. That way:
/// - If a panic happens, the [`std::mem::forget()`] call will be skipped but
///   the destructor will run due to RAII.
/// - If no panic happens, the destructor won't run because this object will be
///   forgotten.
struct PanicNotifier<'a> {
    /// Thread index.
    #[cfg(feature = "log")]
    id: usize,
    /// Status of the main thread.
    main_status: &'a Status<MainStatus>,
}

impl Drop for PanicNotifier<'_> {
    fn drop(&mut self) {
        log_error!(
            "[thread {}] Detected panic in this thread, notifying the main thread",
            self.id
        );
        if let Err(_e) = self.main_status.try_notify_one(MainStatus::WorkerPanic) {
            log_error!(
                "[thread {}] Failed to notify the main thread, the mutex was poisoned: {_e:?}",
                self.id
            );
        }
    }
}
