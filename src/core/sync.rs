// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Synchronization primitives

use super::util::{DynLifetimeView, LifetimeParameterized, Status};
use crate::macros::{log_debug, log_error};
use crossbeam_utils::CachePadded;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// State of a worker thread.
#[derive(Clone, Copy)]
pub enum WorkerState {
    /// The thread is ready to compute a new pipeline once notified by the main
    /// thread.
    Ready,
    /// The thread has been notified by the main thread to exit.
    Finished,
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

/// A 2-element enumeration to distinguish successive rounds. The "colors" are
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

/// Create a [`Lender`] paired with `num_threads` [`Borrower`]s.
pub fn make_lending_group<T: LifetimeParameterized>(
    num_threads: usize,
) -> (Lender<T>, Vec<Borrower<T>>) {
    let round = RoundColor::Blue;
    let shared_context = Arc::new(SharedContext {
        num_active_threads: CachePadded::new(AtomicUsize::new(0)),
        num_panicking_threads: CachePadded::new(AtomicUsize::new(0)),
        worker_status: Status::new(WorkerStatus::Round(round)),
        main_status: Status::new(MainStatus::Waiting),
        value: RwLock::new(DynLifetimeView::empty()),
    });

    let borrowers = (0..num_threads)
        .map(|_id| Borrower {
            #[cfg(feature = "log")]
            id: _id,
            round,
            shared_context: shared_context.clone(),
        })
        .collect();

    let lender = Lender {
        num_threads,
        round,
        shared_context,
    };

    (lender, borrowers)
}

/// Context shared between the main thread and the worker threads.
struct SharedContext<T: LifetimeParameterized> {
    /// Number of worker threads active in the current round.
    num_active_threads: CachePadded<AtomicUsize>,
    /// Number of worker threads that panicked in the current round.
    num_panicking_threads: CachePadded<AtomicUsize>,
    /// Status of the worker threads.
    worker_status: Status<WorkerStatus>,
    /// Status of the main thread.
    main_status: Status<MainStatus>,
    /// Value shared with the worker threads.
    value: RwLock<DynLifetimeView<T>>,
}

/// Context for the main thread to lend values of type `T` to the worker
/// threads.
pub struct Lender<T: LifetimeParameterized> {
    /// Number of worker threads in the pool.
    num_threads: usize,
    /// Color of the current round.
    round: RoundColor,
    /// Context shared between the main thread and the worker threads.
    shared_context: Arc<SharedContext<T>>,
}

impl<T: LifetimeParameterized> Lender<T> {
    /// Lend the given value to the worker threads, waiting for the worker
    /// threads to be done borrowing it.
    pub fn lend(&mut self, value: &T::T<'_>) {
        self.shared_context
            .num_active_threads
            .store(self.num_threads, Ordering::SeqCst);

        self.round.toggle();
        let round = self.round;

        // Safety note: The reference set here is valid until the call to `clear()` at
        // the end of this function, which is after all the worker threads are done
        // reading it (as synchronized with `main_status`).
        self.shared_context.value.write().unwrap().set(value);
        log_debug!("[main thread, round {round:?}] Ready to compute a parallel pipeline.");

        self.shared_context
            .worker_status
            .notify_all(WorkerStatus::Round(round));

        log_debug!("[main thread, round {round:?}] Waiting for all threads to finish computing this pipeline.");

        let mut guard = self
            .shared_context
            .main_status
            .wait_while(|status| *status == MainStatus::Waiting);
        assert_eq!(*guard, MainStatus::Ready);

        let num_panicking_threads = self
            .shared_context
            .num_panicking_threads
            .load(Ordering::SeqCst);
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
        // Safety note: the reference (previously set at the beginning of this function)
        // is cleared here after all the worker threads are done reading it (as
        // synchronized with `main_status`).
        self.shared_context.value.write().unwrap().clear();
    }

    /// Notify the worker threads to exit.
    pub fn finish_workers(&mut self) {
        log_debug!("[main thread] Notifying threads to finish...");
        self.shared_context
            .worker_status
            .notify_all(WorkerStatus::Finished);
    }
}

/// Context for a worker thread to borrow values of type `T` from the main
/// thread.
pub struct Borrower<T: LifetimeParameterized> {
    /// Thread index.
    #[cfg(feature = "log")]
    id: usize,
    /// Color of the current round.
    round: RoundColor,
    /// Context shared between the main thread and the worker threads.
    shared_context: Arc<SharedContext<T>>,
}

impl<T: LifetimeParameterized> Borrower<T> {
    /// Wait for the main thread to lend a value, and runs the given function
    /// `f` on that value.
    ///
    /// - If the main thread lends a value via [`Lender::lend()`], this returns
    ///   [`WorkerState::Ready`] after running `f`.
    /// - If the main thread calls [`Lender::finish_workers()`], this returns
    ///   [`WorkerState::Finished`] and doesn't run `f`.
    pub fn borrow(&mut self, f: impl FnOnce(&T::T<'_>)) -> WorkerState {
        self.round.toggle();
        let round = self.round;

        log_debug!(
            "[thread {}, round {round:?}] Waiting for start signal",
            self.id
        );

        let worker_status: WorkerStatus =
            *self
                .shared_context
                .worker_status
                .wait_while(|status| match status {
                    WorkerStatus::Finished => false,
                    WorkerStatus::Round(r) => *r != round,
                });
        match worker_status {
            WorkerStatus::Finished => {
                log_debug!(
                    "[thread {}, round {round:?}] Received finish signal",
                    self.id
                );
                WorkerState::Finished
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
                    shared_context: &self.shared_context,
                };

                {
                    let guard = self.shared_context.value.read().unwrap();
                    // SAFETY:
                    // - The output lifetime doesn't outlive the underlying `T`, as the main thread
                    //   waits until the [`Notifier`]s from all worker threads are dropped before
                    //   exiting the [`lend()`] function.
                    // - The underlying `T` isn't mutated during this scope: all the threads only
                    //   manipulate immutable references to it.
                    let value = unsafe { guard.get().unwrap() };
                    f(value);
                }

                // Explicit drop for clarity.
                drop(notifier);

                WorkerState::Ready
            }
        }
    }
}

/// Object whose destructor notifies the main thread that a worker thread has
/// finished its computing round (or has panicked).
struct Notifier<'a, T: LifetimeParameterized> {
    /// Thread index.
    #[cfg(feature = "log")]
    id: usize,
    /// Color of the current round.
    #[cfg(feature = "log")]
    round: RoundColor,
    /// Context shared between the main thread and the worker threads.
    shared_context: &'a SharedContext<T>,
}

impl<T: LifetimeParameterized> Drop for Notifier<'_, T> {
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
            self.shared_context
                .num_panicking_threads
                .fetch_add(1, Ordering::SeqCst);
        }

        let thread_count = self
            .shared_context
            .num_active_threads
            .fetch_sub(1, Ordering::SeqCst);
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

            match self
                .shared_context
                .main_status
                .try_notify_one(MainStatus::Ready)
            {
                Ok(_) => log_debug!(
                    "[thread {}, round {round:?}] Notified the main thread.",
                    self.id
                ),
                Err(e) => {
                    log_error!(
                        "[thread {}] Failed to notify the main thread, the mutex was poisoned: {e:?}",
                        self.id
                    );
                    panic!("Failed to notify the main thread, the mutex was poisoned: {e:?}");
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
