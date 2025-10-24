// Copyright 2024-2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::macros::log_debug;
#[cfg(feature = "log_parallelism")]
use crate::macros::{log_info, log_trace};
use crossbeam_utils::CachePadded;
#[cfg(feature = "log_parallelism")]
use std::ops::AddAssign;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(feature = "log_parallelism")]
use std::sync::Mutex;

/// A factory for handing out ranges of items to various threads.
///
/// # Safety
///
/// Implementers of the [`RangeFactory`] must guarantee the following contract.
///
/// Given a number of threads `num_threads` and a factory created by
/// `RangeFactory::new(num_threads)`, from which are derived `num_threads`
/// ranges as `(0..num_threads).iter().map(|i| factory.range(i)).collect()` and
/// an orchestrator as `factory.orchestrator()`:
///
/// - Given a number of elements `num_elements`, calling
///   `orchestrator.reset_ranges(num_elements)` followed by `range.iter()` on
///   all ranges, each index in `0..num_elements` will appear once and only once
///   in the `num_threads` resulting [`SkipIterator`]s (in the union of regular
///   items and skipped ranges obtained via calls to `iter.next()` and
///   `iter.remaining_range()`).
/// - Likewise, calling `orchestrator.reset_ranges(num_elements)` followed by
///   `range.upper_bounded_iter(&bound)` on all ranges will yield each index in
///   `0..num_elements` once and only once.
pub trait RangeFactory {
    type Range: Range;
    type Orchestrator: RangeOrchestrator;

    /// Creates a new factory for a range split across the given number of
    /// threads.
    fn new(num_threads: usize) -> Self;

    /// Returns the orchestrator object for all the ranges created by this
    /// factory.
    fn orchestrator(self) -> Self::Orchestrator;

    /// Returns the range for the given thread.
    fn range(&self, thread_id: usize) -> Self::Range;
}

/// An orchestrator for the ranges given to all the threads.
pub trait RangeOrchestrator {
    /// Resets all the ranges to prepare a new computation round.
    fn reset_ranges(&self, num_elements: usize);

    /// Hook to display various debugging statistics.
    #[cfg(feature = "log_parallelism")]
    fn print_statistics(&self) {}
}

/// A range of items similar to [`std::ops::Range`], but that can steal from or
/// be stolen by other threads.
pub trait Range {
    /// Type of iterator returned by [`iter()`](Self::iter).
    type Iter<'a>: SkipIterator
    where
        Self: 'a;

    /// Type of iterator returned by
    /// [`upper_bounded_iter()`](Self::upper_bounded_iter).
    type UpperBoundedIter<'a, 'bound>: SkipIterator
    where
        Self: 'a;

    /// Returns an iterator over the items in this range. The item can be
    /// dynamically stolen from/by other threads, but the iterator provides
    /// a safe abstraction over that.
    fn iter(&self) -> Self::Iter<'_>;

    /// Returns an iterator over the items in this range. Items larger than the
    /// (dynamic) bound are skipped.
    fn upper_bounded_iter<'a, 'bound>(
        &'a self,
        bound: &'bound AtomicUsize,
    ) -> Self::UpperBoundedIter<'a, 'bound>;
}

/// An iterator trait over `usize` that either returns a next index or a range
/// of skipped indices.
pub trait SkipIterator {
    /// Returns the next item and/or a range of skipped indices.
    ///
    /// The iterator is exhausted if and only if this returns a pair of [`None`]
    /// values.
    fn next(&mut self) -> (Option<usize>, Option<std::ops::Range<usize>>);

    /// Returns any remaining range of indices that have been skipped.
    ///
    /// This iterator must not be used again once this has been called.
    fn remaining_range(&self) -> Option<std::ops::Range<usize>>;
}

/// A factory that hands out a fixed range to each thread, without any stealing.
pub struct FixedRangeFactory {
    /// Number of threads that iterate.
    num_threads: usize,
    /// Total number of elements in the current range.
    num_elements: Arc<AtomicUsize>,
}

// Here is a proof that `FixedRangeFactory` upholds the safety contract of
// `RangeFactory`.
//
// Upon calling `FixedRangeOrchestrator::reset_ranges(num_elements)`, a common
// `AtomicUsize` shared with all the `FixedRange`s is updated to this number of
// elements. This update uses `Ordering::Relaxed`, which is fine because it's
// the caller's responsibility to ensure that the `reset_ranges()` call happens
// before calls to `range.iter()` or `range.upper_bounded_iter(&bound)`.
//
// Then, the range `0..num_elements` is partitioned into `num_threads`
// non-overlapping ranges via the `FixedRange::range()` function.
//
// - In the `range.iter()` case, `SkipIterator::next()` calls simply pop the
//   first element of the local range, and `SkipIterator::remaining_range()`
//   returns the remaining items (if any).
// - In the `range.upper_bounded_iter()` case, `UpperBoundedRange::next()` calls
//   also pop the first element, but may alternatively pop nothing (and leave
//   the iterator's state unchanged) if the upper bound is identified as
//   reached. `UpperBoundedRange::remaining_range()` returns the remaining items
//   that haven't been popped (if any).
//
// In both cases, this ensures full coverage of the `0..num_elements` range, as
// well as uniqueness.
impl RangeFactory for FixedRangeFactory {
    type Range = FixedRange;
    type Orchestrator = FixedRangeOrchestrator;

    fn new(num_threads: usize) -> Self {
        Self {
            num_threads,
            num_elements: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn orchestrator(self) -> FixedRangeOrchestrator {
        FixedRangeOrchestrator {
            num_elements: self.num_elements,
        }
    }

    fn range(&self, thread_id: usize) -> FixedRange {
        FixedRange {
            id: thread_id,
            num_threads: self.num_threads,
            num_elements: self.num_elements.clone(),
        }
    }
}

/// An orchestrator for the [`FixedRangeFactory`].
pub struct FixedRangeOrchestrator {
    /// Total number of elements.
    num_elements: Arc<AtomicUsize>,
}

impl RangeOrchestrator for FixedRangeOrchestrator {
    fn reset_ranges(&self, num_elements: usize) {
        self.num_elements.store(num_elements, Ordering::Relaxed);
    }
}

/// A fixed range.
pub struct FixedRange {
    /// Index of the thread that owns this range.
    id: usize,
    /// Total number of threads.
    num_threads: usize,
    /// Total number of elements.
    num_elements: Arc<AtomicUsize>,
}

impl FixedRange {
    fn range(&self) -> std::ops::Range<usize> {
        let num_elements = self.num_elements.load(Ordering::Relaxed);
        let start = (self.id * num_elements) / self.num_threads;
        let end = ((self.id + 1) * num_elements) / self.num_threads;
        start..end
    }
}

impl Range for FixedRange {
    type Iter<'a> = std::ops::Range<usize>;
    type UpperBoundedIter<'a, 'bound> = UpperBoundedRange<'bound>;

    fn iter(&self) -> Self::Iter<'_> {
        self.range()
    }

    fn upper_bounded_iter<'a, 'bound>(
        &'a self,
        bound: &'bound AtomicUsize,
    ) -> Self::UpperBoundedIter<'a, 'bound> {
        UpperBoundedRange {
            range: self.range(),
            bound,
        }
    }
}

impl SkipIterator for std::ops::Range<usize> {
    #[inline(always)]
    fn next(&mut self) -> (Option<usize>, Option<std::ops::Range<usize>>) {
        (Iterator::next(self), None)
    }

    #[inline(always)]
    fn remaining_range(&self) -> Option<std::ops::Range<usize>> {
        if self.is_empty() {
            None
        } else {
            Some(self.clone())
        }
    }
}

/// An upper-bounded iterator for a [`FixedRange`].
pub struct UpperBoundedRange<'bound> {
    /// Underlying contiguous range.
    range: std::ops::Range<usize>,
    /// Dynamic upper bound.
    bound: &'bound AtomicUsize,
}

impl SkipIterator for UpperBoundedRange<'_> {
    #[inline(always)]
    fn next(&mut self) -> (Option<usize>, Option<std::ops::Range<usize>>) {
        let start = self.range.start;
        if start != self.range.end && start <= self.bound.load(Ordering::Relaxed) {
            self.range.start += 1;
            (Some(start), None)
        } else {
            // The upper bound can only decrease, so once it's reached the iterator is
            // exhausted.
            (None, None)
        }
    }

    #[inline(always)]
    fn remaining_range(&self) -> Option<std::ops::Range<usize>> {
        if self.range.is_empty() {
            None
        } else {
            Some(self.range.clone())
        }
    }
}

/// A factory for ranges that implement work stealing among threads.
///
/// Whenever a thread finishes processing its range, it looks for another range
/// to steal from. It then divides that range into two and steals a half, to
/// continue processing items.
pub struct WorkStealingRangeFactory {
    /// Handle to the ranges of all the threads.
    ranges: Arc<[AtomicRange]>,
    /// Handle to the work-stealing statistics.
    #[cfg(feature = "log_parallelism")]
    stats: Arc<Mutex<WorkStealingStats>>,
}

// TODO: A formal proof that `WorkStealingRangeFactory` upholds the safety
// contract of `RangeFactory` isn't yet written.
impl RangeFactory for WorkStealingRangeFactory {
    type Range = WorkStealingRange;
    type Orchestrator = WorkStealingRangeOrchestrator;

    fn new(num_threads: usize) -> Self {
        if u32::try_from(num_threads).is_err() {
            panic!(
                "cannot spawn {num_threads} threads: only up to {} threads (2^32 - 1) are supported",
                u32::MAX
            );
        }
        Self {
            ranges: (0..num_threads).map(|_| AtomicRange::default()).collect(),
            #[cfg(feature = "log_parallelism")]
            stats: Arc::new(Mutex::new(WorkStealingStats::default())),
        }
    }

    fn orchestrator(self) -> WorkStealingRangeOrchestrator {
        WorkStealingRangeOrchestrator {
            ranges: self.ranges,
            #[cfg(feature = "log_parallelism")]
            stats: self.stats,
        }
    }

    fn range(&self, thread_id: usize) -> WorkStealingRange {
        WorkStealingRange {
            id: thread_id,
            ranges: self.ranges.clone(),
            #[cfg(feature = "log_parallelism")]
            stats: self.stats.clone(),
        }
    }
}

/// An orchestrator for the [`WorkStealingRangeFactory`].
pub struct WorkStealingRangeOrchestrator {
    /// Handle to the ranges of all the threads.
    ranges: Arc<[AtomicRange]>,
    /// Handle to the work-stealing statistics.
    #[cfg(feature = "log_parallelism")]
    stats: Arc<Mutex<WorkStealingStats>>,
}

impl RangeOrchestrator for WorkStealingRangeOrchestrator {
    fn reset_ranges(&self, num_elements: usize) {
        log_debug!("Resetting ranges");
        let num_threads = self.ranges.len() as u64;
        let num_elements = u32::try_from(num_elements).unwrap_or_else(|_| {
            panic!(
                "cannot process range of {num_elements} elements: only ranges of up to {} elements (2^32 - 1) are supported",
                u32::MAX
            );
        }) as u64;
        for (i, range) in self.ranges.iter().enumerate() {
            let i = i as u64;
            // - This multiplication cannot overflow because `i+1` and `num_elements` both
            //   fit in u32.
            // - The result fits in u32 because `i < num_threads` and `num_elements` fits in
            //   u32.
            let start = (i * num_elements) / num_threads;
            let end = ((i + 1) * num_elements) / num_threads;
            range.store(PackedRange::new(start as u32, end as u32));
        }
    }

    #[cfg(feature = "log_parallelism")]
    fn print_statistics(&self) {
        let stats = self.stats.lock().unwrap();
        log_info!("Work-stealing statistics:");
        log_info!("- increments: {}", stats.increments);
        log_info!("- failed_increments: {}", stats.failed_increments);
        log_info!("- other_loads: {}", stats.other_loads);
        log_info!("- thefts: {}", stats.thefts);
        log_info!("- failed_thefts: {}", stats.failed_thefts);
        log_info!("- increments + thefts: {}", stats.increments + stats.thefts);
    }
}

/// A range that implements work stealing.
pub struct WorkStealingRange {
    /// Index of the thread that owns this range.
    id: usize,
    /// Handle to the ranges of all the threads.
    ranges: Arc<[AtomicRange]>,
    /// Handle to the work-stealing statistics.
    #[cfg(feature = "log_parallelism")]
    stats: Arc<Mutex<WorkStealingStats>>,
}

impl Range for WorkStealingRange {
    type Iter<'a> = WorkStealingRangeIterator<'a>;
    type UpperBoundedIter<'a, 'bound> = UpperBoundedWorkStealingRangeIterator<'a, 'bound>;

    fn iter(&self) -> Self::Iter<'_> {
        WorkStealingRangeIterator {
            id: self.id,
            ranges: &self.ranges,
            #[cfg(feature = "log_parallelism")]
            stats: WorkStealingStats::default(),
            #[cfg(feature = "log_parallelism")]
            global_stats: self.stats.clone(),
        }
    }

    fn upper_bounded_iter<'a, 'bound>(
        &'a self,
        bound: &'bound AtomicUsize,
    ) -> Self::UpperBoundedIter<'a, 'bound> {
        UpperBoundedWorkStealingRangeIterator {
            id: self.id,
            ranges: &self.ranges,
            bound,
            #[cfg(feature = "log_parallelism")]
            stats: WorkStealingStats::default(),
            #[cfg(feature = "log_parallelism")]
            global_stats: self.stats.clone(),
        }
    }
}

/// A [start, end) pair that can atomically be modified.
struct AtomicRange(CachePadded<AtomicU64>);

impl Default for AtomicRange {
    #[inline(always)]
    fn default() -> Self {
        AtomicRange::new(PackedRange::default())
    }
}

impl AtomicRange {
    /// Creates a new atomic range.
    #[inline(always)]
    fn new(range: PackedRange) -> Self {
        AtomicRange(CachePadded::new(AtomicU64::new(range.0)))
    }

    /// Atomically loads the range.
    #[inline(always)]
    fn load(&self) -> PackedRange {
        PackedRange(self.0.load(Ordering::SeqCst))
    }

    /// Atomically stores the range.
    #[inline(always)]
    fn store(&self, range: PackedRange) {
        self.0.store(range.0, Ordering::SeqCst)
    }

    /// Atomically compares and exchanges the range. In case of failure, the
    /// range contained in the atomic variable is returned.
    #[inline(always)]
    fn compare_exchange(&self, before: PackedRange, after: PackedRange) -> Result<(), PackedRange> {
        match self
            .0
            .compare_exchange(before.0, after.0, Ordering::SeqCst, Ordering::SeqCst)
        {
            Ok(_) => Ok(()),
            Err(e) => Err(PackedRange(e)),
        }
    }
}

/// A [start, end) range that fits into a `u64`, and can therefore be
/// loaded/stored atomically.
#[derive(Clone, Copy, Default)]
struct PackedRange(u64);

impl PackedRange {
    /// Creates a range with the given [start, end) pair.
    #[inline(always)]
    fn new(start: u32, end: u32) -> Self {
        Self((start as u64) | ((end as u64) << 32))
    }

    /// Reads the start of the range (inclusive).
    #[inline(always)]
    fn start(self) -> u32 {
        self.0 as u32
    }

    /// Reads the end of the range (exclusive).
    #[inline(always)]
    fn end(self) -> u32 {
        (self.0 >> 32) as u32
    }

    #[inline(always)]
    fn to_range(self) -> std::ops::Range<usize> {
        self.start() as usize..self.end() as usize
    }

    /// Reads the length of the range.
    #[inline(always)]
    fn len(self) -> u32 {
        self.end() - self.start()
    }

    /// Checks if the range is empty.
    #[inline(always)]
    fn is_empty(self) -> bool {
        self.start() == self.end()
    }

    /// Upper bound this range by the given maximum.
    #[inline(always)]
    fn upper_bound(self, bound: usize) -> (Self, Self) {
        let start = self.start();
        let end = self.end();

        if end as usize <= bound {
            (Self::new(start, end), Self::default())
        } else if start as usize >= bound {
            (Self::default(), Self::new(start, end))
        } else {
            (Self::new(start, bound as u32), Self::new(bound as u32, end))
        }
    }

    /// Increments the start of the range.
    #[inline(always)]
    fn increment_start(self) -> (u32, Self) {
        debug_assert!(self.start() < self.end());
        (self.start(), PackedRange::new(self.start() + 1, self.end()))
    }

    /// Splits the range into two halves. If the input range is non-empty, the
    /// second half is guaranteed to be non-empty.
    #[inline(always)]
    fn split(self) -> (Self, Self) {
        let start = self.start();
        let end = self.end();
        // TODO(MSRV >= 1.85.0): Use u32::midpoint().
        // The result fits in u32 because the inputs fit in u32.
        let middle = ((start as u64 + end as u64) / 2) as u32;
        (Self::new(start, middle), Self::new(middle, end))
    }
}

#[cfg(feature = "log_parallelism")]
#[derive(Default)]
pub struct WorkStealingStats {
    /// Number of times this thread successfully incremented its range.
    increments: u64,
    /// Number of times this thread failed to increment its range, because
    /// another thread stole it.
    failed_increments: u64,
    /// Number of times this thread loaded the range of another thread
    /// (excluding compare-exchanges).
    other_loads: u64,
    /// Number of times this thread has stolen a range from another thread.
    thefts: u64,
    /// Number of times this thread failed to steal a range because another
    /// thread modified it in the meantime.
    failed_thefts: u64,
}

#[cfg(feature = "log_parallelism")]
impl AddAssign<&WorkStealingStats> for WorkStealingStats {
    fn add_assign(&mut self, other: &WorkStealingStats) {
        self.increments += other.increments;
        self.failed_increments += other.failed_increments;
        self.other_loads += other.other_loads;
        self.thefts += other.thefts;
        self.failed_thefts += other.failed_thefts;
    }
}

/// An iterator for the [`WorkStealingRange`].
pub struct WorkStealingRangeIterator<'a> {
    /// Index of the thread that owns this range.
    id: usize,
    /// Handle to the ranges of all the threads.
    ranges: &'a [AtomicRange],
    /// Local work-stealing statistics.
    #[cfg(feature = "log_parallelism")]
    stats: WorkStealingStats,
    /// Handle to the global work-stealing statistics.
    #[cfg(feature = "log_parallelism")]
    global_stats: Arc<Mutex<WorkStealingStats>>,
}

#[cfg(feature = "log_parallelism")]
impl Drop for WorkStealingRangeIterator<'_> {
    fn drop(&mut self) {
        *self.global_stats.lock().unwrap() += &self.stats;
    }
}

impl SkipIterator for WorkStealingRangeIterator<'_> {
    #[inline(always)]
    fn remaining_range(&self) -> Option<std::ops::Range<usize>> {
        let my_atomic_range: &AtomicRange = &self.ranges[self.id];
        let mut my_range: PackedRange = my_atomic_range.load();

        while !my_range.is_empty() {
            match my_atomic_range.compare_exchange(my_range, PackedRange::default()) {
                Ok(()) => return Some(my_range.to_range()),
                Err(range) => my_range = range,
            }
        }

        None
    }

    #[inline(always)]
    fn next(&mut self) -> (Option<usize>, Option<std::ops::Range<usize>>) {
        let my_atomic_range: &AtomicRange = &self.ranges[self.id];
        let mut my_range: PackedRange = my_atomic_range.load();

        // First phase: try to increment this thread's own range. Retries are needed in
        // case another thread stole part of the range.
        while !my_range.is_empty() {
            let (taken, my_new_range) = my_range.increment_start();
            match my_atomic_range.compare_exchange(my_range, my_new_range) {
                // Increment succeeded.
                Ok(()) => {
                    #[cfg(feature = "log_parallelism")]
                    {
                        self.stats.increments += 1;
                        log_trace!(
                            "[thread {}] Incremented range to {:?}",
                            self.id,
                            my_new_range.to_range()
                        );
                    }
                    return (Some(taken as usize), None);
                }
                // Increment failed: retry with an updated range.
                Err(range) => {
                    my_range = range;
                    #[cfg(feature = "log_parallelism")]
                    {
                        self.stats.failed_increments += 1;
                        log_debug!(
                            "[thread {}] Failed to increment range, new range is {:?}",
                            self.id,
                            range.to_range()
                        );
                    }
                    continue;
                }
            }
        }

        // Second phase: the range is empty, try to steal a range from another thread.
        self.steal(
            #[cfg(feature = "log_parallelism")]
            my_range,
        )
    }
}

impl WorkStealingRangeIterator<'_> {
    /// Helper function for the iterator implementation, to steal a range from
    /// another thread when this thread's range is empty.
    #[cold]
    fn steal(
        &mut self,
        #[cfg(feature = "log_parallelism")] my_range: PackedRange,
    ) -> (Option<usize>, Option<std::ops::Range<usize>>) {
        let my_atomic_range: &AtomicRange = &self.ranges[self.id];

        #[cfg(feature = "log_parallelism")]
        log_debug!(
            "[thread {}] Range {:?} is empty, scanning other threads",
            self.id,
            my_range.to_range()
        );
        let range_count = self.ranges.len();

        // Read a snapshot of the other threads' ranges, to identify the best one to
        // steal (the largest one). This is only used as a hint, and therefore it's fine
        // that the underlying values may be concurrently modified by the other threads
        // and that the snapshot becomes (slightly) out-of-date.
        let mut other_ranges = vec![PackedRange::default(); range_count];
        for (i, range) in other_ranges.iter_mut().enumerate() {
            if i == self.id {
                continue;
            }
            *range = self.ranges[i].load();
        }
        #[cfg(feature = "log_parallelism")]
        {
            self.stats.other_loads += range_count as u64 - 1;
        }

        // Identify the thread with the largest range.
        let mut max_index = 0;
        let mut max_range = PackedRange::default();
        for (i, range) in other_ranges.iter().enumerate() {
            if i == self.id {
                continue;
            }
            if range.len() > max_range.len() {
                max_index = i;
                max_range = *range;
            }
        }

        // Try to steal another thread's range. Retries are needed in case the target
        // thread incremented its range or if another thread stole part of the
        // target thread's range.
        while !max_range.is_empty() {
            // Try to steal half of the range.
            let (remaining, stolen) = max_range.split();
            match self.ranges[max_index].compare_exchange(max_range, remaining) {
                // Theft succeeded.
                Ok(()) => {
                    // Take the first item, and place the rest in this thread's own range.
                    let (taken, my_new_range) = stolen.increment_start();
                    my_atomic_range.store(my_new_range);
                    #[cfg(feature = "log_parallelism")]
                    {
                        self.stats.thefts += 1;
                        log_trace!(
                            "[thread {}] Stole range {}:{:?} from thread {}",
                            self.id,
                            taken,
                            my_new_range.to_range(),
                            max_index
                        );
                    }
                    return (Some(taken as usize), None);
                }
                // Theft failed: update the range and retry.
                Err(range) => {
                    other_ranges[max_index] = range;
                    #[cfg(feature = "log_parallelism")]
                    {
                        self.stats.failed_thefts += 1;
                    }

                    // Re-compute the largest range.
                    max_range = range;
                    for (i, range) in other_ranges.iter().enumerate() {
                        if i == self.id {
                            continue;
                        }
                        if range.len() > max_range.len() {
                            max_index = i;
                            max_range = *range;
                        }
                    }
                }
            }
        }

        // Didn't manage to steal anything: exit the iterator.
        #[cfg(feature = "log_parallelism")]
        log_debug!("[thread {}] Didn't find anything to steal", self.id);
        (None, None)
    }
}

/// A upper-bounded iterator for a [`WorkStealingRange`].
pub struct UpperBoundedWorkStealingRangeIterator<'a, 'bound> {
    /// Index of the thread that owns this range.
    id: usize,
    /// Handle to the ranges of all the threads.
    ranges: &'a [AtomicRange],
    /// Dynamic upper bound.
    bound: &'bound AtomicUsize,
    /// Local work-stealing statistics.
    #[cfg(feature = "log_parallelism")]
    stats: WorkStealingStats,
    /// Handle to the global work-stealing statistics.
    #[cfg(feature = "log_parallelism")]
    global_stats: Arc<Mutex<WorkStealingStats>>,
}

#[cfg(feature = "log_parallelism")]
impl Drop for UpperBoundedWorkStealingRangeIterator<'_, '_> {
    fn drop(&mut self) {
        *self.global_stats.lock().unwrap() += &self.stats;
    }
}

impl SkipIterator for UpperBoundedWorkStealingRangeIterator<'_, '_> {
    #[inline(always)]
    fn remaining_range(&self) -> Option<std::ops::Range<usize>> {
        let my_atomic_range: &AtomicRange = &self.ranges[self.id];
        let mut my_range: PackedRange = my_atomic_range.load();

        while !my_range.is_empty() {
            match my_atomic_range.compare_exchange(my_range, PackedRange::default()) {
                Ok(()) => return Some(my_range.to_range()),
                Err(range) => my_range = range,
            }
        }

        None
    }

    #[inline(always)]
    fn next(&mut self) -> (Option<usize>, Option<std::ops::Range<usize>>) {
        let bound = self.bound.load(Ordering::Relaxed);
        #[cfg(feature = "log_parallelism")]
        log_trace!("[thread {}] Loaded upper bound = {}", self.id, bound);

        let my_atomic_range: &AtomicRange = &self.ranges[self.id];
        let mut my_loaded_range: PackedRange = my_atomic_range.load();
        let (mut my_bounded_range, mut my_residual_range) = my_loaded_range.upper_bound(bound);

        // First phase: try to increment this thread's own range. Retries are needed in
        // case another thread stole part of the range.
        loop {
            if !my_bounded_range.is_empty() {
                let (taken, my_new_range) = my_bounded_range.increment_start();
                match my_atomic_range.compare_exchange(my_loaded_range, my_new_range) {
                    // Increment succeeded.
                    Ok(()) => {
                        #[cfg(feature = "log_parallelism")]
                        {
                            self.stats.increments += 1;
                            log_trace!(
                                "[thread {}] Incremented range to {:?}",
                                self.id,
                                my_new_range.to_range()
                            );
                        }

                        let residual = if my_residual_range.is_empty() {
                            None
                        } else {
                            let residual = my_residual_range.to_range();
                            #[cfg(feature = "log_parallelism")]
                            log_debug!(
                                "[thread {}] Residual range {:?} is not empty (increment), scheduling it for cleanup.",
                                self.id,
                                residual
                            );
                            Some(residual)
                        };

                        return (Some(taken as usize), residual);
                    }
                    // Increment failed: retry with an updated range.
                    Err(range) => {
                        my_loaded_range = range;
                        (my_bounded_range, my_residual_range) = my_loaded_range.upper_bound(bound);
                        #[cfg(feature = "log_parallelism")]
                        {
                            self.stats.failed_increments += 1;
                            log_debug!(
                                "[thread {}] Failed to increment range, new range is {:?}",
                                self.id,
                                range.to_range()
                            );
                        }
                        continue;
                    }
                }
            } else if !my_loaded_range.is_empty() {
                // First, let's make sure other threads don't try to steal this range, which can
                // happen if they have cached another bound.
                match my_atomic_range.compare_exchange(my_loaded_range, my_bounded_range) {
                    Ok(()) => {
                        if !my_residual_range.is_empty() {
                            let residual = my_residual_range.to_range();
                            #[cfg(feature = "log_parallelism")]
                            log_debug!(
                                "[thread {}] Residual range {:?} is not empty (empty bounded range), scheduling it for cleanup.",
                                self.id,
                                residual
                            );
                            return (None, Some(residual));
                        };
                        break;
                    }
                    Err(range) => {
                        my_loaded_range = range;
                        (my_bounded_range, my_residual_range) = my_loaded_range.upper_bound(bound);
                        continue;
                    }
                }
            } else {
                break;
            }
        }

        // Second phase: the range is empty, try to steal a range from another thread.
        self.steal(
            bound,
            #[cfg(feature = "log_parallelism")]
            my_bounded_range,
        )
    }
}

#[derive(Clone, Copy, Default)]
struct OtherRange {
    loaded: PackedRange,
    bounded: PackedRange,
    residual: PackedRange,
}

impl UpperBoundedWorkStealingRangeIterator<'_, '_> {
    /// Helper function for the iterator implementation, to steal a range from
    /// another thread when this thread's range is empty.
    #[cold]
    fn steal(
        &mut self,
        bound: usize,
        #[cfg(feature = "log_parallelism")] my_bounded_range: PackedRange,
    ) -> (Option<usize>, Option<std::ops::Range<usize>>) {
        let my_atomic_range: &AtomicRange = &self.ranges[self.id];

        #[cfg(feature = "log_parallelism")]
        log_debug!(
            "[thread {}] Range {:?} is empty, scanning other threads",
            self.id,
            my_bounded_range.to_range()
        );
        let range_count = self.ranges.len();

        // Read a snapshot of the other threads' ranges, to identify the best one to
        // steal (the largest one). This is only used as a hint, and therefore it's fine
        // that the underlying values may be concurrently modified by the other threads
        // and that the snapshot becomes (slightly) out-of-date.
        let mut other_ranges = vec![OtherRange::default(); range_count];
        for (i, range) in other_ranges.iter_mut().enumerate() {
            if i == self.id {
                continue;
            }
            let loaded = self.ranges[i].load();
            let (bounded, residual) = loaded.upper_bound(bound);
            *range = OtherRange {
                loaded,
                bounded,
                residual,
            };
        }
        #[cfg(feature = "log_parallelism")]
        {
            self.stats.other_loads += range_count as u64 - 1;
        }

        // Identify the thread with the largest range.
        let mut max_index = 0;
        let mut max_range = OtherRange::default();
        for (i, range) in other_ranges.iter().enumerate() {
            if i == self.id {
                continue;
            }
            if range.bounded.len() > max_range.bounded.len() {
                max_index = i;
                max_range = *range;
            }
        }

        // Try to steal another thread's range. Retries are needed in case the target
        // thread incremented its range or if another thread stole part of the
        // target thread's range.
        while !max_range.bounded.is_empty() {
            // Try to steal half of the range.
            let (remaining, stolen) = max_range.bounded.split();
            match self.ranges[max_index].compare_exchange(max_range.loaded, remaining) {
                // Theft succeeded.
                Ok(()) => {
                    let residual = if max_range.residual.is_empty() {
                        None
                    } else {
                        let residual = max_range.residual.to_range();
                        #[cfg(feature = "log_parallelism")]
                        log_debug!(
                            "[thread {}] Residual range {:?} is not empty (stolen), scheduling it for cleanup.",
                            self.id,
                            residual
                        );
                        Some(residual)
                    };

                    // Take the first item, and place the rest in this thread's own range.
                    let (taken, my_new_range) = stolen.increment_start();
                    my_atomic_range.store(my_new_range);
                    #[cfg(feature = "log_parallelism")]
                    {
                        self.stats.thefts += 1;
                        log_trace!(
                            "[thread {}] Stole range {}:{:?} from thread {}",
                            self.id,
                            taken,
                            my_new_range.to_range(),
                            max_index
                        );
                    }

                    return (Some(taken as usize), residual);
                }
                // Theft failed: update the range and retry.
                Err(loaded) => {
                    let (bounded, residual) = loaded.upper_bound(bound);
                    let range = OtherRange {
                        loaded,
                        bounded,
                        residual,
                    };
                    other_ranges[max_index] = range;
                    #[cfg(feature = "log_parallelism")]
                    {
                        self.stats.failed_thefts += 1;
                    }

                    // Re-compute the largest range.
                    max_range = range;
                    for (i, range) in other_ranges.iter().enumerate() {
                        if i == self.id {
                            continue;
                        }
                        if range.bounded.len() > max_range.bounded.len() {
                            max_index = i;
                            max_range = *range;
                        }
                    }
                }
            }
        }

        // Didn't manage to steal anything: exit the iterator.
        #[cfg(feature = "log_parallelism")]
        log_debug!("[thread {}] Didn't find anything to steal", self.id);
        (None, None)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    struct SkipIteratorWrapper<T: SkipIterator>(T);

    impl<T: SkipIterator> Iterator for SkipIteratorWrapper<T> {
        type Item = usize;

        fn next(&mut self) -> Option<Self::Item> {
            loop {
                match self.0.next() {
                    (None, Some(_)) => continue,
                    (next, _) => return next,
                }
            }
        }
    }

    #[test]
    fn test_fixed_range_factory_splits_evenly() {
        let factory = FixedRangeFactory::new(4);
        let ranges: [_; 4] = std::array::from_fn(|i| factory.range(i));
        factory.orchestrator().reset_ranges(200);
        assert_eq!(ranges[0].iter(), 0..50);
        assert_eq!(ranges[1].iter(), 50..100);
        assert_eq!(ranges[2].iter(), 100..150);
        assert_eq!(ranges[3].iter(), 150..200);

        let factory = FixedRangeFactory::new(7);
        let ranges: [_; 7] = std::array::from_fn(|i| factory.range(i));
        factory.orchestrator().reset_ranges(100);
        assert_eq!(ranges[0].iter(), 0..14);
        assert_eq!(ranges[1].iter(), 14..28);
        assert_eq!(ranges[2].iter(), 28..42);
        assert_eq!(ranges[3].iter(), 42..57);
        assert_eq!(ranges[4].iter(), 57..71);
        assert_eq!(ranges[5].iter(), 71..85);
        assert_eq!(ranges[6].iter(), 85..100);
    }

    #[test]
    fn test_fixed_range() {
        let factory = FixedRangeFactory::new(4);
        let ranges: [_; 4] = std::array::from_fn(|i| factory.range(i));
        let orchestrator = factory.orchestrator();

        std::thread::scope(|s| {
            for _ in 0..10 {
                orchestrator.reset_ranges(100);
                let handles = ranges.each_ref().map(|range| {
                    s.spawn(move || SkipIteratorWrapper(range.iter()).collect::<Vec<_>>())
                });
                let values: [Vec<usize>; 4] = handles.map(|handle| handle.join().unwrap());

                // The fixed range implementation always yields the same items in order.
                for (i, set) in values.iter().enumerate() {
                    assert_eq!(*set, (i * 25..(i + 1) * 25).collect::<Vec<_>>());
                }
            }
        });
    }

    #[test]
    fn test_work_stealing_range_multi_threaded() {
        const NUM_THREADS: usize = 4;
        #[cfg(not(miri))]
        const NUM_ELEMENTS: usize = 10000;
        #[cfg(miri)]
        const NUM_ELEMENTS: usize = 100;

        let factory = WorkStealingRangeFactory::new(NUM_THREADS);
        let ranges: [_; NUM_THREADS] = std::array::from_fn(|i| factory.range(i));
        let orchestrator = factory.orchestrator();

        std::thread::scope(|s| {
            for _ in 0..10 {
                orchestrator.reset_ranges(NUM_ELEMENTS);
                let handles = ranges.each_ref().map(|range| {
                    s.spawn(move || SkipIteratorWrapper(range.iter()).collect::<Vec<_>>())
                });
                let values: [Vec<usize>; NUM_THREADS] =
                    handles.map(|handle| handle.join().unwrap());

                // This checks that:
                // - all ranges yield disjoint elements,
                // - each range never yields the same element twice.
                let mut all_values = vec![false; NUM_ELEMENTS];
                for set in values {
                    println!("Values: {set:?}");
                    for x in set {
                        assert!(!all_values[x]);
                        all_values[x] = true;
                    }
                }
                // Check that the whole range is covered.
                assert!(all_values.iter().all(|x| *x));
            }
        });
    }

    #[test]
    #[should_panic(
        expected = "cannot spawn 10000000000 threads: only up to 4294967295 threads (2^32 - 1) are supported"
    )]
    fn test_work_stealing_range_too_many_threads() {
        WorkStealingRangeFactory::new(10_000_000_000);
    }

    #[test]
    #[should_panic(
        expected = "cannot process range of 10000000000 elements: only ranges of up to 4294967295 elements (2^32 - 1) are supported"
    )]
    fn test_work_stealing_range_too_many_items() {
        let factory = WorkStealingRangeFactory::new(4);
        let orchestrator = factory.orchestrator();
        orchestrator.reset_ranges(10_000_000_000);
    }

    #[test]
    fn test_work_stealing_range_many_items() {
        let factory = WorkStealingRangeFactory::new(4);
        let orchestrator = factory.orchestrator();
        orchestrator.reset_ranges(4_000_000_000);

        assert_eq!(
            orchestrator
                .ranges
                .iter()
                .map(|x| {
                    let range = x.load();
                    range.start()..range.end()
                })
                .collect::<Vec<_>>(),
            vec![
                0..1_000_000_000,
                1_000_000_000..2_000_000_000,
                2_000_000_000..3_000_000_000,
                3_000_000_000..4_000_000_000
            ]
        );
    }

    #[test]
    fn test_default_packed_range_is_empty() {
        let range = PackedRange::default();
        assert!(range.is_empty());
        assert_eq!(range.start(), 0);
        assert_eq!(range.end(), 0);
    }

    #[test]
    fn test_packed_range_is_consistent() {
        for i in 0..30 {
            for j in i..30 {
                let range = PackedRange::new(i, j);
                assert_eq!(range.start(), i);
                assert_eq!(range.end(), j);
            }
        }
    }

    #[test]
    fn test_packed_range_upper_bound() {
        let range = PackedRange::new(10, 20);
        for bound in 0..=10 {
            let (left, right) = range.upper_bound(bound as usize);
            assert!(left.is_empty());
            assert_eq!((right.start(), right.end()), (10, 20));
        }
        for bound in 11..=19 {
            let (left, right) = range.upper_bound(bound as usize);
            assert_eq!((left.start(), left.end()), (10, bound));
            assert_eq!((right.start(), right.end()), (bound, 20));
        }
        for bound in 20..=30 {
            let (left, right) = range.upper_bound(bound as usize);
            assert_eq!((left.start(), left.end()), (10, 20));
            assert!(right.is_empty());
        }
    }

    #[test]
    fn test_packed_range_increment_start() {
        let mut range = PackedRange::new(0, 10);

        for i in 1..=10 {
            let (j, new_range) = range.increment_start();
            range = new_range;
            assert_eq!(j, i - 1);
            assert_eq!((range.start(), range.end()), (i, 10));
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "assertion failed: self.start() < self.end()")]
    fn test_packed_range_increment_start_overflow() {
        let range = PackedRange::new(u32::MAX, u32::MAX);
        range.increment_start();
    }

    #[test]
    fn test_packed_range_split() {
        let (left, right) = PackedRange::new(0, 0).split();
        assert!(left.is_empty());
        assert_eq!((left.start(), left.end()), (0, 0));
        assert!(right.is_empty());
        assert_eq!((right.start(), right.end()), (0, 0));

        let (left, right) = PackedRange::new(0, 1).split();
        assert!(left.is_empty());
        assert_eq!((left.start(), left.end()), (0, 0));
        assert!(!right.is_empty());
        assert_eq!((right.start(), right.end()), (0, 1));
    }

    #[test]
    fn test_packed_range_split_overflow() {
        let range = PackedRange::new(u32::MAX, u32::MAX);
        let (left, right) = range.split();
        assert_eq!((left.start(), left.end()), (u32::MAX, u32::MAX));
        assert_eq!((right.start(), right.end()), (u32::MAX, u32::MAX));

        let range = PackedRange::new(u32::MAX - 2, u32::MAX);
        let (left, right) = range.split();
        assert_eq!((left.start(), left.end()), (u32::MAX - 2, u32::MAX - 1));
        assert_eq!((right.start(), right.end()), (u32::MAX - 1, u32::MAX));
    }

    #[test]
    fn test_packed_range_split_is_exhaustive() {
        for i in 0..100 {
            for j in i..100 {
                let (left, right) = PackedRange::new(i, j).split();
                assert!(left.start() <= left.end());
                assert!(right.start() <= right.end());
                assert_eq!(left.start(), i);
                assert_eq!(left.end(), right.start());
                assert_eq!(right.end(), j);
            }
        }
    }

    #[test]
    fn test_packed_range_split_is_fair() {
        for i in 0..100 {
            for j in i..100 {
                let (left, right) = PackedRange::new(i, j).split();
                assert!(left.end() - left.start() <= right.end() - right.start());
                assert!(right.end() - right.start() <= left.end() - left.start() + 1);
                if i != j {
                    assert!(!right.is_empty());
                }
            }
        }
    }
}
