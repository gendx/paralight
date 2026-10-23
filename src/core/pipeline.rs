// Copyright 2024-2025 The Paralight Authors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Pipelines to execute on worker threads.

use super::range::{Range, SkipIterator};
use crate::iter::{Accumulator, SourceCleanup};
use crossbeam_utils::CachePadded;
use std::ops::ControlFlow;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// A pipeline is a task to run on a worker thread, taking as input a range of
/// items to process.
pub trait Pipeline<R: Range> {
    /// Runs this pipeline for the given worker index, processing the items
    /// produced by the given range.
    fn run(&self, worker_id: usize, range: &R);
}

/// A pipeline implementation that accumulates items but skips indices based on
/// a dynamic upper bound.
///
/// This is useful to implement iterator combinators such as
/// [`find_first()`](crate::iter::ParallelIteratorExt::find_first).
pub struct UpperBoundedPipelineImpl<
    'a,
    Output,
    Accum,
    Init: Fn() -> Accum,
    ProcessItem: Fn(Accum, usize) -> ControlFlow<Accum, Accum>,
    Finalize: Fn(Accum) -> Output,
    Cleanup: SourceCleanup,
> {
    /// Total number of items to process.
    pub input_len: usize,
    /// Number of items to process at once in each chunk.
    pub chunk_by: usize,
    /// Dynamic upper bound of indices to process.
    pub bound: CachePadded<AtomicUsize>,
    /// Handle to the outputs of all the worker threads.
    pub outputs: Arc<[Mutex<Option<Output>>]>,
    /// Initialization function to create a new accumulator.
    pub init: Init,
    /// Processing function to run on each index.
    pub process_item: ProcessItem,
    /// Finalizing function to convert an accumulator into an output.
    pub finalize: Finalize,
    /// Cleanup function to run on indices that are skipped.
    pub cleanup: &'a Cleanup,
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
        if self.chunk_by == 1 {
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
        } else {
            let mut iter = ScaledSkipIteratorWrapper {
                iter: range.upper_bounded_iter(&self.bound),
                current: 0..0,
                scaling: self.chunk_by,
                max: self.input_len,
                cleanup: self.cleanup,
            };
            while let Some(i) = iter.next() {
                let acc = (self.process_item)(accumulator, i);
                accumulator = match acc {
                    ControlFlow::Continue(acc) => acc,
                    ControlFlow::Break(acc) => {
                        self.bound.fetch_min(i / self.chunk_by, Ordering::Relaxed);
                        // Also skip the rest of the current chunk.
                        iter.discard_current();
                        acc
                    }
                };
            }
        }
        let output = (self.finalize)(accumulator);
        *self.outputs[worker_id].lock().unwrap() = Some(output);
    }
}

/// A pipeline implementation that accumulates items from an iterator.
pub struct IterPipelineImpl<'a, Output, Accum: Accumulator<usize, Output>, Cleanup: SourceCleanup> {
    /// Handle to the outputs of all the worker threads.
    pub outputs: Arc<[Mutex<Option<Output>>]>,
    /// Accumulator function to run on the indices produced by the range passed
    /// as argument to [`run()`](Self::run).
    pub accum: Accum,
    /// Cleanup function to run on indices that are skipped.
    pub cleanup: &'a Cleanup,
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

pub struct ScaledIterPipelineImpl<
    'a,
    Output,
    Accum: Accumulator<usize, Output>,
    Cleanup: SourceCleanup,
> {
    /// Total number of items to process.
    pub input_len: usize,
    /// Number of items to process at once in each chunk.
    pub chunk_by: usize,
    /// Handle to the outputs of all the worker threads.
    pub outputs: Arc<[Mutex<Option<Output>>]>,
    /// Accumulator function to run on the indices produced by the range passed
    /// as argument to [`run()`](Self::run).
    pub accum: Accum,
    /// Cleanup function to run on indices that are skipped.
    pub cleanup: &'a Cleanup,
}

impl<R, Output, Accum, Cleanup> Pipeline<R> for ScaledIterPipelineImpl<'_, Output, Accum, Cleanup>
where
    R: Range,
    Accum: Accumulator<usize, Output>,
    Cleanup: SourceCleanup,
{
    fn run(&self, worker_id: usize, range: &R) {
        let iter = ScaledSkipIteratorWrapper {
            iter: range.iter(),
            current: 0..0,
            scaling: self.chunk_by,
            max: self.input_len,
            cleanup: self.cleanup,
        };
        let output = self.accum.accumulate(iter);
        *self.outputs[worker_id].lock().unwrap() = Some(output);
    }
}

/// Wrapper around a [`SkipIterator`] to provide an [`Iterator`] implementation,
/// seamlessly running the given cleanup function on skipped indices.
struct SkipIteratorWrapper<'a, I: SkipIterator, Cleanup: SourceCleanup> {
    /// Underlying [`SkipIterator`].
    iter: I,
    /// Cleanup function to run on indices that are skipped.
    cleanup: &'a Cleanup,
}

impl<I: SkipIterator, Cleanup: SourceCleanup> Iterator for SkipIteratorWrapper<'_, I, Cleanup> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            match self.iter.next() {
                (index, None) => return index,
                (index, Some(skipped_range)) => {
                    // SAFETY: Due to the safety guarantees of `RangeFactory`:
                    // - `skipped_range` is included in the range `0..input_len`
                    //   (where `input_len` is the parameter to the
                    //   `ThreadPool::*_pipeline()` call),
                    // - `skipped_range` doesn't overlap with any other range
                    //   passed to `cleanup_item_range()` (here or in the `Drop`
                    //   implementation) nor index passed to `fetch_item()` (any
                    //   index returned by this `next()` function).
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
            // - `range` is included in the range `0..input_len` (where
            //   `input_len` is the parameter to the `ThreadPool::*_pipeline()`
            //   call),
            // - `range` doesn't overlap with any other range passed to
            //   `cleanup_item_range()` (here or in the `next()` implementation)
            //   nor index passed to `fetch_item()` (any index returned by the
            //   `next()` function).
            unsafe {
                self.cleanup.cleanup_item_range(range);
            }
        }
    }
}

/// Wrapper around a [`SkipIterator`] to provide an [`Iterator`] implementation,
/// where each index is scaled by the given `scaling` factor and clamped to
/// `0..max`, seamlessly running the given cleanup function on skipped indices.
///
/// Each underlying index is scaled to `i * scaling..(i + 1) * scaling`, then
/// clamped to `0..max`.
struct ScaledSkipIteratorWrapper<'a, I: SkipIterator, Cleanup: SourceCleanup> {
    /// Underlying [`SkipIterator`].
    iter: I,
    /// Cleanup function to run on indices that are skipped.
    cleanup: &'a Cleanup,
    /// Scaling factor.
    scaling: usize,
    /// Maximum clamping value.
    max: usize,
    /// Current range of expanded indices.
    current: std::ops::Range<usize>,
}

impl<I: SkipIterator, Cleanup: SourceCleanup> ScaledSkipIteratorWrapper<'_, I, Cleanup> {
    #[inline(always)]
    fn discard_current(&mut self) {
        if !self.current.is_empty() {
            // SAFETY: Due to the safety guarantees of `RangeFactory`:
            // - `current` is included in the range `0..max` (where `max` is the
            //   `input_len` parameter to the `ThreadPool::*_pipeline()` call),
            // - `current` doesn't overlap with any other range passed to
            //   `cleanup_item_range()` (here, in the `next()` implementation or
            //   in the `Drop` implementation) nor index passed to
            //   `fetch_item()` (any index returned by the `next()` function).
            //   Indeed, the current range is reset just below.
            unsafe {
                self.cleanup.cleanup_item_range(self.current.clone());
            }
            self.current = 0..0;
        }
    }
}

impl<I: SkipIterator, Cleanup: SourceCleanup> Iterator
    for ScaledSkipIteratorWrapper<'_, I, Cleanup>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(index) = Iterator::next(&mut self.current) {
                return Some(index);
            }
            match self.iter.next() {
                (index, None) => {
                    return index.and_then(|index| {
                        self.current = self.scale_index(index);
                        Iterator::next(&mut self.current)
                    });
                }
                (index, Some(skipped_range)) => {
                    // SAFETY: Due to the safety guarantees of `RangeFactory`:
                    // - `scale(skipped_range)` is included in the range
                    //   `0..max` (where `max` is the `input_len` parameter to
                    //   the `ThreadPool::*_pipeline()` call),
                    // - `scale(skipped_range)` doesn't overlap with any other
                    //   range passed to `cleanup_item_range()` (here, in the
                    //   `discard_current()` implementation or in the `Drop`
                    //   implementation) nor index passed to `fetch_item()` (any
                    //   index returned by this `next()` function).
                    unsafe {
                        self.cleanup.cleanup_item_range(self.scale(skipped_range));
                    }
                    if let Some(index) = index {
                        self.current = self.scale_index(index);
                        return Iterator::next(&mut self.current);
                    }
                }
            }
        }
    }
}

impl<I: SkipIterator, Cleanup: SourceCleanup> Drop for ScaledSkipIteratorWrapper<'_, I, Cleanup> {
    fn drop(&mut self) {
        if !self.current.is_empty() {
            // SAFETY: Due to the safety guarantees of `RangeFactory`:
            // - `current` is included in the range `0..max` (where `max` is the
            //   `input_len` parameter to the `ThreadPool::*_pipeline()` call),
            // - `current` doesn't overlap with any other range passed to
            //   `cleanup_item_range()` (here, in the `discard_current()`
            //   implementation or in the `next()` implementation) nor index
            //   passed to `fetch_item()` (any index returned by the `next()`
            //   function).
            unsafe {
                self.cleanup.cleanup_item_range(self.current.clone());
            }
        }
        if let Some(range) = self.iter.remaining_range() {
            // SAFETY: Due to the safety guarantees of `RangeFactory`:
            // - `scale(range)` is included in the range `0..max` (where `max`
            //   is the `input_len` parameter to the `ThreadPool::*_pipeline()`
            //   call),
            // - `scale(range)` doesn't overlap with any other range passed to
            //   `cleanup_item_range()` (here, in the `discard_current()`
            //   implementation or in the `next()` implementation) nor index
            //   passed to `fetch_item()` (any index returned by the `next()`
            //   function).
            unsafe {
                self.cleanup.cleanup_item_range(self.scale(range));
            }
        }
    }
}

impl<I: SkipIterator, Cleanup: SourceCleanup> ScaledSkipIteratorWrapper<'_, I, Cleanup> {
    /// Scale the given `start..end` range by this wrapper's scaling factor,
    /// clamping it to `0..max`.
    ///
    /// The implementation assumes that the scaled `start * scaling` fits within
    /// `0..max` and doesn't need to be explicitly clamped.
    #[inline(always)]
    fn scale(&self, range: std::ops::Range<usize>) -> std::ops::Range<usize> {
        scale_clamp_range(range, self.scaling, self.max)
    }

    /// Scale the length-one range `index..index + 1` by this wrapper's scaling
    /// factor, clamping it to `0..max`.
    ///
    /// The implementation assumes that the scaled start `index * scaling` fits
    /// within `0..max` and doesn't need to be explicitly clamped.
    #[inline(always)]
    fn scale_index(&self, index: usize) -> std::ops::Range<usize> {
        scale_clamp_index(index, self.scaling, self.max)
    }
}

/// Scale the given `start..end` range by the given `scaling` factor to `start *
/// scaling..end * scaling`, clamping it to `0..max`.
///
/// The implementation assumes that the scaled `start * scaling` fits within
/// `0..max` and doesn't need to be explicitly clamped.
#[inline(always)]
fn scale_clamp_range(
    range: std::ops::Range<usize>,
    scaling: usize,
    max: usize,
) -> std::ops::Range<usize> {
    let start = range.start * scaling;
    let end = range.end.saturating_mul(scaling).min(max);
    start..end
}

/// Scale the length-one range `index..index + 1` by the given `scaling` factor
/// to `index * scaling..(index + 1) * scaling`, clamping it to `0..max`.
///
/// The implementation assumes that the scaled start `index * scaling` fits
/// within `0..max` and doesn't need to be explicitly clamped.
#[inline(always)]
fn scale_clamp_index(index: usize, scaling: usize, max: usize) -> std::ops::Range<usize> {
    let start = index * scaling;
    let end = if max - start >= scaling {
        start + scaling
    } else {
        max
    };
    start..end
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_scale_clamp_index() {
        assert_eq!(scale_clamp_index(0, 10, 42), 0..10);
        assert_eq!(scale_clamp_index(1, 10, 42), 10..20);
        assert_eq!(scale_clamp_index(2, 10, 42), 20..30);
        assert_eq!(scale_clamp_index(3, 10, 42), 30..40);
        assert_eq!(scale_clamp_index(4, 10, 42), 40..42);
        assert_eq!(
            scale_clamp_index(usize::MAX / 10 - 1, 10, usize::MAX - 1),
            (usize::MAX / 10 - 1) * 10..(usize::MAX / 10) * 10
        );
        assert_eq!(
            scale_clamp_index(usize::MAX / 10, 10, usize::MAX - 1),
            (usize::MAX / 10) * 10..usize::MAX - 1
        );
    }

    #[test]
    fn test_scale_clamp_range() {
        assert_eq!(scale_clamp_range(0..3, 10, 42), 0..30);
        assert_eq!(scale_clamp_range(2..5, 10, 42), 20..42);
        assert_eq!(scale_clamp_range(4..5, 10, 42), 40..42);
        assert_eq!(
            scale_clamp_range(42..usize::MAX / 10, 10, usize::MAX - 1),
            420..(usize::MAX / 10) * 10
        );
        assert_eq!(
            scale_clamp_range(42..usize::MAX / 10 + 1, 10, usize::MAX - 1),
            420..usize::MAX - 1
        );
    }
}
