// Copyright 2024-2025 Google LLC
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

/// Wrapper around a [`SkipIterator`] to provides an [`Iterator`]
/// implementation, seamlessly running the given cleanup function on skipped
/// indices.
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
