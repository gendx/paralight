// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Parallel sinks collect items from parallel iterators.

#[cfg(feature = "nightly")]
pub mod array;
pub mod vec;

/// A sink to collect items in parallel.
pub trait ExactParallelSink {
    /// The type of items that this parallel sink collects.
    type Item: Send;

    /// Set to [`false`] if the skip function is guaranteed to be a noop.
    ///
    /// Typically, skipping is a noop when [`std::mem::needs_drop()`] returns
    /// false for the [`Item`](Self::Item) type.
    const NEEDS_CLEANUP: bool;

    /// Creates a new sink able to collect the given number of items.
    fn new(len: usize) -> Self;

    /// Pushes the given item at the given index of this sink.
    ///
    /// # Safety
    ///
    /// Given the length `len` passed to the [`new()`](Self::new) call that
    /// initialized this sink:
    /// - indices passed to [`push_item()`](Self::push_item) must be in the
    ///   `0..len` range,
    /// - each index in `0..len` must be present at most once in all indices
    ///   passed to calls to [`push_item()`](Self::push_item) and ranges passed
    ///   to calls to [`skip_item_range()`](Self::skip_item_range).
    ///
    /// It is therefore undefined behavior to call this function twice with the
    /// same index, with an index contained in a range for which
    /// [`skip_item_range()`](Self::skip_item_range) was invoked, etc.
    ///
    /// You normally shouldn't have to worry about this, because this API is
    /// intended to be called by Paralight's internal multi-threading engine.
    /// This API is public to allow others to implement parallel sinks: when
    /// implementing your own sink(s), you can rely on these `unsafe`
    /// pre-conditions.
    unsafe fn push_item(&self, index: usize, item: Self::Item);

    /// Indicates that the items in the given range won't be pushed.
    ///
    /// # Safety
    ///
    /// Given the length `len` passed to the [`new()`](Self::new) call that
    /// initialized this sink:
    /// - ranges passed to [`skip_item_range()`](Self::skip_item_range) must be
    ///   included in the `0..len` range,
    /// - each index in `0..len` must be present at most once in all indices
    ///   passed to calls to [`push_item()`](Self::push_item) and ranges passed
    ///   to calls to [`skip_item_range()`](Self::skip_item_range).
    ///
    /// It is therefore undefined behavior to call this function twice with the
    /// same range, with overlapping ranges, with a range that contains an
    /// index for which [`push_item()`](Self::push_item) was invoked, etc.
    ///
    /// You normally shouldn't have to worry about this, because this API is
    /// intended to be called by Paralight's internal multi-threading engine.
    /// This API is public to allow others to implement parallel sinks: when
    /// implementing your own sink(s), you can rely on these `unsafe`
    /// pre-conditions.
    unsafe fn skip_item_range(&self, range: std::ops::Range<usize>);

    /// Cancel and cleanup this sink.
    ///
    /// The purpose of this function is to properly cleanup the sink during
    /// [unwinding](https://doc.rust-lang.org/nomicon/unwinding.html) if a panic
    /// occured before all items have been pushed.
    ///
    /// # Safety
    ///
    /// This can only be called after all indices have been passed once and only
    /// once to [`push_item()`](Self::push_item) and
    /// [`skip_item_range()`](Self::skip_item_range).
    unsafe fn cancel(self);
}

/// Trait for collecting items from an [`ExactParallelSink`].
pub trait FromExactParallelSink {
    /// The type of items that this parallel sink collects.
    type Item: Send;

    /// Sink from which this type can be created.
    type Sink: ExactParallelSink<Item = Self::Item> + Sync;

    /// Converts a fully populated sink into this type.
    ///
    /// # Safety
    ///
    /// - This can only be called after all indices have been passed once and
    ///   only once to the sink's [`push_item()`](ExactParallelSink::push_item).
    /// - No call to [`skip_item_range()`](ExactParallelSink::skip_item_range)
    ///   must have been made on the sink (otherwise, you need to call
    ///   [`cancel()`](ExactParallelSink::cancel) instead).
    unsafe fn finalize(sink: Self::Sink) -> Self;
}
