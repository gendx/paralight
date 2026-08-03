// Copyright 2026 The Paralight Authors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Helper module to test a sink that panics when used.

use super::{ExactParallelSink, FromExactParallelSink};
use std::marker::PhantomData;

pub struct Titanic<T> {
    _phantom: PhantomData<T>,
}

impl<T: Send> FromExactParallelSink for Titanic<T> {
    type Item = T;
    type Sink = TitanicSink<T>;

    unsafe fn finalize(_sink: Self::Sink) -> Self {
        panic!("Titanic::finalize()");
    }
}

#[must_use = "iterator adaptors are lazy"]
pub struct TitanicSink<T: Send> {
    _phantom: PhantomData<T>,
}

impl<T: Send> ExactParallelSink for TitanicSink<T> {
    type Item = T;
    const NEEDS_CLEANUP: bool = std::mem::needs_drop::<T>();

    fn new(_len: usize) -> Self {
        Self {
            _phantom: PhantomData,
        }
    }

    unsafe fn push_item(&self, index: usize, _item: Self::Item) {
        panic!("Titanic::push_item({index})");
    }

    unsafe fn skip_item_range(&self, _range: std::ops::Range<usize>) {}

    unsafe fn cancel(self) {}
}

/// SAFETY: A [`Titanic`] doesn't actually store items of type `T`, so it's fine
/// to share it.
unsafe impl<T: Send> Sync for TitanicSink<T> {}
