// Copyright 2024-2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation details of iterator adaptors.

use super::{
    Accumulator, ExactSizeAccumulator, ParallelAdaptor, ParallelAdaptorDescriptor, ParallelIterator,
};
use crossbeam_utils::CachePadded;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::ControlFlow;
#[cfg(feature = "nightly")]
use std::ops::Try;
use std::sync::atomic::AtomicBool;

/// A fuse is an atomic object that starts unset and can transition once to the
/// set state.
///
/// Under the hood, this contains an atomic boolean aligned to a cache line to
/// avoid any risk of false sharing performance overhead.
pub struct Fuse(CachePadded<AtomicBool>);

/// State of a [`Fuse`].
enum FuseState {
    Unset,
    Set,
}

impl Fuse {
    /// Creates a new fuse in the [`Unset`](FuseState::Unset) state.
    pub fn new() -> Self {
        Fuse(CachePadded::new(AtomicBool::new(false)))
    }

    /// Reads the current state of this fuse.
    fn load(&self) -> FuseState {
        use std::sync::atomic::Ordering;

        match self.0.load(Ordering::Relaxed) {
            false => FuseState::Unset,
            true => FuseState::Set,
        }
    }

    /// Sets this fuse to the [`Set`](FuseState::Set) state.
    fn set(&self) {
        use std::sync::atomic::Ordering;

        self.0.store(true, Ordering::Relaxed)
    }
}

// ExactSizeAccumulator implementations.

pub struct IterReducer<Reduce> {
    pub(super) reduce: Reduce,
}

impl<Output, Reduce> ExactSizeAccumulator<Output, Output> for IterReducer<Reduce>
where
    Reduce: Fn(Output, Output) -> Output,
{
    #[inline(always)]
    fn accumulate_exact(&self, iter: impl ExactSizeIterator<Item = Output>) -> Output {
        iter.reduce(&self.reduce).unwrap()
    }
}

pub struct IterFolder<Init, Fold> {
    pub(super) init: Init,
    pub(super) fold: Fold,
}

impl<Item, Output, Init, Fold> ExactSizeAccumulator<Item, Output> for IterFolder<Init, Fold>
where
    Init: Fn(usize) -> Output,
    Fold: Fn(Output, Item) -> Output,
{
    #[inline(always)]
    fn accumulate_exact(&self, iter: impl ExactSizeIterator<Item = Item>) -> Output {
        let init = (self.init)(iter.len());
        iter.fold(init, &self.fold)
    }
}

// Accumulator implementations.

pub struct IterAccumulator<Init, ProcessItem, Finalize> {
    pub(super) init: Init,
    pub(super) process_item: ProcessItem,
    pub(super) finalize: Finalize,
}

impl<Item, Accum, Output, Init, ProcessItem, Finalize> Accumulator<Item, Output>
    for IterAccumulator<Init, ProcessItem, Finalize>
where
    Init: Fn() -> Accum,
    ProcessItem: Fn(Accum, Item) -> Accum,
    Finalize: Fn(Accum) -> Output,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output {
        let mut accumulator = (self.init)();
        for item in iter {
            accumulator = (self.process_item)(accumulator, item);
        }
        (self.finalize)(accumulator)
    }
}

pub struct ShortCircuitingAccumulator<Init, ProcessItem, Finalize> {
    pub(super) fuse: Fuse,
    pub(super) init: Init,
    pub(super) process_item: ProcessItem,
    pub(super) finalize: Finalize,
}

#[cfg(not(feature = "nightly"))]
impl<Item, Accum, Break, Output, Init, ProcessItem, Finalize> Accumulator<Item, Output>
    for ShortCircuitingAccumulator<Init, ProcessItem, Finalize>
where
    Init: Fn() -> Accum,
    ProcessItem: Fn(Accum, Item) -> ControlFlow<Break, Accum>,
    Finalize: Fn(ControlFlow<Break, Accum>) -> Output,
{
    #[inline(always)]
    fn accumulate(&self, mut iter: impl Iterator<Item = Item>) -> Output {
        let mut accumulator = (self.init)();
        let result = 'outer: {
            while let FuseState::Unset = self.fuse.load() {
                let Some(item) = iter.next() else {
                    break;
                };

                match (self.process_item)(accumulator, item) {
                    ControlFlow::Continue(acc) => {
                        accumulator = acc;
                        continue;
                    }
                    control_flow @ ControlFlow::Break(_) => {
                        self.fuse.set();
                        break 'outer control_flow;
                    }
                }
            }
            ControlFlow::Continue(accumulator)
        };
        (self.finalize)(result)
    }
}

#[cfg(feature = "nightly")]
impl<Item, Accum, R, Output, Init, ProcessItem, Finalize> Accumulator<Item, Output>
    for ShortCircuitingAccumulator<Init, ProcessItem, Finalize>
where
    Init: Fn() -> Accum,
    ProcessItem: Fn(Accum, Item) -> R,
    Finalize: Fn(R) -> Output,
    R: Try<Output = Accum>,
{
    #[inline(always)]
    fn accumulate(&self, mut iter: impl Iterator<Item = Item>) -> Output {
        let mut accumulator = (self.init)();
        let result = 'outer: {
            while let FuseState::Unset = self.fuse.load() {
                let Some(item) = iter.next() else {
                    break;
                };

                match (self.process_item)(accumulator, item).branch() {
                    ControlFlow::Continue(acc) => {
                        accumulator = acc;
                        continue;
                    }
                    ControlFlow::Break(residual) => {
                        self.fuse.set();
                        break 'outer R::from_residual(residual);
                    }
                }
            }
            R::from_output(accumulator)
        };
        (self.finalize)(result)
    }
}

pub struct AdaptorAccumulator<Inner, TransformItem> {
    pub(super) inner: Inner,
    pub(super) transform_item: TransformItem,
}

impl<InnerItem, Item, Output, Inner, TransformItem> Accumulator<InnerItem, Output>
    for AdaptorAccumulator<Inner, TransformItem>
where
    Inner: Accumulator<Item, Output>,
    TransformItem: Fn(InnerItem) -> Option<Item>,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = InnerItem>) -> Output {
        self.inner.accumulate(iter.filter_map(&self.transform_item))
    }
}

// Collect adaptor implementations.

pub struct IterCollector<T> {
    pub(super) _phantom: PhantomData<fn() -> T>,
}

impl<Item, T> Accumulator<Item, T> for IterCollector<T>
where
    T: FromIterator<Item>,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> T {
        T::from_iter(iter)
    }
}

// Product/Sum adaptor implementations on top of the (ExactSize)Accumulator
// traits.

pub struct SumAccumulator;

impl<Item, Output> Accumulator<Item, Output> for SumAccumulator
where
    Output: Sum<Item>,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output {
        iter.sum()
    }
}

impl<Item, Output> ExactSizeAccumulator<Item, Output> for SumAccumulator
where
    Output: Sum<Item>,
{
    #[inline(always)]
    fn accumulate_exact(&self, iter: impl ExactSizeIterator<Item = Item>) -> Output {
        iter.sum()
    }
}

pub struct ProductAccumulator;

impl<Item, Output> Accumulator<Item, Output> for ProductAccumulator
where
    Output: Product<Item>,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output {
        iter.product()
    }
}

impl<Item, Output> ExactSizeAccumulator<Item, Output> for ProductAccumulator
where
    Output: Product<Item>,
{
    #[inline(always)]
    fn accumulate_exact(&self, iter: impl ExactSizeIterator<Item = Item>) -> Output {
        iter.product()
    }
}

// Adaptor implementations.

/// This struct is created by the
/// [`cloned()`](super::ParallelIteratorExt::cloned)
/// method on [`ParallelIteratorExt`](super::ParallelIteratorExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and
/// [`ParallelIteratorExt`](super::ParallelIteratorExt) traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Cloned<Inner: ParallelIterator> {
    pub(super) inner: Inner,
}

impl<'a, T, Inner: ParallelIterator> ParallelAdaptor for Cloned<Inner>
where
    T: Clone + 'a,
    Inner: ParallelIterator<Item = &'a T>,
{
    type Item = T;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: |item| Some(item.clone()),
        }
    }
}

/// This struct is created by the
/// [`copied()`](super::ParallelIteratorExt::copied)
/// method on [`ParallelIteratorExt`](super::ParallelIteratorExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and
/// [`ParallelIteratorExt`](super::ParallelIteratorExt) traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Copied<Inner: ParallelIterator> {
    pub(super) inner: Inner,
}

impl<'a, T, Inner: ParallelIterator> ParallelAdaptor for Copied<Inner>
where
    T: Copy + 'a,
    Inner: ParallelIterator<Item = &'a T>,
{
    type Item = T;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: |item| Some(*item),
        }
    }
}

/// This struct is created by the
/// [`filter()`](super::ParallelIteratorExt::filter)
/// method on [`ParallelIteratorExt`](super::ParallelIteratorExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and
/// [`ParallelIteratorExt`](super::ParallelIteratorExt) traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Filter<Inner: ParallelIterator, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner: ParallelIterator, F> ParallelAdaptor for Filter<Inner, F>
where
    F: Fn(&Inner::Item) -> bool + Sync,
{
    type Item = Inner::Item;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: move |item| {
                if (self.f)(&item) {
                    Some(item)
                } else {
                    None
                }
            },
        }
    }
}

/// This struct is created by the
/// [`filter_map()`](super::ParallelIteratorExt::filter_map) method on
/// [`ParallelIteratorExt`](super::ParallelIteratorExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and
/// [`ParallelIteratorExt`](super::ParallelIteratorExt) traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct FilterMap<Inner: ParallelIterator, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner: ParallelIterator, T, F> ParallelAdaptor for FilterMap<Inner, F>
where
    F: Fn(Inner::Item) -> Option<T> + Sync,
{
    type Item = T;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: self.f,
        }
    }
}

/// This struct is created by the
/// [`inspect()`](super::ParallelIteratorExt::inspect)
/// method on [`ParallelIteratorExt`](super::ParallelIteratorExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and
/// [`ParallelIteratorExt`](super::ParallelIteratorExt) traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Inspect<Inner: ParallelIterator, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner: ParallelIterator, F> ParallelAdaptor for Inspect<Inner, F>
where
    F: Fn(&Inner::Item) + Sync,
{
    type Item = Inner::Item;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: move |item| {
                (self.f)(&item);
                Some(item)
            },
        }
    }
}

/// This struct is created by the [`map()`](super::ParallelIteratorExt::map)
/// method on [`ParallelIteratorExt`](super::ParallelIteratorExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and
/// [`ParallelIteratorExt`](super::ParallelIteratorExt) traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Map<Inner: ParallelIterator, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner: ParallelIterator, T, F> ParallelAdaptor for Map<Inner, F>
where
    F: Fn(Inner::Item) -> T + Sync,
{
    type Item = T;
    type Inner = Inner;

    fn descriptor(
        self,
    ) -> ParallelAdaptorDescriptor<
        Self::Item,
        Self::Inner,
        impl Fn(<Self::Inner as ParallelIterator>::Item) -> Option<Self::Item> + Sync,
    > {
        ParallelAdaptorDescriptor {
            inner: self.inner,
            transform_item: move |item| Some((self.f)(item)),
        }
    }
}

/// This struct is created by the
/// [`map_init()`](super::ParallelIteratorExt::map_init)
/// method on [`ParallelIteratorExt`](super::ParallelIteratorExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelIterator`] and
/// [`ParallelIteratorExt`](super::ParallelIteratorExt) traits, but
/// it is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct MapInit<Inner: ParallelIterator, Init, F> {
    pub(super) inner: Inner,
    pub(super) init: Init,
    pub(super) f: F,
}

impl<Inner: ParallelIterator, I, Init, T, F> ParallelIterator for MapInit<Inner, Init, F>
where
    Init: Fn() -> I + Sync,
    F: Fn(&mut I, Inner::Item) -> T + Sync,
{
    type Item = T;

    fn upper_bounded_pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> ControlFlow<Accum, Accum> + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.inner.upper_bounded_pipeline(
            || ((self.init)(), init()),
            |(mut i, accum), index, item| {
                let accum = process_item(accum, index, (self.f)(&mut i, item));
                match accum {
                    ControlFlow::Continue(accum) => ControlFlow::Continue((i, accum)),
                    ControlFlow::Break(accum) => ControlFlow::Break((i, accum)),
                }
            },
            |(_, accum)| finalize(accum),
            reduce,
        )
    }

    fn iter_pipeline<Output, Accum: Send>(
        self,
        accum: impl Accumulator<Self::Item, Accum> + Sync,
        reduce: impl ExactSizeAccumulator<Accum, Output>,
    ) -> Output {
        let accumulator = MapInitAccumulator {
            inner: accum,
            init: self.init,
            f: self.f,
        };
        self.inner.iter_pipeline(accumulator, reduce)
    }
}

struct MapInitAccumulator<Inner, Init, F> {
    inner: Inner,
    init: Init,
    f: F,
}

impl<Item, Output, Inner, I, Init, T, F> Accumulator<Item, Output>
    for MapInitAccumulator<Inner, Init, F>
where
    Inner: Accumulator<T, Output>,
    Init: Fn() -> I,
    F: Fn(&mut I, Item) -> T,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Item>) -> Output {
        let mut i = (self.init)();
        self.inner
            .accumulate(iter.map(|item| (self.f)(&mut i, item)))
    }
}
