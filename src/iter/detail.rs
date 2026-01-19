// Copyright 2024-2026 Google LLC
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
use std::cmp::Ordering;
use std::iter::{Product, Sum};
use std::marker::PhantomData;
use std::ops::ControlFlow;
#[cfg(feature = "nightly")]
use std::ops::{FromResidual, Residual, Try};
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

pub struct TryIterFolder<Init, TryFold> {
    pub(super) init: Init,
    pub(super) try_fold: TryFold,
}

#[cfg(not(feature = "nightly"))]
impl<Item, Output, E, Init, TryFold> ExactSizeAccumulator<Item, Result<Output, E>>
    for TryIterFolder<Init, TryFold>
where
    Init: Fn(usize) -> Output,
    TryFold: Fn(Output, Item) -> Result<Output, E>,
{
    #[inline(always)]
    fn accumulate_exact(&self, mut iter: impl ExactSizeIterator<Item = Item>) -> Result<Output, E> {
        let init = (self.init)(iter.len());
        iter.try_fold(init, &self.try_fold)
    }
}

#[cfg(feature = "nightly")]
impl<Item, Output, R, Init, TryFold> ExactSizeAccumulator<Item, R> for TryIterFolder<Init, TryFold>
where
    Init: Fn(usize) -> Output,
    TryFold: Fn(Output, Item) -> R,
    R: Try<Output = Output>,
{
    #[inline(always)]
    fn accumulate_exact(&self, mut iter: impl ExactSizeIterator<Item = Item>) -> R {
        let init = (self.init)(iter.len());
        iter.try_fold(init, &self.try_fold)
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

pub struct TryIterCollector<T> {
    pub(super) fuse: Fuse,
    pub(super) _phantom: PhantomData<fn() -> T>,
}

#[cfg(not(feature = "nightly"))]
impl<Item, C, E> Accumulator<Result<Item, E>, Result<C, E>> for TryIterCollector<C>
where
    C: FromIterator<Item>,
{
    #[inline(always)]
    fn accumulate(&self, iter: impl Iterator<Item = Result<Item, E>>) -> Result<C, E> {
        let mut error = None;
        let c = C::from_iter(iter.map_while(|item| match item {
            Err(e) => {
                self.fuse.set();
                error = Some(e);
                None
            }
            Ok(x) => match self.fuse.load() {
                FuseState::Set => None,
                FuseState::Unset => Some(x),
            },
        }));
        match error {
            None => Ok(c),
            Some(e) => Err(e),
        }
    }
}

#[cfg(feature = "nightly")]
impl<Item, C> Accumulator<Item, <Item::Residual as Residual<C>>::TryType> for TryIterCollector<C>
where
    Item: Try,
    Item::Residual: Residual<C>,
    C: FromIterator<Item::Output>,
{
    #[inline(always)]
    fn accumulate(
        &self,
        iter: impl Iterator<Item = Item>,
    ) -> <Item::Residual as Residual<C>>::TryType {
        let mut residual = None;
        let c = C::from_iter(iter.map_while(|item| match item.branch() {
            ControlFlow::Break(e) => {
                self.fuse.set();
                residual = Some(e);
                None
            }
            ControlFlow::Continue(x) => match self.fuse.load() {
                FuseState::Set => None,
                FuseState::Unset => Some(x),
            },
        }));
        match residual {
            None => Try::from_output(c),
            Some(e) => FromResidual::from_residual(e),
        }
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

/// The result of the [`minmax()`](super::ParallelIteratorExt::minmax) and
/// related iterator adaptors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MinMaxResult<T> {
    /// The iterator was empty.
    NoElements,
    /// The iterator only had one item.
    OneElement(T),
    /// The iterator had at least two items.
    MinMax {
        /// The minimal item.
        min: T,
        /// The maximal item.
        max: T,
    },
}

impl<T> MinMaxResult<T> {
    pub(super) fn map<U, F>(self, f: F) -> MinMaxResult<U>
    where
        F: Fn(T) -> U,
    {
        match self {
            MinMaxResult::NoElements => MinMaxResult::NoElements,
            MinMaxResult::OneElement(x) => MinMaxResult::OneElement(f(x)),
            MinMaxResult::MinMax { min, max } => MinMaxResult::MinMax {
                min: f(min),
                max: f(max),
            },
        }
    }

    /// Returns a reference to the minimal element, if this result isn't empty.
    ///
    /// ```
    /// # use paralight::iter::MinMaxResult;
    /// assert_eq!(MinMaxResult::<i32>::NoElements.min(), None);
    /// assert_eq!(MinMaxResult::OneElement(42).min(), Some(&42));
    /// assert_eq!(MinMaxResult::MinMax { min: 42, max: 123 }.min(), Some(&42));
    /// ```
    pub fn min(&self) -> Option<&T> {
        match self {
            MinMaxResult::NoElements => None,
            MinMaxResult::OneElement(x) => Some(x),
            MinMaxResult::MinMax { min, max: _ } => Some(min),
        }
    }

    /// Returns the minimal element, if this result isn't empty.
    ///
    /// ```
    /// # use paralight::iter::MinMaxResult;
    /// assert_eq!(MinMaxResult::<i32>::NoElements.into_min(), None);
    /// assert_eq!(MinMaxResult::OneElement(42).into_min(), Some(42));
    /// assert_eq!(
    ///     MinMaxResult::MinMax { min: 42, max: 123 }.into_min(),
    ///     Some(42)
    /// );
    /// ```
    pub fn into_min(self) -> Option<T> {
        match self {
            MinMaxResult::NoElements => None,
            MinMaxResult::OneElement(x) => Some(x),
            MinMaxResult::MinMax { min, max: _ } => Some(min),
        }
    }

    /// Returns a reference to the maximal element, if this result isn't empty.
    ///
    /// ```
    /// # use paralight::iter::MinMaxResult;
    /// assert_eq!(MinMaxResult::<i32>::NoElements.max(), None);
    /// assert_eq!(MinMaxResult::OneElement(42).max(), Some(&42));
    /// assert_eq!(MinMaxResult::MinMax { min: 42, max: 123 }.max(), Some(&123));
    /// ```
    pub fn max(&self) -> Option<&T> {
        match self {
            MinMaxResult::NoElements => None,
            MinMaxResult::OneElement(x) => Some(x),
            MinMaxResult::MinMax { min: _, max } => Some(max),
        }
    }

    /// Returns the maximal element, if this result isn't empty.
    ///
    /// ```
    /// # use paralight::iter::MinMaxResult;
    /// assert_eq!(MinMaxResult::<i32>::NoElements.into_max(), None);
    /// assert_eq!(MinMaxResult::OneElement(42).into_max(), Some(42));
    /// assert_eq!(
    ///     MinMaxResult::MinMax { min: 42, max: 123 }.into_max(),
    ///     Some(123)
    /// );
    /// ```
    pub fn into_max(self) -> Option<T> {
        match self {
            MinMaxResult::NoElements => None,
            MinMaxResult::OneElement(x) => Some(x),
            MinMaxResult::MinMax { min: _, max } => Some(max),
        }
    }

    /// Returns references to the minimal-maximal elements, if this result isn't
    /// empty.
    ///
    /// ```
    /// # use paralight::iter::MinMaxResult;
    /// assert_eq!(MinMaxResult::<i32>::NoElements.as_option(), None);
    /// assert_eq!(MinMaxResult::OneElement(42).as_option(), Some((&42, &42)));
    /// assert_eq!(
    ///     MinMaxResult::MinMax { min: 42, max: 123 }.as_option(),
    ///     Some((&42, &123))
    /// );
    /// ```
    pub fn as_option(&self) -> Option<(&T, &T)>
    where
        T: Clone,
    {
        match self {
            MinMaxResult::NoElements => None,
            MinMaxResult::OneElement(x) => Some((x, x)),
            MinMaxResult::MinMax { min, max } => Some((min, max)),
        }
    }

    /// Returns the minimal-maximal pair, if this result isn't empty.
    ///
    /// ```
    /// # use paralight::iter::MinMaxResult;
    /// assert_eq!(MinMaxResult::<i32>::NoElements.into_option(), None);
    /// assert_eq!(MinMaxResult::OneElement(42).into_option(), Some((42, 42)));
    /// assert_eq!(
    ///     MinMaxResult::MinMax { min: 42, max: 123 }.into_option(),
    ///     Some((42, 123))
    /// );
    /// ```
    pub fn into_option(self) -> Option<(T, T)>
    where
        T: Clone,
    {
        match self {
            MinMaxResult::NoElements => None,
            MinMaxResult::OneElement(x) => Some((x.clone(), x)),
            MinMaxResult::MinMax { min, max } => Some((min, max)),
        }
    }
}

pub struct MinMaxAccumulator<F> {
    pub(super) f: F,
}

impl<Item, F> Accumulator<Item, MinMaxResult<Item>> for &MinMaxAccumulator<F>
where
    F: Fn(&Item, &Item) -> Ordering,
{
    #[inline(always)]
    fn accumulate(&self, mut iter: impl Iterator<Item = Item>) -> MinMaxResult<Item> {
        let (mut min, mut max) = match iter.next() {
            None => return MinMaxResult::NoElements,
            Some(x) => match iter.next() {
                None => return MinMaxResult::OneElement(x),
                Some(y) => match (self.f)(&x, &y) {
                    Ordering::Less | Ordering::Equal => (x, y),
                    Ordering::Greater => (y, x),
                },
            },
        };

        loop {
            let a = match iter.next() {
                Some(a) => a,
                None => break,
            };
            let b = match iter.next() {
                Some(b) => b,
                None => {
                    if let Ordering::Less = (self.f)(&a, &min) {
                        min = a;
                    } else if let Ordering::Greater | Ordering::Equal = (self.f)(&a, &max) {
                        max = a;
                    }
                    break;
                }
            };

            match (self.f)(&a, &b) {
                Ordering::Less | Ordering::Equal => {
                    if let Ordering::Less = (self.f)(&a, &min) {
                        min = a;
                    }
                    if let Ordering::Greater | Ordering::Equal = (self.f)(&b, &max) {
                        max = b;
                    }
                }
                Ordering::Greater => {
                    if let Ordering::Less = (self.f)(&b, &min) {
                        min = b;
                    }
                    if let Ordering::Greater | Ordering::Equal = (self.f)(&a, &max) {
                        max = a;
                    }
                }
            }
        }

        MinMaxResult::MinMax { min, max }
    }
}

impl<Item, F> ExactSizeAccumulator<MinMaxResult<Item>, MinMaxResult<Item>> for &MinMaxAccumulator<F>
where
    F: Fn(&Item, &Item) -> Ordering,
{
    #[inline(always)]
    fn accumulate_exact(
        &self,
        iter: impl ExactSizeIterator<Item = MinMaxResult<Item>>,
    ) -> MinMaxResult<Item> {
        iter.fold(MinMaxResult::NoElements, |x, y| match (x, y) {
            (MinMaxResult::NoElements, z) | (z, MinMaxResult::NoElements) => z,
            (MinMaxResult::OneElement(a), MinMaxResult::OneElement(b)) => match (self.f)(&a, &b) {
                Ordering::Less | Ordering::Equal => MinMaxResult::MinMax { min: a, max: b },
                Ordering::Greater => MinMaxResult::MinMax { min: b, max: a },
            },
            (MinMaxResult::OneElement(z), MinMaxResult::MinMax { min, max }) => {
                if let Ordering::Less | Ordering::Equal = (self.f)(&z, &min) {
                    MinMaxResult::MinMax { min: z, max }
                } else if let Ordering::Greater = (self.f)(&z, &max) {
                    MinMaxResult::MinMax { min, max: z }
                } else {
                    MinMaxResult::MinMax { min, max }
                }
            }
            (MinMaxResult::MinMax { min, max }, MinMaxResult::OneElement(z)) => {
                if let Ordering::Less = (self.f)(&z, &min) {
                    MinMaxResult::MinMax { min: z, max }
                } else if let Ordering::Greater | Ordering::Equal = (self.f)(&z, &max) {
                    MinMaxResult::MinMax { min, max: z }
                } else {
                    MinMaxResult::MinMax { min, max }
                }
            }
            (
                MinMaxResult::MinMax {
                    min: min1,
                    max: max1,
                },
                MinMaxResult::MinMax {
                    min: min2,
                    max: max2,
                },
            ) => {
                let min = match (self.f)(&min1, &min2) {
                    Ordering::Less | Ordering::Equal => min1,
                    Ordering::Greater => min2,
                };
                let max = match (self.f)(&max1, &max2) {
                    Ordering::Greater => max1,
                    Ordering::Less | Ordering::Equal => max2,
                };
                MinMaxResult::MinMax { min, max }
            }
        })
    }
}

// Adaptor implementations.

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn minmaxresult_min() {
        assert_eq!(MinMaxResult::<i32>::NoElements.min(), None);
        assert_eq!(MinMaxResult::OneElement(42).min(), Some(&42));
        assert_eq!(MinMaxResult::MinMax { min: 42, max: 123 }.min(), Some(&42));
    }

    #[test]
    fn minmaxresult_into_min() {
        assert_eq!(MinMaxResult::<i32>::NoElements.into_min(), None);
        assert_eq!(MinMaxResult::OneElement(42).into_min(), Some(42));
        assert_eq!(
            MinMaxResult::MinMax { min: 42, max: 123 }.into_min(),
            Some(42)
        );
    }

    #[test]
    fn minmaxresult_max() {
        assert_eq!(MinMaxResult::<i32>::NoElements.max(), None);
        assert_eq!(MinMaxResult::OneElement(42).max(), Some(&42));
        assert_eq!(MinMaxResult::MinMax { min: 42, max: 123 }.max(), Some(&123));
    }

    #[test]
    fn minmaxresult_into_max() {
        assert_eq!(MinMaxResult::<i32>::NoElements.into_max(), None);
        assert_eq!(MinMaxResult::OneElement(42).into_max(), Some(42));
        assert_eq!(
            MinMaxResult::MinMax { min: 42, max: 123 }.into_max(),
            Some(123)
        );
    }

    #[test]
    fn minmaxresult_as_option() {
        assert_eq!(MinMaxResult::<i32>::NoElements.as_option(), None);
        assert_eq!(MinMaxResult::OneElement(42).as_option(), Some((&42, &42)));
        assert_eq!(
            MinMaxResult::MinMax { min: 42, max: 123 }.as_option(),
            Some((&42, &123))
        );
    }

    #[test]
    fn minmaxresult_into_option() {
        assert_eq!(MinMaxResult::<i32>::NoElements.into_option(), None);
        assert_eq!(MinMaxResult::OneElement(42).into_option(), Some((42, 42)));
        assert_eq!(
            MinMaxResult::MinMax { min: 42, max: 123 }.into_option(),
            Some((42, 123))
        );
    }

    #[test]
    fn minmax_accumulator_edge_cases() {
        let accumulator = &MinMaxAccumulator {
            f: |x: &(i32, i32), y: &(i32, i32)| x.0.cmp(&y.0),
        };

        // OneElement + OneElement
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::OneElement((0, 0)),
                    MinMaxResult::OneElement((0, 1))
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 0),
                max: (0, 1),
            }
        );
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::OneElement((1, 0)),
                    MinMaxResult::OneElement((0, 1))
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 1),
                max: (1, 0),
            }
        );

        // OneElement + MinMax
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::OneElement((0, 0)),
                    MinMaxResult::MinMax {
                        min: (0, 1),
                        max: (0, 2),
                    }
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 0),
                max: (0, 2),
            }
        );
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::OneElement((1, 0)),
                    MinMaxResult::MinMax {
                        min: (0, 1),
                        max: (2, 2),
                    }
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 1),
                max: (2, 2),
            }
        );
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::OneElement((0, 0)),
                    MinMaxResult::MinMax {
                        min: (1, 1),
                        max: (2, 2),
                    }
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 0),
                max: (2, 2),
            }
        );
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::OneElement((2, 0)),
                    MinMaxResult::MinMax {
                        min: (0, 1),
                        max: (1, 2),
                    }
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 1),
                max: (2, 0),
            }
        );

        // MinMax + OneElement
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::MinMax {
                        min: (0, 0),
                        max: (0, 1),
                    },
                    MinMaxResult::OneElement((0, 2)),
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 0),
                max: (0, 2),
            }
        );
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::MinMax {
                        min: (0, 0),
                        max: (2, 1),
                    },
                    MinMaxResult::OneElement((1, 2)),
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 0),
                max: (2, 1),
            }
        );
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::MinMax {
                        min: (1, 0),
                        max: (2, 1),
                    },
                    MinMaxResult::OneElement((0, 2)),
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 2),
                max: (2, 1),
            }
        );
        assert_eq!(
            accumulator.accumulate_exact(
                [
                    MinMaxResult::MinMax {
                        min: (0, 0),
                        max: (1, 1),
                    },
                    MinMaxResult::OneElement((2, 2)),
                ]
                .into_iter()
            ),
            MinMaxResult::MinMax {
                min: (0, 0),
                max: (2, 2),
            }
        );
    }
}
