// Copyright 2024-2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Implementation details of source adaptors.

use super::{
    ExactParallelSource, ExactSourceDescriptor, ParallelSource, SourceCleanup, SourceDescriptor,
};

/// This struct is created by the [`chain()`](super::ParallelSourceExt::chain)
/// method on [`ParallelSourceExt`](super::ParallelSourceExt) and
/// [`chain()`](super::ExactParallelSourceExt::chain) on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the
/// [`ParallelSource`]/[`ParallelSourceExt`](super::ParallelSourceExt) or
/// [`ExactParallelSource`]/
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Chain<First, Second> {
    pub(super) first: First,
    pub(super) second: Second,
}

impl<T: Send, First, Second> ParallelSource for Chain<First, Second>
where
    First: ParallelSource<Item = T>,
    Second: ParallelSource<Item = T>,
{
    type Item = T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor1 = self.first.descriptor();
        let descriptor2 = self.second.descriptor();

        let len1 = descriptor1.len();
        let len2 = descriptor2.len();
        let len = len1.checked_add(len2).unwrap_or_else(|| {
            panic!(
                "called chain() with sources that together produce more than usize::MAX items ({})",
                usize::MAX
            );
        });

        ChainSourceDescriptor {
            descriptor1,
            descriptor2,
            len,
            len1,
        }
    }
}

impl<T: Send, First, Second> ExactParallelSource for Chain<First, Second>
where
    First: ExactParallelSource<Item = T>,
    Second: ExactParallelSource<Item = T>,
{
    type Item = T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let descriptor1 = self.first.exact_descriptor();
        let descriptor2 = self.second.exact_descriptor();

        let len1 = descriptor1.len();
        let len2 = descriptor2.len();
        let len = len1.checked_add(len2).unwrap_or_else(|| {
            panic!(
                "called chain() with sources that together produce more than usize::MAX items ({})",
                usize::MAX
            );
        });

        ChainSourceDescriptor {
            descriptor1,
            descriptor2,
            len,
            len1,
        }
    }
}

struct ChainSourceDescriptor<First, Second> {
    descriptor1: First,
    descriptor2: Second,
    len: usize,
    len1: usize,
}

impl<First, Second> SourceCleanup for ChainSourceDescriptor<First, Second>
where
    First: SourceCleanup,
    Second: SourceCleanup,
{
    const NEEDS_CLEANUP: bool = First::NEEDS_CLEANUP || Second::NEEDS_CLEANUP;

    fn len(&self) -> usize {
        self.len
    }

    // For safety comments: given two sources of lengths `len1` and `len2`, the
    // `ChainSourceDescriptor` creates a bijection of indices between `0..len1 +
    // len2` and `0..len1 | 0..len2`.
    //
    // Therefore:
    // - if the caller passes ranges included in `0..len1 + len2`, ranges passed to
    //   the two downstream `cleanup_item_range()` functions are included in their
    //   respective ranges `0..len1` and `0..len2`,
    // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
    //   and `(exact_)fetch_item()`, the chain adaptor doesn't repeat indices passed
    //   to the two downstream descriptors.
    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= self.len);
            debug_assert!(range.end <= self.len);
            if range.end <= self.len1 {
                // SAFETY: See the function comment. This branch implements the mapping for a
                // range fully included in `0..len1` to `0..len1`.
                unsafe {
                    self.descriptor1.cleanup_item_range(range);
                }
            } else if range.start >= self.len1 {
                // SAFETY: See the function comment. This branch implements the mapping for a
                // range fully included in `len1..len1 + len2` to `0..len2`.
                unsafe {
                    self.descriptor2
                        .cleanup_item_range(range.start - self.len1..range.end - self.len1);
                }
            } else {
                // SAFETY: See the function comment. This branch implements the mapping for a
                // range that overlaps with `len1`.
                //
                // This line implements the mapping of the first half of the range (included in
                // `0..len1`) to `0..len1`.
                unsafe {
                    self.descriptor1.cleanup_item_range(range.start..self.len1);
                }
                // SAFETY: This line implements the mapping of the second half of the range
                // (included in `len1..len1 + len2`) to `0..len2`.
                unsafe {
                    self.descriptor2
                        .cleanup_item_range(0..range.end - self.len1);
                }
            }
        }
    }
}

impl<T: Send, First, Second> SourceDescriptor for ChainSourceDescriptor<First, Second>
where
    First: SourceDescriptor<Item = T>,
    Second: SourceDescriptor<Item = T>,
{
    type Item = T;

    // For safety comments: given two sources of lengths `len1` and `len2`, the
    // `ChainSourceDescriptor` creates a bijection of indices between `0..len1 +
    // len2` and `0..len1 | 0..len2`.
    //
    // Therefore:
    // - if the caller passes indices in `0..len1 + len2`, indices passed to the two
    //   downstream `fetch_item()` functions are included in their respective ranges
    //   `0..len1` and `0..len2`,
    // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
    //   and `fetch_item()`, the chain adaptor doesn't repeat indices passed to the
    //   two downstream descriptors.
    unsafe fn fetch_item(&self, index: usize) -> Option<Self::Item> {
        debug_assert!(index < self.len);
        if index < self.len1 {
            // SAFETY: See the function comment. This branch implements the mapping for an
            // index in `0..len1` to `0..len1`.
            unsafe { self.descriptor1.fetch_item(index) }
        } else {
            // SAFETY: See the function comment. This branch implements the mapping for an
            // index in `len1..len1 + len2` to `0..len2`.
            unsafe { self.descriptor2.fetch_item(index - self.len1) }
        }
    }
}

impl<T: Send, First, Second> ExactSourceDescriptor for ChainSourceDescriptor<First, Second>
where
    First: ExactSourceDescriptor<Item = T>,
    Second: ExactSourceDescriptor<Item = T>,
{
    type Item = T;

    // For safety comments: given two sources of lengths `len1` and `len2`, the
    // `ChainSourceDescriptor` creates a bijection of indices between `0..len1 +
    // len2` and `0..len1 | 0..len2`.
    //
    // Therefore:
    // - if the caller passes indices in `0..len1 + len2`, indices passed to the two
    //   downstream `exact_fetch_item()` functions are included in their respective
    //   ranges `0..len1` and `0..len2`,
    // - if the caller doesn't repeat indices when calling `cleanup_item_range()`
    //   and `exact_fetch_item()`, the chain adaptor doesn't repeat indices passed
    //   to the two downstream descriptors.
    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        if index < self.len1 {
            // SAFETY: See the function comment. This branch implements the mapping for an
            // index in `0..len1` to `0..len1`.
            unsafe { self.descriptor1.exact_fetch_item(index) }
        } else {
            // SAFETY: See the function comment. This branch implements the mapping for an
            // index in `len1..len1 + len2` to `0..len2`.
            unsafe { self.descriptor2.exact_fetch_item(index - self.len1) }
        }
    }
}

/// This struct is created by the [`cloned()`](super::ParallelSourceExt::cloned)
/// method on [`ParallelSourceExt`](super::ParallelSourceExt) and
/// [`cloned()`](super::ExactParallelSourceExt::cloned) on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the
/// [`ParallelSource`]/[`ParallelSourceExt`](super::ParallelSourceExt) or
/// [`ExactParallelSource`]/
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Cloned<Inner> {
    pub(super) inner: Inner,
}

impl<'a, T, Inner> ParallelSource for Cloned<Inner>
where
    T: Clone + 'a,
    Inner: ParallelSource<Item = &'a T>,
{
    type Item = T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        MapSourceDescriptor {
            inner: self.inner.descriptor(),
            f: T::clone,
        }
    }
}

impl<'a, T, Inner> ExactParallelSource for Cloned<Inner>
where
    T: Clone + 'a,
    Inner: ExactParallelSource<Item = &'a T>,
{
    type Item = T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        MapSourceDescriptor {
            inner: self.inner.exact_descriptor(),
            f: T::clone,
        }
    }
}

/// This struct is created by the [`copied()`](super::ParallelSourceExt::copied)
/// method on [`ParallelSourceExt`](super::ParallelSourceExt) and
/// [`copied()`](super::ExactParallelSourceExt::copied) on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the
/// [`ParallelSource`]/[`ParallelSourceExt`](super::ParallelSourceExt) or
/// [`ExactParallelSource`]/
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Copied<Inner> {
    pub(super) inner: Inner,
}

impl<'a, T, Inner> ParallelSource for Copied<Inner>
where
    T: Copy + 'a,
    Inner: ParallelSource<Item = &'a T>,
{
    type Item = T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        MapSourceDescriptor {
            inner: self.inner.descriptor(),
            f: T::clone,
        }
    }
}

impl<'a, T, Inner> ExactParallelSource for Copied<Inner>
where
    T: Copy + 'a,
    Inner: ExactParallelSource<Item = &'a T>,
{
    type Item = T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        MapSourceDescriptor {
            inner: self.inner.exact_descriptor(),
            f: T::clone,
        }
    }
}

/// This struct is created by the
/// [`enumerate()`](super::ExactParallelSourceExt::enumerate) method on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSource`] and
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Enumerate<Inner> {
    pub(super) inner: Inner,
}

impl<Inner: ExactParallelSource> ExactParallelSource for Enumerate<Inner> {
    type Item = (usize, Inner::Item);

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        EnumerateSourceDescriptor {
            inner: self.inner.exact_descriptor(),
        }
    }
}

struct EnumerateSourceDescriptor<Inner> {
    inner: Inner,
}

impl<Inner: SourceCleanup> SourceCleanup for EnumerateSourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    fn len(&self) -> usize {
        self.inner.len()
    }

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // SAFETY: The `EnumerateSourceDescriptor` only implements a mapping of items,
            // while passing through indices to the inner descriptor.
            //
            // Therefore:
            // - if the caller passes ranges included in `0..len`, ranges passed to the
            //   inner `cleanup_item_range()` function are also included in the `0..len`
            //   range,
            // - if the caller doesn't repeat indices, the enumerate adaptor doesn't repeat
            //   indices passed to the inner descriptor.
            unsafe {
                self.inner.cleanup_item_range(range);
            }
        }
    }
}

impl<Inner: ExactSourceDescriptor> ExactSourceDescriptor for EnumerateSourceDescriptor<Inner> {
    type Item = (usize, Inner::Item);

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        // SAFETY: The `EnumerateSourceDescriptor` only implements a mapping of items,
        // while passing through indices to the inner descriptor.
        //
        // Therefore:
        // - if the caller passes indices in `0..len`, indices passed to the inner
        //   `exact_fetch_item()` function are also in the `0..len` range,
        // - if the caller doesn't repeat indices, the enumerate adaptor doesn't repeat
        //   indices passed to the inner descriptor.
        (index, unsafe { self.inner.exact_fetch_item(index) })
    }
}

/// This struct is created by the [`filter()`](super::ParallelSourceExt::filter)
/// method on [`ParallelSourceExt`](super::ParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Filter<Inner, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner, F> ParallelSource for Filter<Inner, F>
where
    Inner: ParallelSource,
    F: Fn(&Inner::Item) -> bool + Sync,
{
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        FilterMapSourceDescriptor {
            inner: self.inner.descriptor(),
            f: move |item| if (self.f)(&item) { Some(item) } else { None },
        }
    }
}

/// This struct is created by the
/// [`filter()`](super::ExactParallelSourceExt::filter) method on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct FilterExact<Inner, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner, F> ParallelSource for FilterExact<Inner, F>
where
    Inner: ExactParallelSource,
    F: Fn(&Inner::Item) -> bool + Sync,
{
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        FilterMapExactSourceDescriptor {
            inner: self.inner.exact_descriptor(),
            f: move |item| if (self.f)(&item) { Some(item) } else { None },
        }
    }
}

/// This struct is created by the
/// [`filter_map()`](super::ParallelSourceExt::filter_map) method on
/// [`ParallelSourceExt`](super::ParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct FilterMap<Inner, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner, T, F> ParallelSource for FilterMap<Inner, F>
where
    Inner: ParallelSource,
    F: Fn(Inner::Item) -> Option<T> + Sync,
{
    type Item = T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        FilterMapSourceDescriptor {
            inner: self.inner.descriptor(),
            f: self.f,
        }
    }
}

/// This struct is created by the
/// [`filter_map()`](super::ExactParallelSourceExt::filter_map) method on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct FilterMapExact<Inner, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner, T, F> ParallelSource for FilterMapExact<Inner, F>
where
    Inner: ExactParallelSource,
    F: Fn(Inner::Item) -> Option<T> + Sync,
{
    type Item = T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        FilterMapExactSourceDescriptor {
            inner: self.inner.exact_descriptor(),
            f: self.f,
        }
    }
}

struct FilterMapSourceDescriptor<Inner, F> {
    inner: Inner,
    f: F,
}

impl<Inner: SourceCleanup, F> SourceCleanup for FilterMapSourceDescriptor<Inner, F> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    fn len(&self) -> usize {
        self.inner.len()
    }

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // SAFETY: This descriptor implements a pass-through of indices to the inner
            // descriptor, therefore safety is preserved by induction.
            unsafe {
                self.inner.cleanup_item_range(range);
            }
        }
    }
}

impl<Inner, T, F> SourceDescriptor for FilterMapSourceDescriptor<Inner, F>
where
    Inner: SourceDescriptor,
    F: Fn(Inner::Item) -> Option<T> + Sync,
{
    type Item = T;

    unsafe fn fetch_item(&self, index: usize) -> Option<Self::Item> {
        // SAFETY: This descriptor implements a pass-through of indices to the inner
        // descriptor, therefore safety is preserved by induction.
        let item = unsafe { self.inner.fetch_item(index) };
        item.and_then(&self.f)
    }
}

struct FilterMapExactSourceDescriptor<Inner, F> {
    inner: Inner,
    f: F,
}

impl<Inner: SourceCleanup, F> SourceCleanup for FilterMapExactSourceDescriptor<Inner, F> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    fn len(&self) -> usize {
        self.inner.len()
    }

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // SAFETY: This descriptor implements a pass-through of indices to the inner
            // descriptor, therefore safety is preserved by induction.
            unsafe {
                self.inner.cleanup_item_range(range);
            }
        }
    }
}

impl<Inner, T, F> SourceDescriptor for FilterMapExactSourceDescriptor<Inner, F>
where
    Inner: ExactSourceDescriptor,
    F: Fn(Inner::Item) -> Option<T> + Sync,
{
    type Item = T;

    unsafe fn fetch_item(&self, index: usize) -> Option<Self::Item> {
        // SAFETY: This descriptor implements a pass-through of indices to the inner
        // descriptor, therefore safety is preserved by induction.
        let item = unsafe { self.inner.exact_fetch_item(index) };
        (self.f)(item)
    }
}

/// This struct is created by the
/// [`inspect()`](super::ParallelSourceExt::inspect) method on
/// [`ParallelSourceExt`](super::ParallelSourceExt) and
/// [`inspect()`](super::ExactParallelSourceExt::inspect) on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the
/// [`ParallelSource`]/[`ParallelSourceExt`](super::ParallelSourceExt) or
/// [`ExactParallelSource`]/
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Inspect<Inner, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner, F> ParallelSource for Inspect<Inner, F>
where
    Inner: ParallelSource,
    F: Fn(&Inner::Item) + Sync,
{
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        MapSourceDescriptor {
            inner: self.inner.descriptor(),
            f: move |item| {
                (self.f)(&item);
                item
            },
        }
    }
}

impl<Inner, F> ExactParallelSource for Inspect<Inner, F>
where
    Inner: ExactParallelSource,
    F: Fn(&Inner::Item) + Sync,
{
    type Item = Inner::Item;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        MapSourceDescriptor {
            inner: self.inner.exact_descriptor(),
            f: move |item| {
                (self.f)(&item);
                item
            },
        }
    }
}

/// This struct is created by the [`map()`](super::ParallelSourceExt::map)
/// method on [`ParallelSourceExt`](super::ParallelSourceExt) and
/// [`map()`](super::ExactParallelSourceExt::map) on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the
/// [`ParallelSource`]/[`ParallelSourceExt`](super::ParallelSourceExt) or
/// [`ExactParallelSource`]/
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Map<Inner, F> {
    pub(super) inner: Inner,
    pub(super) f: F,
}

impl<Inner, T, F> ParallelSource for Map<Inner, F>
where
    Inner: ParallelSource,
    F: Fn(Inner::Item) -> T + Sync,
{
    type Item = T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        MapSourceDescriptor {
            inner: self.inner.descriptor(),
            f: self.f,
        }
    }
}

impl<Inner, T, F> ExactParallelSource for Map<Inner, F>
where
    Inner: ExactParallelSource,
    F: Fn(Inner::Item) -> T + Sync,
{
    type Item = T;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        MapSourceDescriptor {
            inner: self.inner.exact_descriptor(),
            f: self.f,
        }
    }
}

struct MapSourceDescriptor<Inner, F> {
    inner: Inner,
    f: F,
}

impl<Inner: SourceCleanup, F> SourceCleanup for MapSourceDescriptor<Inner, F> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    fn len(&self) -> usize {
        self.inner.len()
    }

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            // SAFETY: This descriptor implements a pass-through of indices to the inner
            // descriptor, therefore safety is preserved by induction.
            unsafe {
                self.inner.cleanup_item_range(range);
            }
        }
    }
}

impl<Inner, T, F> SourceDescriptor for MapSourceDescriptor<Inner, F>
where
    Inner: SourceDescriptor,
    F: Fn(Inner::Item) -> T + Sync,
{
    type Item = T;

    unsafe fn fetch_item(&self, index: usize) -> Option<Self::Item> {
        // SAFETY: This descriptor implements a pass-through of indices to the inner
        // descriptor, therefore safety is preserved by induction.
        let item = unsafe { self.inner.fetch_item(index) };
        item.map(&self.f)
    }
}

impl<Inner, T, F> ExactSourceDescriptor for MapSourceDescriptor<Inner, F>
where
    Inner: ExactSourceDescriptor,
    F: Fn(Inner::Item) -> T + Sync,
{
    type Item = T;

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        // SAFETY: This descriptor implements a pass-through of indices to the inner
        // descriptor, therefore safety is preserved by induction.
        let item = unsafe { self.inner.exact_fetch_item(index) };
        (self.f)(item)
    }
}

/// This struct is created by the [`rev()`](super::ParallelSourceExt::rev)
/// method on [`ParallelSourceExt`](super::ParallelSourceExt) and
/// [`rev()`](super::ExactParallelSourceExt::rev) on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the
/// [`ParallelSource`]/[`ParallelSourceExt`](super::ParallelSourceExt) or
/// [`ExactParallelSource`]/
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Rev<Inner> {
    pub(super) inner: Inner,
}

impl<Inner: ParallelSource> ParallelSource for Rev<Inner> {
    type Item = Inner::Item;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.descriptor();
        let len = descriptor.len();
        RevSourceDescriptor {
            inner: descriptor,
            len,
        }
    }
}

impl<Inner: ExactParallelSource> ExactParallelSource for Rev<Inner> {
    type Item = Inner::Item;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.exact_descriptor();
        let len = descriptor.len();
        RevSourceDescriptor {
            inner: descriptor,
            len,
        }
    }
}

struct RevSourceDescriptor<Inner> {
    inner: Inner,
    len: usize,
}

impl<Inner: SourceCleanup> SourceCleanup for RevSourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= self.len);
            debug_assert!(range.end <= self.len);
            // SAFETY: Given an inner descriptor of length `len`, the `RevSourceDescriptor`
            // implements a bijective mapping of indices from `0..len` to `0..len` given by
            // `rev: x -> len - 1 - x`.
            //
            // Therefore:
            // - if the caller passes ranges included in `0..len`, ranges passed to the
            //   inner `cleanup_item_range()` function are also included in the `0..len`
            //   range,
            // - if the caller doesn't repeat indices, the rev adaptor doesn't repeat
            //   indices passed to the inner descriptor.
            //
            // Given an open-ended input range `start..end` = `start..=end - 1`, the mapped
            // range is `rev(end - 1)..rev(start) + 1` = `len - end..len - start`.
            unsafe {
                self.inner
                    .cleanup_item_range(self.len - range.end..self.len - range.start);
            }
        }
    }
}

impl<Inner: SourceDescriptor> SourceDescriptor for RevSourceDescriptor<Inner> {
    type Item = Inner::Item;

    unsafe fn fetch_item(&self, index: usize) -> Option<Self::Item> {
        debug_assert!(index < self.len);
        // SAFETY: Given an inner descriptor of length `len`, the `RevSourceDescriptor`
        // implements a bijective mapping of indices from `0..len` to `0..len` given by
        // `rev: x -> len - 1 - x`.
        //
        // Therefore:
        // - if the caller passes indices in `0..len`, indices passed to the inner
        //   `fetch_item()` function are also in the `0..len` range,
        // - if the caller doesn't repeat indices, the rev adaptor doesn't repeat
        //   indices passed to the inner descriptor.
        unsafe { self.inner.fetch_item(self.len - index - 1) }
    }
}

impl<Inner: ExactSourceDescriptor> ExactSourceDescriptor for RevSourceDescriptor<Inner> {
    type Item = Inner::Item;

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        // SAFETY: Given an inner descriptor of length `len`, the `RevSourceDescriptor`
        // implements a bijective mapping of indices from `0..len` to `0..len` given by
        // `rev: x -> len - 1 - x`.
        //
        // Therefore:
        // - if the caller passes indices in `0..len`, indices passed to the inner
        //   `exact_fetch_item()` function are also in the `0..len` range,
        // - if the caller doesn't repeat indices, the rev adaptor doesn't repeat
        //   indices passed to the inner descriptor.
        unsafe { self.inner.exact_fetch_item(self.len - index - 1) }
    }
}

/// This struct is created by the
/// [`skip()`](super::ExactParallelSourceExt::skip) method on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSource`] and
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Skip<Inner> {
    pub(super) inner: Inner,
    pub(super) count: usize,
}

impl<Inner: ExactParallelSource> ExactParallelSource for Skip<Inner> {
    type Item = Inner::Item;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.exact_descriptor();
        let inner_len = descriptor.len();
        let count = std::cmp::min(self.count, inner_len);
        SkipSourceDescriptor {
            inner: descriptor,
            len: inner_len - count,
            count,
        }
    }
}

struct SkipSourceDescriptor<Inner: SourceCleanup> {
    inner: Inner,
    len: usize,
    count: usize,
}

impl<Inner: SourceCleanup> SourceCleanup for SkipSourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    fn len(&self) -> usize {
        self.len
    }

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= self.len);
            debug_assert!(range.end <= self.len);
            // SAFETY: Given an inner descriptor of length `len` as well as a parameter
            // `count <= len`, the `SkipSourceDescriptor` implements a bijective mapping of
            // indices from `0..len - count` to `count..len` given by a translation of
            // `count` places.
            //
            // Therefore:
            // - if the caller passes ranges included in `0..len - count`, ranges passed
            //   here to the inner `cleanup_item_range()` function are included in the
            //   `count..len` range,
            // - if the caller doesn't repeat indices, the skip adaptor doesn't repeat
            //   indices passed to the inner descriptor.
            unsafe {
                self.inner
                    .cleanup_item_range(self.count + range.start..self.count + range.end);
            }
        }
    }
}

impl<Inner: ExactSourceDescriptor> ExactSourceDescriptor for SkipSourceDescriptor<Inner> {
    type Item = Inner::Item;

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        // SAFETY: Given an inner descriptor of length `len` as well as a parameter
        // `count <= len`, the `SkipSourceDescriptor` implements a bijective mapping of
        // indices from `0..len - count` to `count..len` given by a translation of
        // `count` places.
        //
        // Therefore:
        // - if the caller passes indices in `0..len - count`, indices passed here to
        //   the inner `exact_fetch_item()` function are in the `count..len` range,
        // - if the caller doesn't repeat indices, the skip adaptor doesn't repeat
        //   indices passed to the inner descriptor.
        unsafe { self.inner.exact_fetch_item(self.count + index) }
    }
}

impl<Inner: SourceCleanup> Drop for SkipSourceDescriptor<Inner> {
    fn drop(&mut self) {
        if Self::NEEDS_CLEANUP && self.count != 0 {
            // SAFETY: Given an inner descriptor of length `len` as well as a parameter
            // `count <= len`, the `SkipSourceDescriptor` implements a bijective mapping of
            // indices from `0..len - count` to `count..len` given by a translation of
            // `count` places.
            //
            // Therefore:
            // - the range `0..count` is included in the inner range `0..len`,
            // - the items in `0..count` aren't passed to the inner descriptor other than in
            //   this drop implementation.
            unsafe {
                self.inner.cleanup_item_range(0..self.count);
            }
        }
    }
}

/// This struct is created by the
/// [`skip_exact()`](super::ExactParallelSourceExt::skip_exact) method on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSource`] and
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct SkipExact<Inner> {
    pub(super) inner: Inner,
    pub(super) count: usize,
}

impl<Inner: ExactParallelSource> ExactParallelSource for SkipExact<Inner> {
    type Item = Inner::Item;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.exact_descriptor();
        let inner_len = descriptor.len();
        assert!(
            self.count <= inner_len,
            "called skip_exact() with more items than this source produces"
        );
        SkipSourceDescriptor {
            inner: descriptor,
            len: inner_len - self.count,
            count: self.count,
        }
    }
}

/// This struct is created by the
/// [`step_by()`](super::ExactParallelSourceExt::step_by) method on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSource`] and
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct StepBy<Inner> {
    pub(super) inner: Inner,
    pub(super) step: usize,
}

impl<Inner: ExactParallelSource> ExactParallelSource for StepBy<Inner> {
    type Item = Inner::Item;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.exact_descriptor();
        let inner_len = descriptor.len();
        assert!(self.step != 0, "called step_by() with a step of zero");
        let len = inner_len.div_ceil(self.step);
        StepBySourceDescriptor {
            inner: descriptor,
            len,
            step: self.step,
            inner_len,
        }
    }
}

struct StepBySourceDescriptor<Inner: SourceCleanup> {
    inner: Inner,
    len: usize,
    step: usize,
    inner_len: usize,
}

impl<Inner: SourceCleanup> SourceCleanup for StepBySourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    fn len(&self) -> usize {
        self.len
    }

    // For safety comments: given an inner descriptor of length `len` as well as a
    // parameter `step != 0`, if we set `len' := ceil(len / step)` the
    // `StepBySourceDescriptor` implements a bijective mapping between `0..len'` and
    // `{0, step, 2*step, ..., (len' - 1)*step}` given by `f: x -> x * step`.
    //
    // Therefore:
    // - if the caller passes indices included in `0..len'`, indices passed to the
    //   inner descriptor are included in the `0..=(len' - 1) * step` range, itself
    //   included in `0..len`,
    // - if the caller doesn't repeat indices, the step-by adaptor doesn't repeat
    //   indices passed to the inner descriptor.
    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= self.len);
            debug_assert!(range.end <= self.len);
            if self.step == 1 {
                // SAFETY: See the function comment. When the step is 1 the mapping is the
                // identity so we just pass the range through.
                unsafe {
                    self.inner.cleanup_item_range(range);
                }
            } else {
                for i in range {
                    // SAFETY: See the function comment. This call with a length-one range cleans up
                    // the item at index `step * i`. The other items are cleaned up in the drop
                    // implementation.
                    unsafe {
                        self.inner
                            .cleanup_item_range(self.step * i..self.step * i + 1);
                    }
                }
            }
        }
    }
}

impl<Inner: ExactSourceDescriptor> ExactSourceDescriptor for StepBySourceDescriptor<Inner> {
    type Item = Inner::Item;

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.len);
        // SAFETY: See the function comment in `Self::cleanup_item_range`. This
        // implements the mapping `i -> step * i`.
        unsafe { self.inner.exact_fetch_item(self.step * index) }
    }
}

impl<Inner: SourceCleanup> Drop for StepBySourceDescriptor<Inner> {
    // For safety comments: see the function comment in `Self::cleanup_item_range`.
    // This drop implementation is the only one to invoke items that aren't
    // multiples of `step`.
    fn drop(&mut self) {
        if Self::NEEDS_CLEANUP && self.step != 1 {
            let full_blocks = self.inner_len / self.step;
            for i in 0..full_blocks {
                // SAFETY: See the function comment. This line cleans up the items that aren't
                // multiples of the `step` in the `step * i..step * (i + 1)` range.
                unsafe {
                    self.inner
                        .cleanup_item_range(self.step * i + 1..self.step * (i + 1));
                }
            }
            let last_block = self.step * full_blocks;
            // This implements the comparison `last_block + 1 < inner_len` without risk of
            // overflow.
            if self.inner_len - last_block > 1 {
                // SAFETY: See the function comment. This line cleans up the items that aren't
                // multiples of the `step` beyond `step * len'`.
                unsafe {
                    self.inner
                        .cleanup_item_range(last_block + 1..self.inner_len);
                }
            }
        }
    }
}

/// This struct is created by the
/// [`take()`](super::ExactParallelSourceExt::take) method on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSource`] and
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct Take<Inner> {
    pub(super) inner: Inner,
    pub(super) count: usize,
}

impl<Inner: ExactParallelSource> ExactParallelSource for Take<Inner> {
    type Item = Inner::Item;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.exact_descriptor();
        let inner_len = descriptor.len();
        let count = std::cmp::min(self.count, inner_len);
        TakeSourceDescriptor {
            inner: descriptor,
            count,
            inner_len,
        }
    }
}

struct TakeSourceDescriptor<Inner: SourceCleanup> {
    inner: Inner,
    count: usize,
    inner_len: usize,
}

impl<Inner: SourceCleanup> SourceCleanup for TakeSourceDescriptor<Inner> {
    const NEEDS_CLEANUP: bool = Inner::NEEDS_CLEANUP;

    fn len(&self) -> usize {
        self.count
    }

    unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            debug_assert!(range.start <= range.end);
            debug_assert!(range.start <= self.count);
            debug_assert!(range.end <= self.count);
            // SAFETY: Given an inner descriptor of length `len` as well as a parameter
            // `count <= len`, the `TakeSourceDescriptor` implements a pass-through mapping
            // of indices from `0..count` to `0..count`.
            //
            // Therefore:
            // - if the caller passes ranges included in `0..count`, ranges passed here to
            //   the inner `cleanup_item_range()` function are included in the `0..count`
            //   range (itself included in `0..len`),
            // - if the caller doesn't repeat indices, the take adaptor doesn't repeat
            //   indices passed to the inner descriptor.
            unsafe {
                self.inner.cleanup_item_range(range);
            }
        }
    }
}

impl<Inner: ExactSourceDescriptor> ExactSourceDescriptor for TakeSourceDescriptor<Inner> {
    type Item = Inner::Item;

    unsafe fn exact_fetch_item(&self, index: usize) -> Self::Item {
        debug_assert!(index < self.count);
        // SAFETY: Given an inner descriptor of length `len` as well as a parameter
        // `count <= len`, the `TakeSourceDescriptor` implements a pass-through mapping
        // of indices from `0..count` to `0..count`.
        //
        // Therefore:
        // - if the caller passes indices in `0..count`, indices passed here to the
        //   inner `exact_fetch_item()` function are in the `0..count` range (itself
        //   included in `0..len`),
        // - if the caller doesn't repeat indices, the take adaptor doesn't repeat
        //   indices passed to the inner descriptor.
        unsafe { self.inner.exact_fetch_item(index) }
    }
}

impl<Inner: SourceCleanup> Drop for TakeSourceDescriptor<Inner> {
    fn drop(&mut self) {
        if Self::NEEDS_CLEANUP && self.count != self.inner_len {
            // SAFETY: Given an inner descriptor of length `len` as well as a parameter
            // `count <= len`, the `TakeSourceDescriptor` implements a pass-through mapping
            // of indices from `0..count` to `0..count`.
            //
            // Therefore:
            // - the range `count..len` is included in the inner range `0..len`,
            // - the items in `count..len` aren't passed to the inner descriptor other than
            //   in this drop implementation.
            unsafe {
                self.inner.cleanup_item_range(self.count..self.inner_len);
            }
        }
    }
}

/// This struct is created by the
/// [`take_exact()`](super::ExactParallelSourceExt::take_exact) method on
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt).
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ExactParallelSource`] and
/// [`ExactParallelSourceExt`](super::ExactParallelSourceExt) traits, but it is
/// nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct TakeExact<Inner> {
    pub(super) inner: Inner,
    pub(super) count: usize,
}

impl<Inner: ExactParallelSource> ExactParallelSource for TakeExact<Inner> {
    type Item = Inner::Item;

    fn exact_descriptor(self) -> impl ExactSourceDescriptor<Item = Self::Item> + Sync {
        let descriptor = self.inner.exact_descriptor();
        let inner_len = descriptor.len();
        assert!(
            self.count <= inner_len,
            "called take_exact() with more items than this source produces"
        );
        TakeSourceDescriptor {
            inner: descriptor,
            count: self.count,
            inner_len,
        }
    }
}
