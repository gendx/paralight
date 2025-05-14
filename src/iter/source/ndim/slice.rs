// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! TODO

use super::{MultiDimParallelSource, MultiDimSourceDescriptor, SourceCleanup};
use std::marker::PhantomData;

/// TODO
pub struct Slice1dParallelSource<'data, T, const A: usize> {
    /// TODO
    pub slice: &'data [T; A],
}

/// TODO
pub struct Slice2dParallelSource<'data, T, const A: usize, const B: usize> {
    /// TODO
    pub slice: &'data [[T; B]; A],
}

/// TODO
pub struct Slice3dParallelSource<'data, T, const A: usize, const B: usize, const C: usize> {
    /// TODO
    pub slice: &'data [[[T; C]; B]; A],
}

/// TODO
pub struct MutSlice1dParallelSource<'data, T, const A: usize> {
    /// TODO
    pub slice: &'data mut [T; A],
}

/// TODO
pub struct MutSlice2dParallelSource<'data, T, const A: usize, const B: usize> {
    /// TODO
    pub slice: &'data mut [[T; B]; A],
}

/// TODO
pub struct MutSlice3dParallelSource<'data, T, const A: usize, const B: usize, const C: usize> {
    /// TODO
    pub slice: &'data mut [[[T; C]; B]; A],
}

impl<'data, T: Sync, const A: usize> MultiDimParallelSource<1>
    for Slice1dParallelSource<'data, T, A>
{
    type Item = &'data T;

    fn descriptor(self) -> impl MultiDimSourceDescriptor<1, Item = Self::Item> {
        Slice1dSourceDescriptor { slice: self.slice }
    }
}

impl<'data, T: Sync, const A: usize, const B: usize> MultiDimParallelSource<2>
    for Slice2dParallelSource<'data, T, A, B>
{
    type Item = &'data T;

    fn descriptor(self) -> impl MultiDimSourceDescriptor<2, Item = Self::Item> {
        Slice2dSourceDescriptor { slice: self.slice }
    }
}

impl<'data, T: Sync, const A: usize, const B: usize, const C: usize> MultiDimParallelSource<3>
    for Slice3dParallelSource<'data, T, A, B, C>
{
    type Item = &'data T;

    fn descriptor(self) -> impl MultiDimSourceDescriptor<3, Item = Self::Item> {
        Slice3dSourceDescriptor { slice: self.slice }
    }
}

struct Slice1dSourceDescriptor<'data, T: Sync, const A: usize> {
    slice: &'data [T; A],
}

impl<T: Sync, const A: usize> SourceCleanup for Slice1dSourceDescriptor<'_, T, A> {
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {}
}

impl<'data, T: Sync, const A: usize> MultiDimSourceDescriptor<1>
    for Slice1dSourceDescriptor<'data, T, A>
{
    type Item = &'data T;

    fn len(&self) -> [usize; 1] {
        [A]
    }

    unsafe fn fetch_item(&self, [i]: [usize; 1]) -> Self::Item {
        &self.slice[i]
    }
}

struct Slice2dSourceDescriptor<'data, T: Sync, const A: usize, const B: usize> {
    slice: &'data [[T; B]; A],
}

impl<T: Sync, const A: usize, const B: usize> SourceCleanup
    for Slice2dSourceDescriptor<'_, T, A, B>
{
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {}
}

impl<'data, T: Sync, const A: usize, const B: usize> MultiDimSourceDescriptor<2>
    for Slice2dSourceDescriptor<'data, T, A, B>
{
    type Item = &'data T;

    fn len(&self) -> [usize; 2] {
        [A, B]
    }

    unsafe fn fetch_item(&self, [i, j]: [usize; 2]) -> Self::Item {
        &self.slice[i][j]
    }
}

struct Slice3dSourceDescriptor<'data, T: Sync, const A: usize, const B: usize, const C: usize> {
    slice: &'data [[[T; C]; B]; A],
}

impl<T: Sync, const A: usize, const B: usize, const C: usize> SourceCleanup
    for Slice3dSourceDescriptor<'_, T, A, B, C>
{
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {}
}

impl<'data, T: Sync, const A: usize, const B: usize, const C: usize> MultiDimSourceDescriptor<3>
    for Slice3dSourceDescriptor<'data, T, A, B, C>
{
    type Item = &'data T;

    fn len(&self) -> [usize; 3] {
        [A, B, C]
    }

    unsafe fn fetch_item(&self, [i, j, k]: [usize; 3]) -> Self::Item {
        &self.slice[i][j][k]
    }
}

impl<'data, T: Send, const A: usize> MultiDimParallelSource<1>
    for MutSlice1dParallelSource<'data, T, A>
{
    type Item = &'data mut T;

    fn descriptor(self) -> impl MultiDimSourceDescriptor<1, Item = Self::Item> {
        let ptr = MutPtrWrapper(self.slice.as_mut_ptr());
        MutSlice1dSourceDescriptor::<_, A> {
            ptr,
            _phantom: PhantomData,
        }
    }
}

impl<'data, T: Send, const A: usize, const B: usize> MultiDimParallelSource<2>
    for MutSlice2dParallelSource<'data, T, A, B>
{
    type Item = &'data mut T;

    fn descriptor(self) -> impl MultiDimSourceDescriptor<2, Item = Self::Item> {
        let ptr = MutPtrWrapper(self.slice.as_mut_ptr() as *mut T);
        MutSlice2dSourceDescriptor::<_, A, B> {
            ptr,
            _phantom: PhantomData,
        }
    }
}

impl<'data, T: Send, const A: usize, const B: usize, const C: usize> MultiDimParallelSource<3>
    for MutSlice3dParallelSource<'data, T, A, B, C>
{
    type Item = &'data mut T;

    fn descriptor(self) -> impl MultiDimSourceDescriptor<3, Item = Self::Item> {
        let ptr = MutPtrWrapper(self.slice.as_mut_ptr() as *mut T);
        MutSlice3dSourceDescriptor::<_, A, B, C> {
            ptr,
            _phantom: PhantomData,
        }
    }
}

struct MutSlice1dSourceDescriptor<'data, T: Send + 'data, const A: usize> {
    ptr: MutPtrWrapper<T>,
    _phantom: PhantomData<&'data ()>,
}

impl<'data, T: Send + 'data, const A: usize> SourceCleanup
    for MutSlice1dSourceDescriptor<'data, T, A>
{
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {}
}

impl<'data, T: Send + 'data, const A: usize> MultiDimSourceDescriptor<1>
    for MutSlice1dSourceDescriptor<'data, T, A>
{
    type Item = &'data mut T;

    fn len(&self) -> [usize; 1] {
        [A]
    }

    unsafe fn fetch_item(&self, [i]: [usize; 1]) -> Self::Item {
        assert!(i < A);
        let base_ptr: *mut T = self.ptr.get();
        // SAFETY: TODO
        let item_ptr: *mut T = unsafe { base_ptr.add(i) };
        // SAFETY: TODO
        let item: &mut T = unsafe { &mut *item_ptr };
        item
    }
}

struct MutSlice2dSourceDescriptor<'data, T: Send + 'data, const A: usize, const B: usize> {
    ptr: MutPtrWrapper<T>,
    _phantom: PhantomData<&'data ()>,
}

impl<'data, T: Send + 'data, const A: usize, const B: usize> SourceCleanup
    for MutSlice2dSourceDescriptor<'data, T, A, B>
{
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {}
}

impl<'data, T: Send + 'data, const A: usize, const B: usize> MultiDimSourceDescriptor<2>
    for MutSlice2dSourceDescriptor<'data, T, A, B>
{
    type Item = &'data mut T;

    fn len(&self) -> [usize; 2] {
        [A, B]
    }

    unsafe fn fetch_item(&self, [i, j]: [usize; 2]) -> Self::Item {
        assert!(i < A && j < B);
        let base_ptr: *mut T = self.ptr.get();
        // SAFETY: TODO
        let item_ptr: *mut T = unsafe { base_ptr.add(i * B + j) };
        // SAFETY: TODO
        let item: &mut T = unsafe { &mut *item_ptr };
        item
    }
}

struct MutSlice3dSourceDescriptor<
    'data,
    T: Send + 'data,
    const A: usize,
    const B: usize,
    const C: usize,
> {
    ptr: MutPtrWrapper<T>,
    _phantom: PhantomData<&'data ()>,
}

impl<'data, T: Send + 'data, const A: usize, const B: usize, const C: usize> SourceCleanup
    for MutSlice3dSourceDescriptor<'data, T, A, B, C>
{
    const NEEDS_CLEANUP: bool = false;

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {}
}

impl<'data, T: Send + 'data, const A: usize, const B: usize, const C: usize>
    MultiDimSourceDescriptor<3> for MutSlice3dSourceDescriptor<'data, T, A, B, C>
{
    type Item = &'data mut T;

    fn len(&self) -> [usize; 3] {
        [A, B, C]
    }

    unsafe fn fetch_item(&self, [i, j, k]: [usize; 3]) -> Self::Item {
        assert!(i < A && j < B && k < C);
        let base_ptr: *mut T = self.ptr.get();
        // SAFETY: TODO
        let item_ptr: *mut T = unsafe { base_ptr.add((i * B + j) * C + k) };
        // SAFETY: TODO
        let item: &mut T = unsafe { &mut *item_ptr };
        item
    }
}

/// TODO
struct MutPtrWrapper<T>(*mut T);
impl<T> MutPtrWrapper<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

/// SAFETY:
///
/// TODO
unsafe impl<T: Send> Sync for MutPtrWrapper<T> {}
