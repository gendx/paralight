// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{IntoParallelSource, ParallelSource, SourceCleanup, SourceDescriptor};
use std::mem::ManuallyDrop;

/// A parallel source over an [array](array). This struct is created by the
/// [`into_par_iter()`](IntoParallelSource::into_par_iter) method on
/// [`IntoParallelSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct ArrayParallelSource<T, const N: usize> {
    array: [T; N],
}

impl<T: Send, const N: usize> IntoParallelSource for [T; N] {
    type Item = T;
    type Source = ArrayParallelSource<T, N>;

    fn into_par_iter(self) -> Self::Source {
        ArrayParallelSource { array: self }
    }
}

impl<T: Send, const N: usize> ParallelSource for ArrayParallelSource<T, N> {
    type Item = T;

    fn descriptor(
        self,
    ) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync, impl SourceCleanup + Sync>
    {
        let mut array = ManuallyDrop::new(self.array);
        let mut_ptr = array.as_mut_ptr();
        let ptr = PtrWrapper(mut_ptr as *const T);
        SourceDescriptor {
            len: N,
            fetch_item: move |index| {
                assert!(index < N);
                let base_ptr: *const T = ptr.get();
                // SAFETY:
                // - The offset in bytes `index * size_of::<T>()` fits in an `isize`, because
                //   the index is smaller than the length of the (well-formed) input array. This
                //   is ensured by the thread pool's `pipeline()` function (which yields indices
                //   in the range `0..N`), and further confirmed by the assertion.
                // - The `base_ptr` is derived from an allocated object (the input array), and
                //   the entire range between `base_ptr` and the resulting `item_ptr` is in
                //   bounds of that allocated object. This is because the index is smaller than
                //   the length of the input array.
                let item_ptr: *const T = unsafe { base_ptr.add(index) };
                // SAFETY:
                // - The `item_ptr` is properly aligned, as it is constructed by calling `add()`
                //   on the aligned `base_ptr`.
                // - The `item_ptr` points to a properly initialized value of type `T`, the
                //   element from the input array at position `index`.
                // - The `item_ptr` is valid for reads. This is ensured by the thread pool's
                //   `pipeline()` function (which yields distinct indices in the range `0..N`),
                //   i.e. this item hasn't been read (and moved out of the array) yet.
                //   Additionally, there are no concurrent writes to this slot in the array.
                let item: T = unsafe { std::ptr::read(item_ptr) };
                item
            },
            cleanup: OwnedSliceSourceCleanup {
                ptr: MutPtrWrapper(mut_ptr),
            },
        }
    }
}

struct OwnedSliceSourceCleanup<T> {
    ptr: MutPtrWrapper<T>,
}

impl<T> SourceCleanup for OwnedSliceSourceCleanup<T> {
    const NEEDS_CLEANUP: bool = std::mem::needs_drop::<T>();

    fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        if Self::NEEDS_CLEANUP {
            let base_ptr: *mut T = self.ptr.get();
            // SAFETY:
            // - The offset in bytes `range.start * size_of::<T>()` fits in an `isize`,
            //   because the range is included in the length of the (well-formed) input
            //   array. This is ensured by the thread pool's `pipeline()` function (which
            //   only yields in-bound ranges for cleanup).
            // - The `base_ptr` is derived from an allocated object (the input array), and
            //   the entire range between `base_ptr` and the resulting `start_ptr` is in
            //   bounds of that allocated object. This is because the range start is smaller
            //   than the length of the input array.
            let start_ptr: *mut T = unsafe { base_ptr.add(range.start) };
            let slice: *mut [T] =
                std::ptr::slice_from_raw_parts_mut(start_ptr, range.end - range.start);
            // SAFETY:
            // - The `slice` is properly aligned, as it is constructed by calling `add()` on
            //   the aligned `base_ptr`.
            // - The `slice` isn't null, as it is constructed by calling `add()` on the
            //   non-null `base_ptr`.
            // - The `slice` is valid for reads and writes. This is ensured by the thread
            //   pool's `pipeline()` function, which yields non-overlapping indices and
            //   cleanup ranges. I.e. the range of items in this slice isn't accessed by
            //   anything else.
            // - The `slice` is valid for dropping, as it is a part of the input array that
            //   nothing else accesses.
            // - Nothing else is accessing the `slice` while `drop_in_place` is executing.
            //
            // The `slice` is never of size zero, but the above properties (aligned,
            // non-null, etc.) would still hold if it was.
            unsafe { std::ptr::drop_in_place(slice) };
        }
    }
}

/// A parallel source over a [`Vec`]. This struct is created by the
/// [`into_par_iter()`](IntoParallelSource::into_par_iter) method on
/// [`IntoParallelSource`].
///
/// You most likely won't need to interact with this struct directly, as it
/// implements the [`ParallelSource`] and
/// [`ParallelSourceExt`](super::ParallelSourceExt) traits, but it
/// is nonetheless public because of the `must_use` annotation.
#[must_use = "iterator adaptors are lazy"]
pub struct VecParallelSource<T> {
    vec: Vec<T>,
}

impl<T: Send> IntoParallelSource for Vec<T> {
    type Item = T;
    type Source = VecParallelSource<T>;

    fn into_par_iter(self) -> Self::Source {
        VecParallelSource { vec: self }
    }
}

impl<T: Send> IntoParallelSource for Box<[T]> {
    type Item = T;
    type Source = VecParallelSource<T>;

    fn into_par_iter(self) -> Self::Source {
        // There's no Box::<[T]>::from_raw_parts(), so we just piggy back on Vec.
        VecParallelSource {
            vec: self.into_vec(),
        }
    }
}

impl<T: Send> ParallelSource for VecParallelSource<T> {
    type Item = T;

    fn descriptor(
        self,
    ) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync, impl SourceCleanup + Sync>
    {
        let mut vec = ManuallyDrop::new(self.vec);
        let mut_ptr = vec.as_mut_ptr();
        let len = vec.len();
        let capacity = vec.capacity();
        let ptr = PtrWrapper(mut_ptr as *const T);
        SourceDescriptor {
            len,
            fetch_item: move |index| {
                assert!(index < len);
                let base_ptr: *const T = ptr.get();
                // SAFETY:
                // - The offset in bytes `index * size_of::<T>()` fits in an `isize`, because
                //   the index is smaller than the length of the (well-formed) input vector.
                //   This is ensured by the thread pool's `pipeline()` function (which yields
                //   indices in the range `0..len`), and further confirmed by the assertion.
                // - The `base_ptr` is derived from an allocated object (the input vector), and
                //   the entire range between `base_ptr` and the resulting `item_ptr` is in
                //   bounds of that allocated object. This is because the index is smaller than
                //   the length of the input vector.
                let item_ptr: *const T = unsafe { base_ptr.add(index) };
                // SAFETY:
                // - The `item_ptr` is properly aligned, as it is constructed by calling `add()`
                //   on the aligned `base_ptr`.
                // - The `item_ptr` points to a properly initialized value of type `T`, the
                //   element from the input vector at position `index`.
                // - The `item_ptr` is valid for reads. This is ensured by the thread pool's
                //   `pipeline()` function (which yields distinct indices in the range
                //   `0..len`), i.e. this item hasn't been read (and moved out of the vector)
                //   yet. Additionally, there are no concurrent writes to this slot in the
                //   vector.
                let item: T = unsafe { std::ptr::read(item_ptr) };
                item
            },
            cleanup: VecSourceCleanup {
                slice: OwnedSliceSourceCleanup {
                    ptr: MutPtrWrapper(mut_ptr),
                },
                capacity,
            },
        }
    }
}

struct VecSourceCleanup<T> {
    slice: OwnedSliceSourceCleanup<T>,
    capacity: usize,
}

impl<T> Drop for VecSourceCleanup<T> {
    fn drop(&mut self) {
        let base_ptr: *mut T = self.slice.ptr.get();
        // SAFETY:
        // - The `base_ptr` has been allocated with the global allocator, as it is
        //   derived from the source vector.
        // - `T` has the same alignement as what `base_ptr` was allocated with, because
        //   `base_ptr` derives from a vector of `T`s.
        // - `T * capacity` is the size of what `base_ptr` was allocated with, because
        //   that's the capacity of the source vector.
        // - `length <= capacity` because the `length` is set to zero here.
        // - The first `length` values are properly initialized values of type `T`
        //   because the `length` is set to zero.
        // - The allocated size in bytes isn't larger than `isize::MAX`, because that's
        //   derived from the source vector.
        let vec: Vec<T> = unsafe { Vec::from_raw_parts(base_ptr, 0, self.capacity) };
        drop(vec);
    }
}

impl<T> SourceCleanup for VecSourceCleanup<T> {
    const NEEDS_CLEANUP: bool = OwnedSliceSourceCleanup::<T>::NEEDS_CLEANUP;

    fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
        self.slice.cleanup_item_range(range);
    }
}

/// A helper struct for the implementation of [`OwnedSliceSourceCleanup`], that
/// wraps a [`*mut T`](pointer). This enables sending [`&mut [T]`](slice) to
/// other threads.
struct MutPtrWrapper<T>(*mut T);
impl<T> MutPtrWrapper<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

/// SAFETY:
///
/// A [`MutPtrWrapper`] is meant to be shared among threads as a way to send
/// items of type [`&mut [T]`](slice) to other threads (see the safety
/// comments in [`OwnedSliceSourceCleanup::cleanup_item_range`]). Therefore we
/// make it [`Sync`] if and only if [`&mut [T]`](slice) is [`Send`], which is
/// when `T` is [`Send`].
unsafe impl<T: Send> Sync for MutPtrWrapper<T> {}

/// A helper struct for the implementation of [`ArrayParallelSource`] and
/// [`VecParallelSource`], that wraps a [`*const T`](pointer). This enables
/// sending `T` to other threads.
struct PtrWrapper<T>(*const T);
impl<T> PtrWrapper<T> {
    fn get(&self) -> *const T {
        self.0
    }
}

/// SAFETY:
///
/// A [`PtrWrapper`] is meant to be shared among threads as a way to send items
/// of type `T` to other threads (see the safety comments in
/// [`ArrayParallelSource::descriptor`] and [`VecParallelSource::descriptor`]).
/// Therefore we make it [`Sync`] if and only if `T` is [`Send`].
unsafe impl<T: Send> Sync for PtrWrapper<T> {}
