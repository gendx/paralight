// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::{IntoParallelSource, ParallelSource, SourceDescriptor};

/// A parallel source over a [slice](slice). This struct is created by the
/// [`par_iter()`](super::IntoParallelRefSource::par_iter) method on
/// [`IntoParallelRefSource`](super::IntoParallelRefSource).
#[must_use = "iterator adaptors are lazy"]
pub struct SliceParallelSource<'data, T> {
    slice: &'data [T],
}

impl<'data, T: Sync> IntoParallelSource for &'data [T] {
    type Item = &'data T;
    type Source = SliceParallelSource<'data, T>;

    fn into_par_iter(self) -> Self::Source {
        SliceParallelSource { slice: self }
    }
}

impl<'data, T: Sync> ParallelSource for SliceParallelSource<'data, T> {
    type Item = &'data T;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        SourceDescriptor {
            len: self.slice.len(),
            fetch_item: |index| &self.slice[index],
        }
    }
}

#[allow(clippy::too_long_first_doc_paragraph)]
/// A parallel source over a [mutable slice](slice). This struct is created by
/// the [`par_iter_mut()`](super::IntoParallelRefMutSource::par_iter_mut) method
/// on [`IntoParallelRefMutSource`](super::IntoParallelRefMutSource).
#[must_use = "iterator adaptors are lazy"]
pub struct MutSliceParallelSource<'data, T> {
    slice: &'data mut [T],
}

impl<'data, T: Send> IntoParallelSource for &'data mut [T] {
    type Item = &'data mut T;
    type Source = MutSliceParallelSource<'data, T>;

    fn into_par_iter(self) -> Self::Source {
        MutSliceParallelSource { slice: self }
    }
}

impl<'data, T: Send> ParallelSource for MutSliceParallelSource<'data, T> {
    type Item = &'data mut T;

    fn descriptor(self) -> SourceDescriptor<Self::Item, impl Fn(usize) -> Self::Item + Sync> {
        let len = self.slice.len();
        let ptr = MutPtrWrapper(self.slice.as_mut_ptr());
        SourceDescriptor {
            len,
            fetch_item: move |index| {
                assert!(index < len);
                let base_ptr: *mut T = ptr.get();
                // SAFETY:
                // - The offset in bytes `index * size_of::<T>()` fits in an `isize`, because
                //   the index is smaller than the length of the (well-formed) input slice. This
                //   is ensured by the thread pool's `pipeline()` function (which yields indices
                //   in the range `0..len`), and further confirmed by the assertion.
                // - The `base_ptr` is derived from an allocated object (the input slice), and
                //   the entire range between `base_ptr` and the resulting `item_ptr` is in
                //   bounds of that allocated object. This is because the index is smaller than
                //   the length of the input slice.
                let item_ptr: *mut T = unsafe { base_ptr.add(index) };
                // SAFETY:
                //
                // From https://doc.rust-lang.org/std/ptr/index.html#pointer-to-reference-conversion:
                // - The `item_ptr` is properly aligned, as it is constructed by calling `add()`
                //   on the aligned `base_ptr`.
                // - The `item_ptr` is not null, as the `base_ptr` isn't null (obtained from a
                //   well-formed non-empty slice) and the `index` is in bounds of the slice (no
                //   wrap around).
                // - The `item_ptr` is dereferenceable, as the whole memory range of length
                //   `size_of::<T>()` starting from it is within bounds of a single allocated
                //   object (the input slice).
                // - The `item_ptr` points to a valid value of type `T`, the element from the
                //   input slice at position `index`.
                // - The `item_ptr` follows the aliasing rules: while this reference exists
                //   (within this scope and in particular during the call to `process_item()`),
                //   the memory it points to isn't accessed through any other pointer or
                //   reference. This is ensured because the thread pool's `pipeline()` function
                //   yields distinct indices, and because the slice is exclusively owned during
                //   the scope of this `ParallelIterator::pipeline()` function.
                //
                // Lastly, materializing this mutable reference on any thread is sound given
                // that `T` is `Send`. Indeed, this amounts to sending a `&mut T` across
                // threads, and `&mut T` is `Send` if and only if `T` is `Send`. This is the
                // rationale for why `MutPtrWrapper` implements `Sync` when `T` is `Send`.
                let item: &mut T = unsafe { &mut *item_ptr };
                item
            },
        }
    }
}

/// A helper struct for the implementation of [`MutSliceParallelSource`], that
/// wraps a [`*mut T`](pointer). This enables sending [`&mut T`](reference)
/// derived from a [`&mut [T]`](slice) to other threads.
struct MutPtrWrapper<T>(*mut T);
impl<T> MutPtrWrapper<T> {
    fn get(&self) -> *mut T {
        self.0
    }
}

/// SAFETY:
///
/// A [`MutPtrWrapper`] is meant to be shared among threads as a way to send
/// items of type [`&mut T`](reference) to other threads (see the safety
/// comments in [`MutSliceParallelSource::descriptor`]). Therefore we make it
/// [`Sync`] if and only if [`&mut T`](reference) is [`Send`], which is when `T`
/// is [`Send`].
unsafe impl<T: Send> Sync for MutPtrWrapper<T> {}
