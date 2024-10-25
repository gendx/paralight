// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use super::ParallelIterator;
use crate::ThreadPool;

/// Trait for converting into a [`ParallelIterator`] on a [`ThreadPool`].
pub trait IntoParallelIterator {
    /// The type of items that this parallel iterator produces.
    ///
    /// Items are sent to worker threads (where they are then consumed by the
    /// `process_item` function parameter of the
    /// [`ParallelIterator::pipeline()`]), hence the required [`Send`] bound.
    type Item: Send;

    /// Target parallel iterator type.
    type Iter<'pool, 'scope: 'pool>: ParallelIterator<Item = Self::Item>;

    /// Converts `self` into a parallel iterator to be processed on the given
    /// thread pool.
    fn into_par_iter<'pool, 'scope: 'pool>(
        self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope>;
}

/// Trait for converting into a [`ParallelIterator`] that produces references.
///
/// This trait is automatically implemented for `T` where [`&T`](reference)
/// implements [`IntoParallelIterator`].
pub trait IntoParallelRefIterator<'data> {
    /// The type of items that this parallel iterator produces.
    ///
    /// Like for [`IntoParallelIterator`], items are sent to worker threads
    /// (where they are then consumed by the `process_item` function
    /// parameter of the [`ParallelIterator::pipeline()`]), hence the required
    /// [`Send`] bound.
    type Item: Send;

    /// Target parallel iterator type.
    type Iter<'pool, 'scope: 'pool>: ParallelIterator
    where
        Self: 'data;

    /// Converts `&self` into a parallel iterator to be processed on the given
    /// thread pool.
    ///
    /// ```rust
    /// # use paralight::iter::{IntoParallelRefIterator, ParallelIteratorExt};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = input
    ///     .par_iter(&mut thread_pool)
    ///     .copied()
    ///     .reduce(|| 0, |x, y| x + y);
    /// assert_eq!(sum, 5 * 11);
    /// # });
    /// ```
    fn par_iter<'pool, 'scope: 'pool>(
        &'data self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope>;
}

impl<'data, T: 'data + ?Sized> IntoParallelRefIterator<'data> for T
where
    &'data T: IntoParallelIterator,
{
    type Item = <&'data T as IntoParallelIterator>::Item;
    type Iter<'pool, 'scope: 'pool> = <&'data T as IntoParallelIterator>::Iter<'pool, 'scope>;

    fn par_iter<'pool, 'scope: 'pool>(
        &'data self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope> {
        self.into_par_iter(thread_pool)
    }
}

/// Trait for converting into a [`ParallelIterator`] that produces mutable
/// references.
///
/// This trait is automatically implemented for `T` where [`&mut T`](reference)
/// implements [`IntoParallelIterator`].
pub trait IntoParallelRefMutIterator<'data> {
    /// The type of items that this parallel iterator produces.
    ///
    /// Like for [`IntoParallelIterator`], items are sent to worker threads
    /// (where they are then consumed by the `process_item` function
    /// parameter of the [`ParallelIterator::pipeline()`]), hence the required
    /// [`Send`] bound.
    type Item: Send;

    /// Target parallel iterator type.
    type Iter<'pool, 'scope: 'pool>: ParallelIterator
    where
        Self: 'data;

    /// Converts `&mut self` into a parallel iterator to be processed on the
    /// given thread pool.
    ///
    /// ```rust
    /// # use paralight::iter::{IntoParallelRefMutIterator, ParallelIteratorExt};
    /// # use paralight::{CpuPinningPolicy, ThreadCount, RangeStrategy, ThreadPoolBuilder};
    /// # let pool_builder = ThreadPoolBuilder {
    /// #     num_threads: ThreadCount::AvailableParallelism,
    /// #     range_strategy: RangeStrategy::WorkStealing,
    /// #     cpu_pinning: CpuPinningPolicy::No,
    /// # };
    /// # pool_builder.scope(|mut thread_pool| {
    /// let mut values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    /// let sum = values.par_iter_mut(&mut thread_pool).for_each(|x| {
    ///     *x *= 2;
    /// });
    /// assert_eq!(values, [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]);
    /// # });
    /// ```
    fn par_iter_mut<'pool, 'scope: 'pool>(
        &'data mut self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope>;
}

impl<'data, T: 'data + ?Sized> IntoParallelRefMutIterator<'data> for T
where
    &'data mut T: IntoParallelIterator,
{
    type Item = <&'data mut T as IntoParallelIterator>::Item;
    type Iter<'pool, 'scope: 'pool> = <&'data mut T as IntoParallelIterator>::Iter<'pool, 'scope>;

    fn par_iter_mut<'pool, 'scope: 'pool>(
        &'data mut self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope> {
        self.into_par_iter(thread_pool)
    }
}

/// A parallel iterator over a slice. This struct is created by the
/// [`par_iter()`](IntoParallelRefIterator::par_iter) method on
/// [`IntoParallelRefIterator`].
#[must_use = "iterator adaptors are lazy"]
pub struct SliceParallelIterator<'pool, 'scope: 'pool, 'data, T> {
    thread_pool: &'pool mut ThreadPool<'scope>,
    slice: &'data [T],
}

impl<'data, T: Sync> IntoParallelIterator for &'data [T] {
    type Item = &'data T;
    type Iter<'pool, 'scope: 'pool> = SliceParallelIterator<'pool, 'scope, 'data, T>;

    fn into_par_iter<'pool, 'scope: 'pool>(
        self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope> {
        SliceParallelIterator {
            thread_pool,
            slice: self,
        }
    }
}

impl<'pool, 'scope: 'pool, 'data, T: Sync> ParallelIterator
    for SliceParallelIterator<'pool, 'scope, 'data, T>
{
    type Item = &'data T;

    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> Accum + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        self.thread_pool.pipeline(
            self.slice.len(),
            init,
            |acc, index| process_item(acc, index, &self.slice[index]),
            finalize,
            reduce,
        )
    }
}

/// A parallel iterator over a mutable slice. This struct is created by the
/// [`par_iter_mut()`](IntoParallelRefMutIterator::par_iter_mut) method on
/// [`IntoParallelRefMutIterator`].
#[must_use = "iterator adaptors are lazy"]
pub struct MutSliceParallelIterator<'pool, 'scope: 'pool, 'data, T> {
    thread_pool: &'pool mut ThreadPool<'scope>,
    slice: &'data mut [T],
}

impl<'data, T: Send> IntoParallelIterator for &'data mut [T] {
    type Item = &'data mut T;
    type Iter<'pool, 'scope: 'pool> = MutSliceParallelIterator<'pool, 'scope, 'data, T>;

    fn into_par_iter<'pool, 'scope: 'pool>(
        self,
        thread_pool: &'pool mut ThreadPool<'scope>,
    ) -> Self::Iter<'pool, 'scope> {
        MutSliceParallelIterator {
            thread_pool,
            slice: self,
        }
    }
}

impl<'pool, 'scope: 'pool, 'data, T: Send> ParallelIterator
    for MutSliceParallelIterator<'pool, 'scope, 'data, T>
{
    type Item = &'data mut T;

    fn pipeline<Output: Send, Accum>(
        self,
        init: impl Fn() -> Accum + Sync,
        process_item: impl Fn(Accum, usize, Self::Item) -> Accum + Sync,
        finalize: impl Fn(Accum) -> Output + Sync,
        reduce: impl Fn(Output, Output) -> Output,
    ) -> Output {
        let ptr = MutPtrWrapper(self.slice.as_mut_ptr());
        let len = self.slice.len();
        self.thread_pool.pipeline(
            len,
            init,
            |acc, index| {
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
                process_item(acc, index, item)
            },
            finalize,
            reduce,
        )
    }
}

/// A helper struct for the implementation of [`MutSliceParallelIterator`], that
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
/// comments in [`MutSliceParallelIterator::pipeline`]). Therefore we make it
/// [`Sync`] if and only if [`&mut T`](reference) is [`Send`], which is when `T`
/// is [`Send`].
unsafe impl<T: Send> Sync for MutPtrWrapper<T> {}
