// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! TODO

use super::{
    MultiDimParallelSource, MultiDimSourceDescriptor, ParallelSource, RewindableSource,
    SourceCleanup, SourceDescriptor,
};

/// TODO
///
/// ```
/// # use paralight::iter::{
/// #     IntoParallelSource, MultiDimParallelSourceExt, ParallelIteratorExt,
/// #     ParallelSourceExt, ProductableSource,
/// # };
/// # use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
/// # let mut thread_pool = ThreadPoolBuilder {
/// #     num_threads: ThreadCount::AvailableParallelism,
/// #     range_strategy: RangeStrategy::WorkStealing,
/// #     cpu_pinning: CpuPinningPolicy::No,
/// # }
/// # .build();
/// (
///     (0..2).into_par_iter(),
///     (0..3).into_par_iter(),
///     (0..4).into_par_iter(),
/// )
///     .cartesian_product()
///     .flatten()
///     .with_thread_pool(&mut thread_pool)
///     .for_each(|(i, j, k)| eprintln!("({i}, {j}, {k})"));
/// ```
pub trait ProductableSource<const DIM: usize>: Sized
where
    CartesianProduct<Self>: MultiDimParallelSource<DIM>,
{
    /// TODO
    fn cartesian_product(self) -> CartesianProduct<Self> {
        CartesianProduct(self)
    }
}

#[must_use = "iterator adaptors are lazy"]
pub struct CartesianProduct<T>(T);

struct CartesianProductSourceDescriptor<T, const DIM: usize> {
    descriptors: T,
    len: [usize; DIM],
}

macro_rules! cartesian_product_tuple {
    ( $dim:expr, $($tuple:ident $index:ident $i:tt),+ ) => {
        impl<$($tuple,)+> ProductableSource<$dim> for ($($tuple,)+)
        where $($tuple: ParallelSource + RewindableSource,)+ {}

        impl<$($tuple,)+> MultiDimParallelSource<$dim> for CartesianProduct<($($tuple,)+)>
        where $($tuple: ParallelSource + RewindableSource,)+ {
            type Item = ( $($tuple::Item,)+ );

            fn descriptor(self) -> impl MultiDimSourceDescriptor<$dim, Item = Self::Item> + Sync {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor(),)+ );
                let len = [ $(descriptors.$i.len(),)+ ];
                CartesianProductSourceDescriptor {
                    descriptors,
                    len,
                }
            }
        }

        // SAFETY: TODO
        unsafe impl<$($tuple,)+> RewindableSource for CartesianProduct<($($tuple,)+)>
        where $($tuple: ParallelSource + RewindableSource,)+ {}

        impl<$($tuple,)+> SourceCleanup
        for CartesianProductSourceDescriptor<($($tuple,)+), $dim>
        where $($tuple: SourceDescriptor,)+ {
            const NEEDS_CLEANUP: bool = {
                // This should be implied by the RewindableSource trait, but we check this again as
                // defense in depth.
                $( assert!(!$tuple::NEEDS_CLEANUP); )+
                false
            };

            unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {}
        }

        impl<$($tuple,)+> MultiDimSourceDescriptor<$dim>
        for CartesianProductSourceDescriptor<($($tuple,)+), $dim>
        where $($tuple: SourceDescriptor,)+ {
            type Item = ( $($tuple::Item,)+ );

            fn len(&self) -> [usize; $dim] {
                self.len
            }

            unsafe fn fetch_item(&self, [ $($index,)+ ]: [usize; $dim]) -> Self::Item {
                ( $(
                    // SAFETY: TODO
                    unsafe {
                        self.descriptors.$i.fetch_item($index)
                    },
                )+ )
            }
        }
    }
}

cartesian_product_tuple!(1, A a 0);
cartesian_product_tuple!(2, A a 0, B b 1);
cartesian_product_tuple!(3, A a 0, B b 1, C c 2);
cartesian_product_tuple!(4, A a 0, B b 1, C c 2, D d 3);
cartesian_product_tuple!(5, A a 0, B b 1, C c 2, D d 3, E e 4);
cartesian_product_tuple!(6, A a 0, B b 1, C c 2, D d 3, E e 4, F f 5);
cartesian_product_tuple!(7, A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6);
cartesian_product_tuple!(8, A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7);
cartesian_product_tuple!(9, A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7, I i 8);
cartesian_product_tuple!(10, A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7, I i 8, J j 9);
cartesian_product_tuple!(11, A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7, I i 8, J j 9, K k 10);
cartesian_product_tuple!(12, A a 0, B b 1, C c 2, D d 3, E e 4, F f 5, G g 6, H h 7, I i 8, J j 9, K k 10, L l 11);
