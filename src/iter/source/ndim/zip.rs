// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! TODO

use super::{MultiDimParallelSource, MultiDimSourceDescriptor, RewindableSource, SourceCleanup};

/// TODO
pub trait MultiDimZipableSource<const DIM: usize>: Sized
where
    MultiDimZipEq<Self>: MultiDimParallelSource<DIM>,
{
    /// TODO
    fn ndim_zip_eq(self) -> MultiDimZipEq<Self> {
        MultiDimZipEq(self)
    }
}

/// TODO
#[must_use = "iterator adaptors are lazy"]
pub struct MultiDimZipEq<T>(T);

struct MultiDimZipEqSourceDescriptor<T, const DIM: usize> {
    descriptors: T,
    len: [usize; DIM],
}

macro_rules! assert_all_eq {
    ( $lens:expr, $zero:tt $(, $i:tt)* ) => {
        $( assert_eq!(
            $lens.0,
            $lens.$i,
            "called ndim_zip_eq() with sources of different lengths"
        ); )*
    }
}

macro_rules! or_bools {
    ( $tuple:expr, $zero:tt $(, $i:tt)* ) => {
        $tuple.0 $( || $tuple.$i )*
    }
}

macro_rules! ndim_zipable_tuple {
    ( $($tuple:ident $i:tt),+ ) => {
        impl<const DIM: usize, $($tuple,)+> MultiDimZipableSource<DIM> for ($($tuple,)+)
        where $($tuple: MultiDimParallelSource<DIM>,)+ {}

        impl<const DIM: usize, $($tuple,)+> MultiDimParallelSource<DIM> for MultiDimZipEq<($($tuple,)+)>
        where $($tuple: MultiDimParallelSource<DIM>,)+ {
            type Item = ( $($tuple::Item,)+ );

            fn descriptor(self) -> impl MultiDimSourceDescriptor<DIM, Item = Self::Item> + Sync {
                let tuple = self.0;
                let descriptors = ( $(tuple.$i.descriptor(),)+ );
                let lens = ( $(descriptors.$i.len(),)+ );
                assert_all_eq!(lens, $($i),+);
                MultiDimZipEqSourceDescriptor {
                    descriptors,
                    len: lens.0,
                }
            }
        }

        // SAFETY: TODO
        unsafe impl<$($tuple,)+> RewindableSource for MultiDimZipEq<($($tuple,)+)>
        where $($tuple: RewindableSource,)+ {}

        impl<const DIM: usize, $($tuple,)+> SourceCleanup
        for MultiDimZipEqSourceDescriptor<($($tuple,)+), DIM>
        where $($tuple: SourceCleanup,)+ {
            const NEEDS_CLEANUP: bool = {
                let need_cleanups = ( $($tuple::NEEDS_CLEANUP,)+ );
                or_bools!(need_cleanups, $($i),+)
            };

            unsafe fn cleanup_item_range(&self, range: std::ops::Range<usize>) {
                if Self::NEEDS_CLEANUP {
                    $(
                        // SAFETY: TODO
                        unsafe {
                            self.descriptors.$i.cleanup_item_range(range.clone());
                        }
                    )+
                }
            }
        }

        impl<const DIM: usize, $($tuple,)+> MultiDimSourceDescriptor<DIM>
        for MultiDimZipEqSourceDescriptor<($($tuple,)+), DIM>
        where $($tuple: MultiDimSourceDescriptor<DIM>,)+ {
            type Item = ( $($tuple::Item,)+ );

            fn len(&self) -> [usize; DIM] {
                self.len
            }

            unsafe fn fetch_item(&self, index: [usize; DIM]) -> Self::Item {
                ( $(
                    // SAFETY: TODO
                    unsafe {
                        self.descriptors.$i.fetch_item(index)
                    },
                )+ )
            }
        }
    }
}

ndim_zipable_tuple!(A 0);
ndim_zipable_tuple!(A 0, B 1);
ndim_zipable_tuple!(A 0, B 1, C 2);
ndim_zipable_tuple!(A 0, B 1, C 2, D 3);
ndim_zipable_tuple!(A 0, B 1, C 2, D 3, E 4);
ndim_zipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5);
ndim_zipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6);
ndim_zipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7);
ndim_zipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8);
ndim_zipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9);
ndim_zipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10);
ndim_zipable_tuple!(A 0, B 1, C 2, D 3, E 4, F 5, G 6, H 7, I 8, J 9, K 10, L 11);
