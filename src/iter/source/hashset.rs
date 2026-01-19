// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Helper module to test non-exact parallel sources, by implementing
//! [`ParallelSource`] on a simplified hash set type.

use super::{
    IntoParallelRefSource, ParallelSource, SimpleSourceDescriptor, SourceCleanup, SourceDescriptor,
};
use hashbrown::hash_table::Entry;
use hashbrown::{DefaultHashBuilder, HashTable};
use std::hash::{BuildHasher, Hash};

pub struct MyHashSet<T> {
    table: HashTable<T>,
    hasher: DefaultHashBuilder,
}

impl<T> MyHashSet<T> {
    pub fn with_capacity(n: usize) -> Self {
        Self {
            table: HashTable::with_capacity(n),
            hasher: DefaultHashBuilder::default(),
        }
    }
}

impl<T: Hash + Eq> MyHashSet<T> {
    pub fn insert(&mut self, t: T) -> bool {
        match self.table.entry(
            self.hasher.hash_one(&t),
            |val| *val == t,
            |val| self.hasher.hash_one(val),
        ) {
            Entry::Vacant(entry) => {
                entry.insert(t);
                true
            }
            Entry::Occupied(mut entry) => {
                *entry.get_mut() = t;
                false
            }
        }
    }
}

impl<T: Hash + Eq> FromIterator<T> for MyHashSet<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let mut set = Self::with_capacity(iter.size_hint().0);
        for x in iter {
            set.insert(x);
        }
        set
    }
}

impl<'data, T: 'data + Sync> IntoParallelRefSource<'data> for MyHashSet<T> {
    type Item = &'data T;
    type Source = HashTableParallelSource<'data, T>;

    fn par_iter(&'data self) -> Self::Source {
        HashTableParallelSource { table: &self.table }
    }
}

#[must_use = "iterator adaptors are lazy"]
pub struct HashTableParallelSource<'data, T> {
    table: &'data HashTable<T>,
}

impl<'data, T: Sync> ParallelSource for HashTableParallelSource<'data, T> {
    type Item = &'data T;

    fn descriptor(self) -> impl SourceDescriptor<Item = Self::Item> + Sync {
        HashTableSourceDescriptor { table: self.table }
    }
}

struct HashTableSourceDescriptor<'data, T: Sync> {
    table: &'data HashTable<T>,
}

impl<T: Sync> SourceCleanup for HashTableSourceDescriptor<'_, T> {
    const NEEDS_CLEANUP: bool = false;

    fn len(&self) -> usize {
        self.table.num_buckets()
    }

    unsafe fn cleanup_item_range(&self, _range: std::ops::Range<usize>) {
        // Nothing to cleanup
    }
}

impl<'data, T: Sync> SimpleSourceDescriptor for HashTableSourceDescriptor<'data, T> {
    type Item = &'data T;

    unsafe fn simple_fetch_item(&self, index: usize) -> Option<Self::Item> {
        self.table.get_bucket(index)
    }
}
