// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Thread pool implementations.

#[cfg(feature = "rayon")]
mod rayon;
#[cfg(feature = "default-thread-pool")]
mod thread_pool;

#[cfg(feature = "rayon")]
pub use rayon::RayonThreadPool;
use std::num::NonZeroUsize;
#[cfg(feature = "default-thread-pool")]
pub use thread_pool::{CpuPinningPolicy, ThreadPool, ThreadPoolBuilder};

/// Number of threads to spawn in a thread pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThreadCount {
    /// Spawn the number of threads returned by
    /// [`std::thread::available_parallelism()`].
    AvailableParallelism,
    /// Spawn the given number of threads.
    Count(NonZeroUsize),
}

impl ThreadCount {
    /// Resolves the number of threads to spawn.
    pub fn count(self) -> NonZeroUsize {
        match self {
            ThreadCount::AvailableParallelism => std::thread::available_parallelism()
                .expect("Getting the available parallelism failed"),
            ThreadCount::Count(count) => count,
        }
    }
}

impl TryFrom<usize> for ThreadCount {
    type Error = <NonZeroUsize as TryFrom<usize>>::Error;

    fn try_from(thread_count: usize) -> Result<Self, Self::Error> {
        let count = NonZeroUsize::try_from(thread_count)?;
        Ok(ThreadCount::Count(count))
    }
}

/// Strategy to distribute ranges of work items among threads.
#[derive(Clone, Copy)]
pub enum RangeStrategy {
    /// Each thread processes a fixed range of items.
    Fixed,
    /// Threads can steal work from each other.
    WorkStealing,
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_thread_count_try_from_usize() {
        assert!(ThreadCount::try_from(0).is_err());
        assert_eq!(
            ThreadCount::try_from(1),
            Ok(ThreadCount::Count(NonZeroUsize::try_from(1).unwrap()))
        );
    }
}
