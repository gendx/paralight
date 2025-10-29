// Copyright 2024-2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Core engine: thread pool, work stealing and synchronization primitives.

mod pipeline;
mod range;
#[cfg(feature = "rayon")]
mod rayon;
mod sync;
mod thread_pool;
mod util;

#[cfg(feature = "rayon")]
pub use rayon::RayonThreadPool;
pub use thread_pool::{
    CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPool, ThreadPoolBuilder,
};
