// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple program that computes the sum of a slice using Paralight with the
//! fixed strategy.

use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};
use std::hint::black_box;

fn main() {
    let mut thread_pool = ThreadPoolBuilder {
        num_threads: ThreadCount::AvailableParallelism,
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::IfSupported,
    }
    .build();

    let input_size = 1_000_000;

    let input = (0..input_size).collect::<Vec<u64>>();
    let sum = black_box(input)
        .par_iter()
        .with_thread_pool(&mut thread_pool)
        .sum::<u64>();
    println!("sum = {sum}");
}
