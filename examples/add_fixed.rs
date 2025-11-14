// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple program that adds two slices element-wise using Paralight with the
//! fixed strategy.

use paralight::prelude::*;
use std::hint::black_box;

fn main() {
    let mut thread_pool = ThreadPoolBuilder {
        num_threads: ThreadCount::AvailableParallelism,
        range_strategy: RangeStrategy::Fixed,
        cpu_pinning: CpuPinningPolicy::IfSupported,
    }
    .build();

    let input_size = 100_000_000;

    let mut output = vec![0; input_size as usize];
    let left = (0..input_size).collect::<Vec<u64>>();
    let right = (0..input_size).collect::<Vec<u64>>();

    let output_slice = output.as_mut_slice();
    let left_slice = left.as_slice();
    let right_slice = right.as_slice();

    (
        black_box(output_slice.par_iter_mut()),
        black_box(left_slice).par_iter(),
        black_box(right_slice).par_iter(),
    )
        .zip_eq()
        .with_thread_pool(&mut thread_pool)
        .for_each(|(out, a, b)| *out = *a + *b);
    println!("added {} elements", black_box(output).len());
}
