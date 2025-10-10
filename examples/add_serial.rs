// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple program that adds two slices element-wise using the standard library.

use std::hint::black_box;

fn main() {
    let input_size = 100_000_000;

    let mut output = vec![0; input_size as usize];
    let left = (0..input_size).collect::<Vec<u64>>();
    let right = (0..input_size).collect::<Vec<u64>>();

    let output_slice = output.as_mut_slice();
    let left_slice = left.as_slice();
    let right_slice = right.as_slice();

    black_box(left_slice)
        .iter()
        .zip(black_box(right_slice).iter())
        .zip(black_box(output_slice.iter_mut()))
        .for_each(|((a, b), out)| *out = *a + *b);
    println!("added {} elements", black_box(output).len());
}
