// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Simple program that computes the sum of a slice using the standard library.

use std::hint::black_box;

fn main() {
    let input_size = 1_000_000;

    let input = (0..input_size).collect::<Vec<u64>>();
    let sum = black_box(input).iter().sum::<u64>();
    println!("sum = {sum}");
}
