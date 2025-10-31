// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! CLI tool to run examples of parallel tasks.

use clap::{Parser, ValueEnum};
use rand::seq::index;
use rand::SeedableRng;
use rand_chacha::ChaCha12Rng;
use std::hint::black_box;
use std::num::NonZeroUsize;

fn main() {
    let cli = Cli::parse();

    match cli.parallelism {
        Parallelism::Serial => dispatch_serial(cli),
        Parallelism::Paralight => dispatch_paralight(cli),
        Parallelism::Rayon => {
            #[cfg(not(feature = "rayon"))]
            panic!("Please compile this example with the `rayon` feature to use `--parallelism=rayon`.");
            #[cfg(feature = "rayon")]
            dispatch_rayon(cli);
        }
        Parallelism::ParalightOnRayon => {
            #[cfg(not(feature = "rayon"))]
            panic!("Please compile this example with the `rayon` feature to use `--parallelism=paralight-on-rayon`.");
            #[cfg(feature = "rayon")]
            dispatch_paralight_on_rayon(cli);
        }
    }
}

fn dispatch_serial(cli: Cli) {
    match (cli.scenario, cli.owned_input, cli.boxed_items) {
        (Scenario::Sum, false, false) => {
            let input = (0..cli.input_size).collect::<Vec<u64>>();
            let sum = black_box(input).iter().sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Sum, true, false) => {
            let input = (0..cli.input_size).collect::<Vec<u64>>();
            let sum = black_box(input).into_iter().sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Sum, false, true) => {
            let input = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let sum = black_box(input).iter().map(|x| **x).sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Sum, true, true) => {
            let input = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let sum = black_box(input).into_iter().map(|x| *x).sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Add, false, false) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).collect::<Vec<u64>>();
            let right = (0..cli.input_size).collect::<Vec<u64>>();

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
        (Scenario::Add, true, false) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).collect::<Vec<u64>>();
            let right = (0..cli.input_size).collect::<Vec<u64>>();

            let output_slice = output.as_mut_slice();

            black_box(left)
                .into_iter()
                .zip(black_box(right))
                .zip(black_box(output_slice.iter_mut()))
                .for_each(|((a, b), out)| *out = a + b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::Add, false, true) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let right = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();

            let output_slice = output.as_mut_slice();
            let left_slice = left.as_slice();
            let right_slice = right.as_slice();

            black_box(left_slice)
                .iter()
                .zip(black_box(right_slice).iter())
                .zip(black_box(output_slice.iter_mut()))
                .for_each(|((a, b), out)| *out = **a + **b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::Add, true, true) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let right = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();

            let output_slice = output.as_mut_slice();

            black_box(left)
                .into_iter()
                .zip(black_box(right))
                .zip(black_box(output_slice.iter_mut()))
                .for_each(|((a, b), out)| *out = *a + *b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::FindAny | Scenario::FindFirst, false, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice).iter().find(|x| **x);
            println!("found = {found:?}");
        }
        (Scenario::FindAny | Scenario::FindFirst, true, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let found = black_box(input).into_iter().find(|x| *x);
            println!("found = {found:?}");
        }
        (Scenario::FindAny | Scenario::FindFirst, false, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice).iter().find(|x| ***x);
            println!("found = {found:?}");
        }
        (Scenario::FindAny | Scenario::FindFirst, true, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let found = black_box(input).into_iter().find(|x| **x);
            println!("found = {found:?}");
        }
    }
}

fn dispatch_paralight(cli: Cli) {
    use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

    let mut thread_pool = ThreadPoolBuilder {
        num_threads: match cli.num_threads {
            Some(num_threads) => ThreadCount::Count(num_threads),
            None => ThreadCount::AvailableParallelism,
        },
        range_strategy: match cli.range_strategy {
            RangeStrategyCli::Fixed => RangeStrategy::Fixed,
            RangeStrategyCli::WorkStealing => RangeStrategy::WorkStealing,
        },
        cpu_pinning: CpuPinningPolicy::IfSupported,
    }
    .build();

    dispatch_thread_pool(cli, &mut thread_pool);
}

#[cfg(feature = "rayon")]
fn dispatch_paralight_on_rayon(cli: Cli) {
    use paralight::{RangeStrategy, RayonThreadPool, ThreadCount};

    let thread_pool = RayonThreadPool::new_global(
        match cli.num_threads {
            Some(num_threads) => ThreadCount::Count(num_threads),
            None => ThreadCount::AvailableParallelism,
        },
        match cli.range_strategy {
            RangeStrategyCli::Fixed => RangeStrategy::Fixed,
            RangeStrategyCli::WorkStealing => RangeStrategy::WorkStealing,
        },
    );

    dispatch_thread_pool(cli, &thread_pool);
}

fn dispatch_thread_pool(cli: Cli, thread_pool: impl paralight::iter::GenericThreadPool) {
    use paralight::iter::{
        IntoParallelRefMutSource, IntoParallelRefSource, IntoParallelSource, ParallelIteratorExt,
        ParallelSourceExt, ZipableSource,
    };

    match (cli.scenario, cli.owned_input, cli.boxed_items) {
        (Scenario::Sum, false, false) => {
            let input = (0..cli.input_size).collect::<Vec<u64>>();
            let sum = black_box(input)
                .par_iter()
                .with_thread_pool(thread_pool)
                .sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Sum, true, false) => {
            let input = (0..cli.input_size).collect::<Vec<u64>>();
            let sum = black_box(input)
                .into_par_iter()
                .with_thread_pool(thread_pool)
                .sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Sum, false, true) => {
            let input = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let sum = black_box(input)
                .par_iter()
                .with_thread_pool(thread_pool)
                .map(|x| **x)
                .sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Sum, true, true) => {
            let input = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let sum = black_box(input)
                .into_par_iter()
                .with_thread_pool(thread_pool)
                .map(|x| *x)
                .sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Add, false, false) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).collect::<Vec<u64>>();
            let right = (0..cli.input_size).collect::<Vec<u64>>();

            let output_slice = output.as_mut_slice();
            let left_slice = left.as_slice();
            let right_slice = right.as_slice();

            (
                black_box(output_slice.par_iter_mut()),
                black_box(left_slice).par_iter(),
                black_box(right_slice).par_iter(),
            )
                .zip_eq()
                .with_thread_pool(thread_pool)
                .for_each(|(out, a, b)| *out = *a + *b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::Add, true, false) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).collect::<Vec<u64>>();
            let right = (0..cli.input_size).collect::<Vec<u64>>();

            let output_slice = output.as_mut_slice();

            (
                black_box(output_slice.par_iter_mut()),
                black_box(left).into_par_iter(),
                black_box(right).into_par_iter(),
            )
                .zip_eq()
                .with_thread_pool(thread_pool)
                .for_each(|(out, a, b)| *out = a + b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::Add, false, true) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let right = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();

            let output_slice = output.as_mut_slice();
            let left_slice = left.as_slice();
            let right_slice = right.as_slice();

            (
                black_box(output_slice.par_iter_mut()),
                black_box(left_slice).par_iter(),
                black_box(right_slice).par_iter(),
            )
                .zip_eq()
                .with_thread_pool(thread_pool)
                .for_each(|(out, a, b)| *out = **a + **b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::Add, true, true) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let right = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();

            let output_slice = output.as_mut_slice();

            (
                black_box(output_slice.par_iter_mut()),
                black_box(left).into_par_iter(),
                black_box(right).into_par_iter(),
            )
                .zip_eq()
                .with_thread_pool(thread_pool)
                .for_each(|(out, a, b)| *out = *a + *b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::FindAny, false, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice)
                .par_iter()
                .with_thread_pool(thread_pool)
                .find_any(|x| **x);
            println!("found = {found:?}");
        }
        (Scenario::FindAny, true, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let found = black_box(input)
                .into_par_iter()
                .with_thread_pool(thread_pool)
                .find_any(|x| *x);
            println!("found = {found:?}");
        }
        (Scenario::FindAny, false, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice)
                .par_iter()
                .with_thread_pool(thread_pool)
                .find_any(|x| ***x);
            println!("found = {found:?}");
        }
        (Scenario::FindAny, true, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let found = black_box(input)
                .into_par_iter()
                .with_thread_pool(thread_pool)
                .find_any(|x| **x);
            println!("found = {found:?}");
        }
        (Scenario::FindFirst, false, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice)
                .par_iter()
                .with_thread_pool(thread_pool)
                .find_first(|x| **x);
            println!("found = {found:?}");
        }
        (Scenario::FindFirst, true, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let found = black_box(input)
                .into_par_iter()
                .with_thread_pool(thread_pool)
                .find_first(|x| *x);
            println!("found = {found:?}");
        }
        (Scenario::FindFirst, false, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice)
                .par_iter()
                .with_thread_pool(thread_pool)
                .find_first(|x| ***x);
            println!("found = {found:?}");
        }
        (Scenario::FindFirst, true, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let found = black_box(input)
                .into_par_iter()
                .with_thread_pool(thread_pool)
                .find_first(|x| **x);
            println!("found = {found:?}");
        }
    }
}

#[cfg(feature = "rayon")]
fn dispatch_rayon(cli: Cli) {
    use rayon::iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    };

    match (cli.scenario, cli.owned_input, cli.boxed_items) {
        (Scenario::Sum, false, false) => {
            let input = (0..cli.input_size).collect::<Vec<u64>>();
            let sum = black_box(input).par_iter().sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Sum, true, false) => {
            let input = (0..cli.input_size).collect::<Vec<u64>>();
            let sum = black_box(input).into_par_iter().sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Sum, false, true) => {
            let input = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let sum = black_box(input).par_iter().map(|x| **x).sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Sum, true, true) => {
            let input = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let sum = black_box(input).into_par_iter().map(|x| *x).sum::<u64>();
            println!("sum = {sum}");
        }
        (Scenario::Add, false, false) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).collect::<Vec<u64>>();
            let right = (0..cli.input_size).collect::<Vec<u64>>();

            let output_slice = output.as_mut_slice();
            let left_slice = left.as_slice();
            let right_slice = right.as_slice();

            black_box(output_slice.par_iter_mut())
                .zip(
                    black_box(left_slice)
                        .par_iter()
                        .zip(black_box(right_slice).par_iter()),
                )
                .for_each(|(out, (a, b))| *out = *a + *b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::Add, true, false) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).collect::<Vec<u64>>();
            let right = (0..cli.input_size).collect::<Vec<u64>>();

            let output_slice = output.as_mut_slice();

            black_box(output_slice.par_iter_mut())
                .zip(
                    black_box(left)
                        .into_par_iter()
                        .zip(black_box(right).into_par_iter()),
                )
                .for_each(|(out, (a, b))| *out = a + b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::Add, false, true) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let right = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();

            let output_slice = output.as_mut_slice();
            let left_slice = left.as_slice();
            let right_slice = right.as_slice();

            black_box(output_slice.par_iter_mut())
                .zip(
                    black_box(left_slice)
                        .par_iter()
                        .zip(black_box(right_slice).par_iter()),
                )
                .for_each(|(out, (a, b))| *out = **a + **b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::Add, true, true) => {
            let mut output = vec![0; cli.input_size as usize];
            let left = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();
            let right = (0..cli.input_size).map(Box::new).collect::<Vec<Box<u64>>>();

            let output_slice = output.as_mut_slice();

            black_box(output_slice.par_iter_mut())
                .zip(
                    black_box(left)
                        .into_par_iter()
                        .zip(black_box(right).into_par_iter()),
                )
                .for_each(|(out, (a, b))| *out = *a + *b);
            println!("added {} elements", black_box(output).len());
        }
        (Scenario::FindAny, false, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice).par_iter().find_any(|x| **x);
            println!("found = {found:?}");
        }
        (Scenario::FindAny, true, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let found = black_box(input).into_par_iter().find_any(|x| *x);
            println!("found = {found:?}");
        }
        (Scenario::FindAny, false, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice).par_iter().find_any(|x| ***x);
            println!("found = {found:?}");
        }
        (Scenario::FindAny, true, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let found = black_box(input).into_par_iter().find_any(|x| **x);
            println!("found = {found:?}");
        }
        (Scenario::FindFirst, false, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice).par_iter().find_first(|x| **x);
            println!("found = {found:?}");
        }
        (Scenario::FindFirst, true, false) => {
            let input = fill_needles(cli.input_size as usize, cli.density);
            let found = black_box(input).into_par_iter().find_first(|x| *x);
            println!("found = {found:?}");
        }
        (Scenario::FindFirst, false, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let input_slice = input.as_slice();
            let found = black_box(input_slice).par_iter().find_first(|x| ***x);
            println!("found = {found:?}");
        }
        (Scenario::FindFirst, true, true) => {
            let input = fill_boxed_needles(cli.input_size as usize, cli.density);
            let found = black_box(input).into_par_iter().find_first(|x| **x);
            println!("found = {found:?}");
        }
    }
}

/// Creates a vector of `input_size` booleans, `density` of which are set to
/// [`true`] (the needles). The set of needles follows a uniform distribution,
/// but is fixed by a constant-time seed for reproducibility.
fn fill_needles(input_size: usize, density: usize) -> Vec<bool> {
    let mut input = vec![false; input_size];

    let mut rng = ChaCha12Rng::seed_from_u64(42);
    let needles = index::sample(&mut rng, input_size, density);
    for needle in needles {
        input[needle] = true;
    }

    input
}

/// Creates a vector of `input_size` [`Box`]ed booleans, `density` of which are
/// set to [`true`] (the needles). The set of needles follows a uniform
/// distribution, but is fixed by a constant-time seed for reproducibility.
#[allow(clippy::vec_box)]
fn fill_boxed_needles(input_size: usize, density: usize) -> Vec<Box<bool>> {
    let mut input = vec![Box::new(false); input_size];

    let mut rng = ChaCha12Rng::seed_from_u64(42);
    let needles = index::sample(&mut rng, input_size, density);
    for needle in needles {
        *input[needle] = true;
    }

    input
}

/// CLI tool to run examples of parallel tasks.
#[derive(Parser, Debug, PartialEq, Eq)]
#[command(version)]
struct Cli {
    /// Number of worker threads. Default to the available parallelism.
    #[arg(long)]
    num_threads: Option<NonZeroUsize>,

    /// Policy to split work among threads.
    #[arg(long, value_enum)]
    range_strategy: RangeStrategyCli,

    /// Parallelism mode.
    #[arg(long, value_enum, default_value_t = Parallelism::Paralight)]
    parallelism: Parallelism,

    /// Scenario to run in parallel.
    #[arg(long, value_enum)]
    scenario: Scenario,

    /// Number of items in the input.
    #[arg(long, default_value_t = 1_000_000)]
    input_size: u64,

    /// Whether to consume an owned input rather than a slice.
    #[arg(long, default_value_t = false)]
    owned_input: bool,

    /// Whether to use boxed items rather than plain items.
    #[arg(long, default_value_t = false)]
    boxed_items: bool,

    /// Density of items that match the search. Used only for the find-any and
    /// find-first scenarios.
    #[arg(long, default_value_t = 1)]
    density: usize,
}

/// Policy to split work among threads.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
enum RangeStrategyCli {
    /// Each worker thread processes a fixed sequential chunk of items.
    Fixed,
    /// Work threads can steal items from each other.
    WorkStealing,
}

/// Scenario to run.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
enum Scenario {
    /// Sum a slice of numbers.
    Sum,
    /// Add two slices element-wise.
    Add,
    /// Find any value.
    FindAny,
    /// Find the first value.
    FindFirst,
}

/// Parallelism mode.
#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
enum Parallelism {
    /// Process the input serially.
    Serial,
    /// Use paralight.
    Paralight,
    /// Use rayon. Only available with the `rayon` feature enabled.
    Rayon,
    /// Use paralight on top of rayon's global thread pool. Only available with
    /// the `rayon` feature enabled.
    ParalightOnRayon,
}
