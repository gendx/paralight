// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn main() {
    divan::main();
}

const NUM_THREADS: &[usize] = &[1, 2, 4, 8];
const LENGTHS: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

/// Baseline benchmarks using serial iterators (without any multi-threading
/// involved).
mod serial {
    use super::LENGTHS;

    #[divan::bench(args = LENGTHS)]
    fn sum(bencher: divan::Bencher, len: usize) {
        let input = (0..=len as u64).collect::<Vec<u64>>();
        bencher.bench_local(|| input.iter().sum::<u64>())
    }
}

/// Benchmarks using Rayon.
mod rayon {
    use super::{LENGTHS, NUM_THREADS};
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn sum_rayon<const NUM_THREADS: usize>(bencher: divan::Bencher, len: usize) {
        let input = (0..=len as u64).collect::<Vec<u64>>();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(NUM_THREADS)
            .build()
            .unwrap();
        // Ideally we'd prefer to run bench_local() inside the Rayon thread pool, but
        // that doesn't work because divan::Bencher isn't Send (and bench_local()
        // consumes it).
        bencher.bench_local(|| thread_pool.install(|| input.par_iter().sum::<u64>()));
    }
}

/// Benchmarks using Paralight.
mod paralight {
    use super::{LENGTHS, NUM_THREADS};
    use paralight::{RangeStrategy, ThreadAccumulator, ThreadPoolBuilder};
    use std::num::NonZeroUsize;

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn sum_fixed<const NUM_THREADS: usize>(bencher: divan::Bencher, len: usize) {
        sum_impl::<NUM_THREADS>(bencher, len, RangeStrategy::Fixed)
    }

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn sum_work_stealing<const NUM_THREADS: usize>(bencher: divan::Bencher, len: usize) {
        sum_impl::<NUM_THREADS>(bencher, len, RangeStrategy::WorkStealing)
    }

    fn sum_impl<const NUM_THREADS: usize>(
        bencher: divan::Bencher,
        len: usize,
        range_strategy: RangeStrategy,
    ) {
        let input = (0..=len as u64).map(|x| x.into()).collect::<Vec<u64>>();
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(NUM_THREADS).unwrap(),
            range_strategy,
        };
        pool_builder.scope(
            &input,
            || SumAccumulator,
            move |thread_pool| {
                bencher.bench_local(|| thread_pool.process_inputs().reduce(|a, b| a + b).unwrap());
            },
        );
    }

    struct SumAccumulator;

    impl ThreadAccumulator<u64, u64> for SumAccumulator {
        type Accumulator<'a> = u64;
        fn init(&self) -> u64 {
            0
        }
        fn process_item(&self, accumulator: &mut u64, _index: usize, x: &u64) {
            *accumulator += *x;
        }
        fn finalize(&self, accumulator: u64) -> u64 {
            accumulator
        }
    }
}
