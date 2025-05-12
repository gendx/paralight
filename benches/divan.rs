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
const LENGTHS: &[usize] = &[10_000, 100_000, 1_000_000];

/// Baseline benchmarks using serial iterators (without any multi-threading
/// involved).
mod serial {
    use super::LENGTHS;
    use divan::counter::BytesCount;
    use divan::{black_box, Bencher};

    #[divan::bench(args = LENGTHS)]
    fn sum(bencher: Bencher, len: usize) {
        let input = (0..len as u64).collect::<Vec<u64>>();
        let input_slice = input.as_slice();
        bencher
            .counter(BytesCount::of_many::<u64>(len))
            .bench_local(|| black_box(input_slice).iter().sum::<u64>())
    }

    #[divan::bench(args = LENGTHS)]
    fn add(bencher: Bencher, len: usize) {
        let left = (0..len as u64).collect::<Vec<u64>>();
        let right = (0..len as u64).collect::<Vec<u64>>();
        let mut output = vec![0; len];

        let left_slice = left.as_slice();
        let right_slice = right.as_slice();
        let output_slice = output.as_mut_slice();

        bencher
            .counter(BytesCount::of_many::<u64>(len * 2))
            .bench_local(|| {
                black_box(left_slice)
                    .iter()
                    .zip(black_box(right_slice))
                    .zip(black_box(output_slice.iter_mut()))
                    .for_each(|((&a, &b), out)| *out = a + b)
            })
    }
}

/// Benchmarks using Rayon.
mod rayon {
    use super::{LENGTHS, NUM_THREADS};
    use divan::counter::BytesCount;
    use divan::{black_box, Bencher};
    use rayon::iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    };

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn sum_rayon<const NUM_THREADS: usize>(bencher: Bencher, len: usize) {
        let input = (0..len as u64).collect::<Vec<u64>>();
        let input_slice = input.as_slice();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(NUM_THREADS)
            .build()
            .unwrap();
        // Ideally we'd prefer to run bench_local() inside the Rayon thread pool, but
        // that doesn't work because divan::Bencher isn't Send (and bench_local()
        // consumes it).
        bencher
            .counter(BytesCount::of_many::<u64>(len))
            .bench_local(|| thread_pool.install(|| black_box(input_slice).par_iter().sum::<u64>()));
    }

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn add_rayon<const NUM_THREADS: usize>(bencher: Bencher, len: usize) {
        let left = (0..len as u64).collect::<Vec<u64>>();
        let right = (0..len as u64).collect::<Vec<u64>>();
        let mut output = vec![0; len];

        let left_slice = left.as_slice();
        let right_slice = right.as_slice();
        let output_slice = output.as_mut_slice();

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(NUM_THREADS)
            .build()
            .unwrap();
        // Ideally we'd prefer to run bench_local() inside the Rayon thread pool, but
        // that doesn't work because divan::Bencher isn't Send (and bench_local()
        // consumes it).
        bencher
            .counter(BytesCount::of_many::<u64>(len * 2))
            .bench_local(|| {
                thread_pool.install(|| {
                    black_box(left_slice)
                        .par_iter()
                        .zip(black_box(right_slice))
                        .zip(black_box(output_slice.par_iter_mut()))
                        .for_each(|((&a, &b), out)| *out = a + b)
                })
            });
    }
}

/// Benchmarks using Paralight.
mod paralight {
    use super::{LENGTHS, NUM_THREADS};
    use divan::counter::BytesCount;
    use divan::{black_box, Bencher};
    use paralight::iter::{
        IntoParallelRefMutSource, IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt,
        ZipableSource,
    };
    use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn sum_fixed<const NUM_THREADS: usize>(bencher: Bencher, len: usize) {
        sum_impl::<NUM_THREADS>(bencher, len, RangeStrategy::Fixed)
    }

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn sum_work_stealing<const NUM_THREADS: usize>(bencher: Bencher, len: usize) {
        sum_impl::<NUM_THREADS>(bencher, len, RangeStrategy::WorkStealing)
    }

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn sum_totem<const NUM_THREADS: usize>(bencher: Bencher, len: usize) {
        sum_impl::<NUM_THREADS>(bencher, len, RangeStrategy::Totem)
    }

    fn sum_impl<const NUM_THREADS: usize>(
        bencher: Bencher,
        len: usize,
        range_strategy: RangeStrategy,
    ) {
        let input = (0..len as u64).collect::<Vec<u64>>();
        let input_slice = input.as_slice();
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::try_from(NUM_THREADS).unwrap(),
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        }
        .build();

        bencher
            .counter(BytesCount::of_many::<u64>(len))
            .bench_local(|| {
                black_box(input_slice)
                    .par_iter()
                    .with_thread_pool(&mut thread_pool)
                    .sum::<u64>()
            });
    }

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn add_fixed<const NUM_THREADS: usize>(bencher: Bencher, len: usize) {
        add_impl::<NUM_THREADS>(bencher, len, RangeStrategy::Fixed)
    }

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn add_work_stealing<const NUM_THREADS: usize>(bencher: Bencher, len: usize) {
        add_impl::<NUM_THREADS>(bencher, len, RangeStrategy::WorkStealing)
    }

    #[divan::bench(consts = NUM_THREADS, args = LENGTHS)]
    fn add_totem<const NUM_THREADS: usize>(bencher: Bencher, len: usize) {
        add_impl::<NUM_THREADS>(bencher, len, RangeStrategy::Totem)
    }

    fn add_impl<const NUM_THREADS: usize>(
        bencher: Bencher,
        len: usize,
        range_strategy: RangeStrategy,
    ) {
        let mut output = vec![0; len];
        let left = (0..len as u64).collect::<Vec<u64>>();
        let right = (0..len as u64).collect::<Vec<u64>>();

        let output_slice = output.as_mut_slice();
        let left_slice = left.as_slice();
        let right_slice = right.as_slice();

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::try_from(NUM_THREADS).unwrap(),
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        }
        .build();

        bencher
            .counter(BytesCount::of_many::<u64>(len * 2))
            .bench_local(|| {
                (
                    black_box(output_slice.par_iter_mut()),
                    black_box(left_slice).par_iter(),
                    black_box(right_slice).par_iter(),
                )
                    .zip_eq()
                    .with_thread_pool(&mut thread_pool)
                    .for_each(|(out, &a, &b)| *out = a + b)
            });
    }
}
