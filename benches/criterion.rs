// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use ::paralight::RangeStrategy;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::mem::size_of;

const NUM_THREADS: &[usize] = &[1, 2, 4, 8];
const LENGTHS: &[usize] = &[10_000, 100_000, 1_000_000, 10_000_000];

fn sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum");
    for len in LENGTHS {
        group.throughput(Throughput::Bytes((len * size_of::<u64>()) as u64));
        group.bench_with_input(BenchmarkId::new("serial", len), len, serial::sum);
        for &num_threads in NUM_THREADS {
            group.bench_with_input(
                BenchmarkId::new(format!("rayon@{num_threads}"), len),
                len,
                |bencher, len| rayon::sum(bencher, num_threads, len),
            );
            for (range_strategy, range_name) in [
                (RangeStrategy::Fixed, "fixed"),
                (RangeStrategy::WorkStealing, "work-stealing"),
            ] {
                group.bench_with_input(
                    BenchmarkId::new(format!("paralight_{range_name}@{num_threads}"), len),
                    len,
                    |bencher, len| paralight::sum(bencher, range_strategy, num_threads, len),
                );
            }
        }
    }
    group.finish();
}

fn add(c: &mut Criterion) {
    let mut group = c.benchmark_group("add");
    for len in LENGTHS {
        group.throughput(Throughput::Bytes((len * size_of::<u64>()) as u64));
        group.bench_with_input(BenchmarkId::new("serial", len), len, serial::add);
        for &num_threads in NUM_THREADS {
            group.bench_with_input(
                BenchmarkId::new(format!("rayon@{num_threads}"), len),
                len,
                |bencher, len| rayon::add(bencher, num_threads, len),
            );
            for (range_strategy, range_name) in [
                (RangeStrategy::Fixed, "fixed"),
                (RangeStrategy::WorkStealing, "work-stealing"),
            ] {
                group.bench_with_input(
                    BenchmarkId::new(format!("paralight_{range_name}@{num_threads}"), len),
                    len,
                    |bencher, len| paralight::add(bencher, range_strategy, num_threads, len),
                );
            }
        }
    }
    group.finish();
}

/// Baseline benchmarks using serial iterators (without any multi-threading
/// involved).
mod serial {
    use criterion::{black_box, Bencher};

    pub fn sum(bencher: &mut Bencher, len: &usize) {
        let input = (0..*len as u64).collect::<Vec<u64>>();
        let input_slice = input.as_slice();
        bencher.iter(|| black_box(input_slice).iter().sum::<u64>());
    }

    pub fn add(bencher: &mut Bencher, len: &usize) {
        let left = (0..*len as u64).collect::<Vec<u64>>();
        let right = (0..*len as u64).collect::<Vec<u64>>();
        let mut output = vec![0; *len];

        let left_slice = left.as_slice();
        let right_slice = right.as_slice();
        let output_slice = output.as_mut_slice();

        bencher.iter(|| {
            black_box(left_slice)
                .iter()
                .zip(black_box(right_slice))
                .zip(black_box(output_slice.iter_mut()))
                .for_each(|((&a, &b), out)| *out = a + b)
        });
    }
}

/// Benchmarks using Rayon.
mod rayon {
    use criterion::{black_box, Bencher};
    use rayon::iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    };

    pub fn sum(bencher: &mut Bencher, num_threads: usize, len: &usize) {
        let input = (0..*len as u64).collect::<Vec<u64>>();
        let input_slice = input.as_slice();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        thread_pool.install(|| bencher.iter(|| black_box(input_slice).par_iter().sum::<u64>()));
    }

    pub fn add(bencher: &mut Bencher, num_threads: usize, len: &usize) {
        let left = (0..*len as u64).collect::<Vec<u64>>();
        let right = (0..*len as u64).collect::<Vec<u64>>();
        let mut output = vec![0; *len];

        let left_slice = left.as_slice();
        let right_slice = right.as_slice();
        let output_slice = output.as_mut_slice();

        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        thread_pool.install(|| {
            bencher.iter(|| {
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
    use criterion::{black_box, Bencher};
    use paralight::iter::{
        IntoParallelRefMutSource, IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt,
        ZipableSource,
    };
    use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

    pub fn sum(
        bencher: &mut Bencher,
        range_strategy: RangeStrategy,
        num_threads: usize,
        len: &usize,
    ) {
        let input = (0..*len as u64).collect::<Vec<u64>>();
        let input_slice = input.as_slice();
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::try_from(num_threads).unwrap(),
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        }
        .build();

        bencher.iter(|| {
            black_box(input_slice)
                .par_iter()
                .with_thread_pool(&mut thread_pool)
                .copied()
                .reduce(|| 0, |x, y| x + y)
        });
    }

    pub fn add(
        bencher: &mut Bencher,
        range_strategy: RangeStrategy,
        num_threads: usize,
        len: &usize,
    ) {
        let mut output = vec![0; *len];
        let left = (0..*len as u64).collect::<Vec<u64>>();
        let right = (0..*len as u64).collect::<Vec<u64>>();

        let output_slice = output.as_mut_slice();
        let left_slice = left.as_slice();
        let right_slice = right.as_slice();

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::try_from(num_threads).unwrap(),
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        }
        .build();

        bencher.iter(|| {
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

criterion_group!(benches, sum, add);
criterion_main!(benches);
