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
                BenchmarkId::new(&format!("rayon@{num_threads}"), len),
                len,
                |bencher, len| rayon::sum(bencher, num_threads, len),
            );
            for (range_strategy, range_name) in [
                (RangeStrategy::Fixed, "fixed"),
                (RangeStrategy::WorkStealing, "work-stealing"),
            ] {
                group.bench_with_input(
                    BenchmarkId::new(&format!("paralight_{range_name}@{num_threads}"), len),
                    len,
                    |bencher, len| paralight::sum(bencher, range_strategy, num_threads, len),
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
        let input = (0..=*len as u64).collect::<Vec<u64>>();
        let input_slice = input.as_slice();
        bencher.iter(|| black_box(input_slice).iter().sum::<u64>());
    }
}

/// Benchmarks using Rayon.
mod rayon {
    use criterion::{black_box, Bencher};
    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    pub fn sum(bencher: &mut Bencher, num_threads: usize, len: &usize) {
        let input = (0..=*len as u64).collect::<Vec<u64>>();
        let input_slice = input.as_slice();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        thread_pool.install(|| bencher.iter(|| black_box(input_slice).par_iter().sum::<u64>()));
    }
}

/// Benchmarks using Paralight.
mod paralight {
    use criterion::{black_box, Bencher};
    use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
    use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

    pub fn sum(
        bencher: &mut Bencher,
        range_strategy: RangeStrategy,
        num_threads: usize,
        len: &usize,
    ) {
        let input = (0..=*len as u64).collect::<Vec<u64>>();
        let input_slice = input.as_slice();
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::try_from(num_threads).unwrap(),
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        pool_builder.scope(|mut thread_pool| {
            bencher.iter(|| {
                black_box(input_slice)
                    .par_iter()
                    .with_thread_pool(&mut thread_pool)
                    .copied()
                    .reduce(|| 0, |x, y| x + y)
            });
        });
    }
}

criterion_group!(benches, sum);
criterion_main!(benches);
