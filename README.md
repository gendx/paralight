# Paralight: a lightweight parallelism library for indexed structures

[![Crate](https://img.shields.io/crates/v/paralight.svg?logo=rust)](https://crates.io/crates/paralight)
[![Documentation](https://img.shields.io/docsrs/paralight?logo=rust)](https://docs.rs/paralight)
[![Minimum Rust 1.75.0](https://img.shields.io/badge/rust-1.75.0%2B-orange.svg?logo=rust)](https://releases.rs/docs/1.75.0/)
[![Lines of Code](https://www.aschey.tech/tokei/github/gendx/paralight?category=code&branch=main)](https://github.com/gendx/paralight)
[![Dependencies](https://deps.rs/repo/github/gendx/paralight/status.svg)](https://deps.rs/repo/github/gendx/paralight)
[![License](https://img.shields.io/crates/l/paralight.svg)](https://github.com/gendx/paralight/blob/main/LICENSE)
[![Codecov](https://codecov.io/gh/gendx/paralight/branch/main/graph/badge.svg)](https://app.codecov.io/gh/gendx/paralight/tree/main)
[![Build Status](https://github.com/gendx/paralight/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/gendx/paralight/actions/workflows/build.yml)
[![Test Status](https://github.com/gendx/paralight/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/gendx/paralight/actions/workflows/tests.yml)

This library allows you to distribute computation over slices among multiple
threads. Each thread processes a subset of the items, and a final step reduces
the outputs from all threads into a single result.

```rust
use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

// Define thread pool parameters.
let pool_builder = ThreadPoolBuilder {
    num_threads: ThreadCount::AvailableParallelism,
    range_strategy: RangeStrategy::WorkStealing,
    cpu_pinning: CpuPinningPolicy::No,
};

// Create a scoped thread pool.
let sum = pool_builder.scope(
    |mut thread_pool| {
        // Compute the sum of a slice.
        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .reduce(|| 0, |x, y| x + y)
    },
);
assert_eq!(sum, 5 * 11);
```

Note: In principle, Paralight could be extended to support other inputs than
slices as long as they are *indexed*, but for now only slices are supported.
Come back to check when future versions are published!

## Thread pool configuration

The [`ThreadPoolBuilder`](ThreadPoolBuilder) provides an explicit way to
configure your thread pool, giving you fine-grained control over performance for
your workload. There is no default, which is deliberate because the suitable
parameters depend on your workload.

### Number of worker threads

Paralight allows you to specify the number of worker threads to spawn in a
thread pool with the [`ThreadCount`](ThreadCount) enum:

- [`AvailableParallelism`](ThreadCount::AvailableParallelism) uses the number of
  threads returned by the standard library's
  [`available_parallelism()`](std::thread::available_parallelism) function,
- [`Count(_)`](ThreadCount::Count) uses the specified number of threads, which
  must be non-zero.

For convenience, [`ThreadCount`](ThreadCount) implements the
[`TryFrom<usize>`](TryFrom) trait to create a [`Count(_)`](ThreadCount::Count)
instance, validating that the given number of threads is not zero.

### Work-stealing strategy

Paralight offers two strategies in the [`RangeStrategy`](RangeStrategy) enum to
distribute computation among threads:

- [`Fixed`](RangeStrategy::Fixed) splits the input evenly and hands out a fixed
  sequential range of items to each worker thread,
- [`WorkStealing`](RangeStrategy::WorkStealing) starts with the fixed
  distribution, but lets each worker thread steal items from others once it is
  done processing its items.

Note: In work-stealing mode, each thread processes an arbitrary subset of items
in arbitrary order, meaning that the reduction operation must be both
[commutative](https://en.wikipedia.org/wiki/Commutative_property) and
[associative](https://en.wikipedia.org/wiki/Associative_property) to yield a
deterministic result (in contrast to the standard library's
[`Iterator`](std::iter::Iterator) trait that processes items in order).
Fortunately, a lot of common operations are commutative and associative, but be
mindful of this.

Recommendation: If your pipeline is performing roughly the same amont of work
for each item, you should probably use the [`Fixed`](RangeStrategy::Fixed)
strategy, to avoid paying the synchronization cost of work-stealing. This is
especially true if the amount of work per item is small (e.g. some simple
arithmetic operations). If the amoung of work per item is highly variable and/or
large, you should probably use the [`WorkStealing`](RangeStrategy::WorkStealing)
strategy (e.g. parsing strings, processing files).

### CPU pinning

Paralight allows pinning each worker thread to one CPU, on platforms that
support it. For now, this is implemented for platforms whose
[`target_os`](https://doc.rust-lang.org/reference/conditional-compilation.html#target_os)
is among `android`, `dragonfly`, `freebsd` and `linux` (platforms that support
`libc::sched_setaffinity()` via the
[`nix` crate](https://crates.io/crates/nix)).

Paralight offers three policies in the [`CpuPinningPolicy`](CpuPinningPolicy)
enum:

- [`No`](CpuPinningPolicy::No) doesn't pin worker threads to CPUs,
- [`IfSupported`](CpuPinningPolicy::IfSupported) attempts to pin each worker
  thread to a distinct CPU on supported platforms, but proceeds without pinning
  if running on an unsupported platform or if the pinning function fails,
- [`Always`](CpuPinningPolicy::Always) pins each worker thread to a distinct
  CPU, panicking if the platform isn't supported or if the pinning function
  returns an error.

Whether CPU pinning is useful or detrimental depends on your workload. If you're
processing the same data over and over again (e.g. calling
[`par_iter()`](iter::IntoParallelRefSource::par_iter) multiple times on the same
data), CPU pinning can help ensure that each subset of the data is always
processed on the same CPU core and stays fresh in the lower-level per-core
caches, speeding up memory accesses. This however depends on the amount of data:
if it's too large, it may not fit in per-core caches anyway.

If your program is not running alone on your machine but is competing with other
programs, CPU pinning may be detrimental, as a worker thread will be blocked
whenever its required core is used by another program, even if another core is
free and other worker threads are done (especially with the
[`Fixed`](RangeStrategy::Fixed) strategy). This of course depends on how the
scheduler works on your OS.

## Limitations

With the [`WorkStealing`](RangeStrategy::WorkStealing) strategy, inputs with
more than [`u32::MAX`](u32::MAX) elements are currently not supported.

```rust,should_panic
use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, WithThreadPool};
use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

let pool_builder = ThreadPoolBuilder {
    num_threads: ThreadCount::AvailableParallelism,
    range_strategy: RangeStrategy::WorkStealing,
    cpu_pinning: CpuPinningPolicy::No,
};

let sum = pool_builder.scope(
    |mut thread_pool| {
        let input = vec![0u8; 5_000_000_000];
        input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .reduce(|| 0, |x, y| x + y)
    },
);
assert_eq!(sum, 0);
```

## Debugging

Two optional features are available if you want to debug performance.

- `log`, based on the [`log` crate](https://crates.io/crates/log) prints basic
  information about inter-thread synchronization: thread creation/shutdown, when
  each thread starts/finishes a computation, etc.
- `log_parallelism` prints detailed traces about which items are processed by
  which thread, and work-stealing statistics (e.g. how many times work was
  stolen among threads).

Note that in any case neither the input items nor the resulting computation are
logged. Only the _indices_ of the items in the input may be present in the logs.
If you're concerned that these indices leak too much information about your
data, you need to make sure that you depend on Paralight with the `log` and
`log_parallelism` features disabled.

## Disclaimer

This is not an officially supported Google product.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

This software is distributed under the terms of both the
[MIT license](LICENSE-MIT) and the
[Apache License (Version 2.0)](LICENSE-APACHE).

See [`LICENSE`](LICENSE) for details.
