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

This library allows you to distribute computation over slices (and other
*indexed* sources) among multiple threads. Each thread processes a subset of the
items, and a final step reduces the outputs from all threads into a single
result.

```rust
use paralight::iter::{
    IntoParallelRefSource, IntoParallelRefMutSource, ParallelIteratorExt, ParallelSourceExt,
    ZipableSource,
};
use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

// Create a thread pool with the given parameters.
let mut thread_pool = ThreadPoolBuilder {
    num_threads: ThreadCount::AvailableParallelism,
    range_strategy: RangeStrategy::WorkStealing,
    cpu_pinning: CpuPinningPolicy::No,
}
.build();

// Compute the sum of a slice.
let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let sum = input
    .par_iter()
    .with_thread_pool(&mut thread_pool)
    .sum::<i32>();
assert_eq!(sum, 5 * 11);

// Add slices together.
let mut output = [0; 10];
let left = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let right = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20];

(output.par_iter_mut(), left.par_iter(), right.par_iter())
    .zip_eq()
    .with_thread_pool(&mut thread_pool)
    .for_each(|(out, &a, &b)| *out = a + b);

assert_eq!(output, [12, 14, 16, 18, 20, 22, 24, 26, 28, 30]);
```

Paralight currently supports inputs that are a combination of slices and ranges,
but can be extended to support other sources as long as they are *indexed*. This
is done via the [`ParallelSource`](iter::ParallelSource) and
[`IntoParallelSource`](iter::IntoParallelSource) traits.

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

## Using a thread pool

To create parallel pipelines, be mindful that the
[`with_thread_pool()`](iter::ParallelSourceExt::with_thread_pool) function takes
the thread pool by mutable reference [`&mut`](reference). This is a deliberate
design choice because only one pipeline can be run at a time on a given thread
pool.

To release the resources (i.e. the worker threads) created by a
[`ThreadPool`](ThreadPool), simply [`drop()`](drop) it.

If you want to create a global thread pool, you will have to wrap it in a
[`Mutex`](std::sync::Mutex) (or other suitable synchronization primitive) and
manually lock it to obtain a suitable `&mut ThreadPool`. You can for example
combine a mutex with the [`LazyLock`](std::sync::LazyLock) pattern.

```rust,no_run
use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
use paralight::{
    CpuPinningPolicy, RangeStrategy, ThreadPool, ThreadCount, ThreadPoolBuilder,
};
use std::ops::DerefMut;
use std::sync::{LazyLock, Mutex};

// A static thread pool protected by a mutex.
static THREAD_POOL: LazyLock<Mutex<ThreadPool>> = LazyLock::new(|| {
    Mutex::new(
        ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::WorkStealing,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build(),
    )
});

let items = (0..100).collect::<Vec<_>>();
let sum = items
    .par_iter()
    .with_thread_pool(THREAD_POOL.lock().unwrap().deref_mut())
    .sum::<i32>();
assert_eq!(sum, 99 * 50);
```

However, if you wrap a thread pool in a mutex like this, be mindful of potential
panics or deadlocks if you try to run several nested parallel iterators on the
same thread pool!

This limitation isn't specific to Paralight though, this happens for any usage
of a [`Mutex`](std::sync::Mutex) that you try to lock recursively while already
acquired.

This pitfall is the reason why Paralight doesn't provide an implicit global
thread pool.

```rust,no_run
# use paralight::iter::{IntoParallelRefSource, ParallelIteratorExt, ParallelSourceExt};
# use paralight::{
#     CpuPinningPolicy, RangeStrategy, ThreadPool, ThreadCount, ThreadPoolBuilder,
# };
# use std::ops::DerefMut;
# use std::sync::{LazyLock, Mutex};
#
# static THREAD_POOL: LazyLock<Mutex<ThreadPool>> = LazyLock::new(|| {
#     Mutex::new(
#         ThreadPoolBuilder {
#             num_threads: ThreadCount::AvailableParallelism,
#             range_strategy: RangeStrategy::WorkStealing,
#             cpu_pinning: CpuPinningPolicy::No,
#         }
#         .build(),
#     )
# });
let matrix = (0..100)
    .map(|i| (0..100).map(|j| i + j).collect::<Vec<_>>())
    .collect::<Vec<_>>();

let sum = matrix
    .par_iter()
    // Lock the mutex on the outer loop (over the rows).
    .with_thread_pool(THREAD_POOL.lock().unwrap().deref_mut())
    .map(|row| {
        row.par_iter()
            // ⚠️ Trying to lock the mutex again here will panic or deadlock!
            .with_thread_pool(THREAD_POOL.lock().unwrap().deref_mut())
            .sum::<i32>()
    })
    .sum::<i32>();

// ⚠️ This statement is never reached due to the panic/deadlock!
assert_eq!(sum, 990_000);
```

## Limitations

With the [`WorkStealing`](RangeStrategy::WorkStealing) strategy, inputs with
more than [`u32::MAX`](u32::MAX) elements are currently not supported.

```rust,should_panic
use paralight::iter::{IntoParallelSource, ParallelIteratorExt, ParallelSourceExt};
use paralight::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPoolBuilder};

let mut thread_pool = ThreadPoolBuilder {
    num_threads: ThreadCount::AvailableParallelism,
    range_strategy: RangeStrategy::WorkStealing,
    cpu_pinning: CpuPinningPolicy::No,
}
.build();

let _sum = (0..5_000_000_000_usize)
    .into_par_iter()
    .with_thread_pool(&mut thread_pool)
    .sum::<usize>();
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

## Experimental nightly APIs

Some experimental APIs are available under the `nightly` Cargo feature, for
users who compile with a
[nightly](https://rust-lang.github.io/rustup/concepts/channels.html#working-with-nightly-rust)
Rust toolchain. As the underlying implementation is based on
[experimental features](https://doc.rust-lang.org/unstable-book/) of the Rust
language, these APIs are provided without guarantee and may break at any time
when a new nightly toolchain is released.

## Disclaimer

This is not an officially supported Google product.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

This software is distributed under the terms of both the
[MIT license](LICENSE-MIT) and the
[Apache License (Version 2.0)](LICENSE-APACHE).

See [`LICENSE`](LICENSE) for details.
