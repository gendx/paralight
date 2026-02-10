# Paralight: a lightweight parallelism library for indexed structures

[![Crate](https://img.shields.io/crates/v/paralight.svg?logo=rust)](https://crates.io/crates/paralight)
[![Documentation](https://img.shields.io/docsrs/paralight/0.0.11?logo=rust)](https://docs.rs/paralight/0.0.11/)
[![Minimum Rust 1.80.0](https://img.shields.io/crates/msrv/paralight/0.0.11.svg?logo=rust&color=orange)](https://releases.rs/docs/1.80.0/)
[![Lines of Code](https://www.aschey.tech/tokei/github/gendx/paralight?category=code&branch=0.0.11)](https://github.com/gendx/paralight/tree/0.0.11)
[![Dependencies](https://deps.rs/crate/paralight/0.0.11/status.svg)](https://deps.rs/crate/paralight/0.0.11)
[![License](https://img.shields.io/crates/l/paralight/0.0.11.svg)](https://github.com/gendx/paralight/blob/0.0.11/LICENSE)
[![Codecov](https://codecov.io/gh/gendx/paralight/branch/0.0.11/graph/badge.svg)](https://app.codecov.io/gh/gendx/paralight/tree/0.0.11)
[![Build Status](https://github.com/gendx/paralight/actions/workflows/build.yml/badge.svg?branch=0.0.11)](https://github.com/gendx/paralight/actions/workflows/build.yml)
[![Test Status](https://github.com/gendx/paralight/actions/workflows/tests.yml/badge.svg?branch=0.0.11)](https://github.com/gendx/paralight/actions/workflows/tests.yml)

This library allows you to distribute computation over *indexed* sources
([slices](slice), [ranges](std::ops::Range), [`Vec`](Vec), etc.) among multiple
threads. It aims to uphold the highest standards of documentation, testing and
safety, see the [FAQ](#faq) below.

It is designed to be as lightweight as possible, following the principles
outlined in the blog post
[*Optimization adventures: making a parallel Rust workload 10x faster with (or without) Rayon*](https://gendignoux.com/blog/2024/11/18/rust-rayon-optimized.html#a-hand-rolled-replacement-of-rayon).
Benchmarks on a real-world use case can be seen
[here](https://gendignoux.com/blog/2024/12/02/rust-data-oriented-design.html#results).

```rust
use paralight::prelude::*;

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

Paralight supports various indexed sources out-of-the-box ([slices](slice),
[ranges](std::ops::Range), etc.), and can be extended to other types via the
[`ParallelSource`](iter::ParallelSource) trait, together with the conversion
traits ([`IntoParallelSource`](iter::IntoParallelSource),
[`IntoParallelRefSource`](iter::IntoParallelRefSource) and
[`IntoParallelRefMutSource`](iter::IntoParallelRefMutSource)).

## Thread pool configuration

The [`ThreadPoolBuilder`](threads::ThreadPoolBuilder) provides an explicit way
to configure your thread pool, giving you fine-grained control over performance
for your workload. There is no default, which is deliberate because the suitable
parameters depend on your workload.

### Number of worker threads

Paralight allows you to specify the number of worker threads to spawn in a
thread pool with the [`ThreadCount`](threads::ThreadCount) enum:

- [`AvailableParallelism`](threads::ThreadCount::AvailableParallelism) uses the
  number of threads returned by the standard library's
  [`available_parallelism()`](std::thread::available_parallelism) function,
- [`Count(_)`](threads::ThreadCount::Count) uses the specified number of
  threads, which must be non-zero.

For convenience, [`ThreadCount`](threads::ThreadCount) implements the
[`TryFrom<usize>`](TryFrom) trait to create a
[`Count(_)`](threads::ThreadCount::Count) instance, validating that the given
number of threads is not zero.

**Recommendation:** It depends. While
[`AvailableParallelism`](threads::ThreadCount::AvailableParallelism) may be a
good default, it usually returns twice the number of CPU cores (at least on
Intel) to account for
[hyper-threading](https://en.wikipedia.org/wiki/Hyper-threading). Whether this
is optimal or not depends on your workload, for example whether it's compute
bound or memory bound, whether a single thread can saturate the resources of one
core or not, etc. Generally, the long list of caveats mentioned in the
documentation of [`available_parallelism()`](std::thread::available_parallelism)
applies.

On some workloads, hyper-threading doesn't provide a performance boost over
using only one thread per core, because two hyper-threads would compete on
resources on the core they share (e.g. memory caches). In this case, using half
of what [`available_parallelism()`](std::thread::available_parallelism) returns
can reduce contention and perform better.

If your program is not running alone on your machine but is competing with other
programs, using too many threads can also be detrimental to the overall
performance of your system.

### Work-stealing strategy

Paralight offers two strategies in the [`RangeStrategy`](threads::RangeStrategy)
enum to distribute computation among threads:

- [`Fixed`](threads::RangeStrategy::Fixed) splits the input evenly and hands out
  a fixed sequential range of items to each worker thread,
- [`WorkStealing`](threads::RangeStrategy::WorkStealing) starts with the fixed
  distribution, but lets each worker thread steal items from others once it is
  done processing its items.

**Recommendation:** If your pipeline is performing roughly the same amont of
work for each item, you should probably use the
[`Fixed`](threads::RangeStrategy::Fixed) strategy, to avoid paying the
synchronization cost of work-stealing. This is especially true if the amount of
work per item is small (e.g. some simple arithmetic operations). If the amoung
of work per item is highly variable and/or large, you should probably use the
[`WorkStealing`](threads::RangeStrategy::WorkStealing) strategy (e.g. parsing
strings, processing files).

**Note:** In work-stealing mode, each thread processes an arbitrary subset of
items in arbitrary order, meaning that a reduction operation must be both
[commutative](https://en.wikipedia.org/wiki/Commutative_property) and
[associative](https://en.wikipedia.org/wiki/Associative_property) to yield a
deterministic result (in contrast to the standard library's
[`Iterator`](std::iter::Iterator) trait that processes items in sequential
order). Fortunately, a lot of common operations are commutative and associative,
but be mindful of this.

```rust,should_panic
use paralight::prelude::*;

let mut thread_pool = ThreadPoolBuilder {
    num_threads: ThreadCount::AvailableParallelism,
    range_strategy: RangeStrategy::WorkStealing,
    cpu_pinning: CpuPinningPolicy::No,
}
.build();

let s = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    .par_iter()
    .with_thread_pool(&mut thread_pool)
    .map(|c: &char| c.to_string())
    .reduce(String::new, |mut a: String, b: String| {
        a.push_str(&b);
        a
    });
// ⚠️ There is no guarantee that this check passes. In practice, `s` contains any permutation
// of the input, such as "fgdebachij".
assert_eq!(s, "abcdefghij");

// This makes sure the example panics anyway if the permutation is (by luck) the identity.
panic!("Congratulations, you won the lottery and the assertion passed this time!");
```

### CPU pinning

Paralight allows pinning each worker thread to one CPU, on platforms that
support it. For now, this is implemented for platforms whose
[`target_os`](https://doc.rust-lang.org/reference/conditional-compilation.html#target_os)
is among `android`, `dragonfly`, `freebsd`, `linux` (platforms that support
`libc::sched_setaffinity()` via the [`nix` crate](https://crates.io/crates/nix))
and `windows` (using
[`SetThreadAffinityMask()`](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-setthreadaffinitymask)
via the [`windows-sys` crate](https://crates.io/crates/windows-sys)).

Paralight offers three policies in the
[`CpuPinningPolicy`](threads::CpuPinningPolicy) enum:

- [`No`](threads::CpuPinningPolicy::No) doesn't pin worker threads to CPUs,
- [`IfSupported`](threads::CpuPinningPolicy::IfSupported) attempts to pin each
  worker thread to a distinct CPU on supported platforms, but proceeds without
  pinning if running on an unsupported platform or if the pinning function
  fails,
- [`Always`](threads::CpuPinningPolicy::Always) pins each worker thread to a
  distinct CPU, panicking if the platform isn't supported or if the pinning
  function returns an error.

**Recommendation:** Whether CPU pinning is useful or detrimental depends on your
workload. If you're processing the same data over and over again (e.g. calling
[`par_iter()`](iter::IntoParallelRefSource::par_iter) multiple times on the same
data), CPU pinning can help ensure that each subset of the data is always
processed on the same CPU core and stays fresh in the lower-level per-core
caches, speeding up memory accesses. This however depends on the amount of data:
if it's too large, it may not fit in per-core caches anyway.

If your program is not running alone on your machine but is competing with other
programs, CPU pinning may be detrimental, as a worker thread will be blocked
whenever its required core is used by another program, even if another core is
free and other worker threads are done (especially with the
[`Fixed`](threads::RangeStrategy::Fixed) strategy). This of course depends on
how the scheduler works on your OS.

## Using a thread pool

To create parallel pipelines, be mindful that the
[`with_thread_pool()`](iter::ParallelSourceExt::with_thread_pool) function takes
a [`ThreadPool`](threads::ThreadPool) by mutable reference [`&mut`](reference).
This is a deliberate design choice because only one pipeline can be run at a
time on a given Paralight thread pool (for more flexible options, see
["Bringing your own thread pool"](#bringing-your-own-thread-pool) below).

To release the resources (i.e. the worker threads) created by a
[`ThreadPool`](threads::ThreadPool), simply [`drop()`](drop) it.

If you want to create a global thread pool, you will have to wrap it in a
[`Mutex`](std::sync::Mutex) (or other suitable synchronization primitive) and
manually lock it to obtain a suitable `&mut ThreadPool`. You can for example
combine a mutex with the [`LazyLock`](std::sync::LazyLock) pattern.

```rust,no_run
use paralight::prelude::*;
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
# use paralight::prelude::*;
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

## Bringing your own thread pool

As an alternative to the provided [`ThreadPool`](threads::ThreadPool)
implementation, you can use Paralight with any thread pool that implements the
[`GenericThreadPool`](iter::GenericThreadPool) interface, via the
[`with_thread_pool()`](iter::ParallelSourceExt::with_thread_pool) adaptor.

Note that the [`GenericThreadPool`](iter::GenericThreadPool) trait is marked as
`unsafe` due to the requirements that your implementation must uphold.

If you don't need the default [`ThreadPool`](threads::ThreadPool)
implementation, you can disable the `default-thread-pool` feature of Paralight
and still benefit from all the iterator API.

### Rayon thread pools

For convenience, the [`RayonThreadPool`](threads::RayonThreadPool) wrapper
around [Rayon](https://docs.rs/rayon) is available under the `rayon` feature,
and implements [`GenericThreadPool`](iter::GenericThreadPool).

However, note that this backend isn't tested with Miri nor ThreadSanitizer,
because Rayon is broken with them.

This wrapper allows you to control how many tasks are run on Rayon's thread
pool. It is recommended to match the number of threads in the pool (as well as
the available parallelism), to have one Paralight task per thread (and per
available CPU core). In that case, performance should be similar to vanilla
Paralight. This is of course only a guideline: as usual benchmark and profile
your code to know which configuration is optimal.

```rust
# // TODO: Enable Miri once supported by Rayon and its dependencies:
# // https://github.com/crossbeam-rs/crossbeam/issues/1181.
# #[cfg(all(feature = "rayon", not(miri)))]
# {
use paralight::prelude::*;

let thread_pool = RayonThreadPool::new_global(
    ThreadCount::try_from(rayon_core::current_num_threads())
        .expect("Paralight cannot operate with 0 threads"),
    RangeStrategy::WorkStealing,
);

let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let sum = input.par_iter().with_thread_pool(&thread_pool).sum::<i32>();
assert_eq!(sum, 5 * 11);
# }
```

## Limitations

With the [`WorkStealing`](threads::RangeStrategy::WorkStealing) strategy, inputs
with more than [`u32::MAX`](u32::MAX) elements are currently not supported.

```rust,should_panic
use paralight::prelude::*;

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

## FAQ

### Documentation

All public APIs of Paralight are documented, which is enforced by the
`forbid(missing_docs)` lint. The aim is to have at least one example per API,
naturally
[tested via `rustdoc`](https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html).

### Testing

Paralight is thoroughly tested, with
[code coverage](https://app.codecov.io/gh/gendx/paralight/) as close to 100% as
possible.

The testing strategy combines
[`rustdoc` examples](https://doc.rust-lang.org/rustdoc/write-documentation/documentation-tests.html),
top-level stress tests and unit tests on the most critical components.

### Safety

Paralight aims to use as little `unsafe` code as possible. As a first measure,
the following lints are enabled throughout the code base, to make sure each use
of an `unsafe` API is explained and each new `unsafe` function documents its
pre- and post-conditions.

```rust
#![forbid(
    missing_docs,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::multiple_unsafe_ops_per_block,
    clippy::undocumented_unsafe_blocks,
)]
# //! Here are some docs
```

Additionally, [extensive testing](#testing) is conducted with
[ThreadSanitizer](https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html#threadsanitizer)
and [Miri](https://github.com/rust-lang/miri). Vanilla Paralight is 100%
compatible with them, including Miri's detection of memory leaks.

This multi-layered approach to safety is crucial given how complex mixing memory
safety with multi-threading is, and it has indeed
[caught a bug](https://github.com/gendx/paralight/commit/59c995672634aead96a4d977fe1fcab1e0faa9a5)
during development.

Note: The [Rayon thread pool integration](#rayon-thread-pools) is unfortunately
not tested under Miri, as Rayon in general
[triggers Miri errors](https://github.com/crossbeam-rs/crossbeam/issues/1181).
Likewise, Rayon local thread pools
[aren't compatible](https://github.com/rayon-rs/rayon/issues/1275) with
ThreadSanitizer.

### Use of `unsafe` code

As mentioned, Paralight uses as little `unsafe` code as possible. Here is the
list of places where `unsafe` is needed.

- Lifetime-erasure in
  [threads/thread_pool/util.rs](https://github.com/gendx/paralight/blob/main/src/threads/thread_pool/util.rs).
  The goal is essentially to share an `Arc<Mutex<&'a T>>` between the main
  threads and worker threads where `T` contains a description of a parallel
  pipeline. The difficulty is that the lifetime `'a` is only valid for a limited
  scope (for example, a parallel iterator may capture local variables by
  reference). Even though synchronization is in place to make sure the worker
  threads only access this pipeline during `'a`, there is no way to write a safe
  Rust type for `Arc<Mutex<&'a T>>` where the lifetime `'a` changes over time
  (the same mutex is reused for successive pipelines sent to Paralight).
  Glossing over the details, a type akin to `Arc<Mutex<&'static T>>` is used
  instead (i.e. the lifetime is marked `'static`), with `unsafe` code to rescope
  the `'static` lifetime to a local `'a` as needed. [`Send`](Send) and
  [`Sync`](Sync) implementations are also provided (when sound) on this wrapper
  type.
- The [`SliceParallelSource`](iter::SliceParallelSource) API in
  [iter/source/slice.rs](https://github.com/gendx/paralight/blob/main/src/iter/source/slice.rs)
  uses [`slice::get_unchecked()`](slice::get_unchecked) as it guides the
  compiler to better optimize the code. In particular, missed vectorized loops
  [were observed](https://github.com/gendx/paralight/issues/12) without it.
- The [`MutSliceParallelSource`](iter::MutSliceParallelSource) API in
  [iter/source/slice.rs](https://github.com/gendx/paralight/blob/main/src/iter/source/slice.rs).
  The goal is to provide parallel iterators that produce mutable references
  `&mut T` to items of a mutable slice `&mut [T]`. Sharing a `&mut [T]` with all
  the threads wouldn't work as it would violate Rust's aliasing rules. Using the
  [`slice::split_at_mut()`](slice::split_at_mut) API would not work either, as
  it isn't known in advance where a split will occur nor which worker thread
  will consume which item (due to work stealing). An approach that decomposes
  the slice into a pointer-length pair is used instead, making it possible to
  share the raw pointer with all the worker threads.
- Similarly, the [`VecParallelSource`](iter::VecParallelSource) API in
  [iter/source/vec.rs](https://github.com/gendx/paralight/blob/main/src/iter/source/vec.rs)
  provides parallel iterators that consume a `Vec<T>`. This is achieved by
  decomposing the vector into a pointer-length-capacity triple, and sharing the
  base pointer with all the worker threads so that they can consume items of
  type `T` (via [`std::ptr::read()`](std::ptr::read)). Additionally, the
  original `Vec<T>` allocation is released when the iterator is dropped (to
  avoid memory leaks), which involves reconstructing it from the
  pointer-allocation pair (via [`Vec::from_raw_parts()`](Vec::from_raw_parts)).
- Likewise, the [`ArrayParallelSource`](iter::ArrayParallelSource) API in
  [iter/source/array.rs](https://github.com/gendx/paralight/blob/main/src/iter/source/array.rs)
  provides parallel iterators that consume a `[T; N]`. The situation is similar
  to `Vec<T>`, except that the items aren't allocated on the heap behind a
  pointer, but directly in the array. This involves a more careful combination
  of wrapper types. Note that a
  [prior implementation](https://github.com/gendx/paralight/commit/8c5fab6c52e2495bd89ac4f1fef78a18470d7335)
  was quickly
  [reverted](https://github.com/gendx/paralight/commit/59c995672634aead96a4d977fe1fcab1e0faa9a5)
  due to being unsound, highlighting once again the importance of code coverage
  and the effectiveness of [Miri](https://github.com/rust-lang/miri).
- Windows API calls are used to set thread affinity in
  [threads/thread_pool/mod.rs](https://github.com/gendx/paralight/blob/main/src/threads/thread_pool/mod.rs)
  when requested.
- Lastly, the definition of the [`SourceDescriptor`](iter::SourceDescriptor)
  trait in
  [iter/source/mod.rs](https://github.com/gendx/paralight/blob/main/src/iter/source/mod.rs)
  has `unsafe` methods because it requires the caller to pass each index once
  and only once. Indeed, the safety of the previously mentioned iterator sources
  (mutable slice, vector, array) assumes a correct calling pattern. The
  `SourceDescriptor` trait is public (so that dependents of Paralight can define
  their own sources of items), so these `unsafe` functions leak in the public
  API. Internally, this causes `unsafe` blocks each time the trait is
  implemented, and the associated safety comments are a good opportunity to
  check correctness.
- Symmetrically, [`GenericThreadPool`](iter::GenericThreadPool) is an `unsafe`
  trait, implemented for `&mut ThreadPool` in
  [threads/thread_pool/mod.rs](https://github.com/gendx/paralight/blob/main/src/threads/thread_pool/mod.rs)
  and for `&RayonThreadPool` in
  [threads/rayon.rs](https://github.com/gendx/paralight/blob/main/src/threads/rayon.rs).
  This in turn relies on _correctness_ of the (safe) work-stealing
  implementation in
  [core/range.rs](https://github.com/gendx/paralight/blob/main/src/core/range.rs),
  that is still missing a formal proof.

And that's all the `unsafe` code there is!

## Disclaimer

This is not an officially supported Google product.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

Note that Paralight is still an early stage project, with many design changes as
new features are added. Therefore it is highly recommended to file a
[GitHub issue](https://github.com/gendx/paralight/issues) for discussion before
submitting a pull request, unless you're making a trivial change or bug fix.

## License

This software is distributed under the terms of both the
[MIT license](LICENSE-MIT) and the
[Apache License (Version 2.0)](LICENSE-APACHE).

See [`LICENSE`](LICENSE) for details.
