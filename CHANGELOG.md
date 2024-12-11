# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.0.5 - 2024-12-11

### Added

- 12 find-any-like adaptors on `ParallelIteratorExt`, that short-circuit once a
  match is found: `all()`, `any()`, `eq()`, `eq_by_key()`, `eq_by_keys()`,
  `find_any()`, `find_map_any()`, `ne()`, `ne_by_key()`, `ne_by_keys()`,
  `try_for_each()` and `try_for_each_init()`. A nightly API is available for
  `try_for_each()` and `try_for_each_init()` for all types implementing the
  `std::ops::Try` trait.
- 10 find-first-like adaptors on `ParallelIteratorExt`, that restrict the search
  to previous items once a match is found: `cmp()`, `cmp_by()`, `cmp_by_key()`,
  `cmp_by_keys()`, `find_first()`, `find_map_first()`, `partial_cmp()`,
  `partial_cmp_by()`, `partial_cmp_by_key()` and `partial_cmp_by_keys()`.
- 4 additional adaptors on `ParallelIteratorExt`: `for_each_init()`,
  `map_init()`, `product()` and `sum()`.
- To support these, the `Accumulator` trait and 3 methods on `ParallelIterator`:
  `iter_pipeline()`, `short_circuiting_pipeline()` and
  `upper_bounded_pipeline()`.
- The `step_by()` adaptor on `ParallelSourceExt`.
- The `num_threads()` getter on `ThreadPool`, to query the number of worker
  threads that have been spawned.
- A new `ParallelAdaptor` trait, that `Cloned`, `Copied`, `Filter`, `FilterMap`,
  `Inspect` and `Map` implement as an intermediate step before
  `ParallelIterator`.
- More recommendations for the thread pool configuration in the README.
- Tests with the thread sanitizer (TSAN).

### Changed

- The nightly APIs for `ParallelIterator` on ranges have been migrated to the
  latest version of the `std::iter::Step` trait.
- Slightly changed the format of logging statements.
- Internal synchronization primitives are now aligned to a cache line via the
  `CachePadded` wrapper provided by the `crossbeam-utils` crate.
- Micro-optimized the layout of the shared context with worker threads, using a
  single `Arc` instead of one `Arc` per field.
- Code coverage now also tracks coverage of the nightly APIs, and logging
  statements up to the debug level.
- Reduced the number of iterations in tests to make them complete faster.

## 0.0.4 - 2024-11-12

### Added

- Support for mutable slices (producing mutable references to the items) and
  ranges (producing integers) as parallel sources.
- The `ParallelSource` trait, as an intermediate interface between parallel
  sources and iterators, with 7 adaptors: `chain()`, `enumerate()`, `rev()`,
  `skip()`, `skip_exact()`, `take()` and `take_exact()`.
- 3 adaptors on tuples (up to 12 elements) of parallel sources, via the
  `ZipableSource` trait: `zip_eq()`, `zip_max()` and `zip_min()`.
- A `nightly` feature for experimental APIs available only with a Rust nightly
  toolchain.
- A benchmark for element-wise addition of slices.

### Changed

- Thread pools are now static rather than scoped: replaced the
  `ThreadPoolBuilder::scope()` function by `ThreadPoolBuilder::build()` and
  removed the `'scope` lifetime parameter of the corresponding `ThreadPool`
  struct.
- The `par_iter()` and similar methods don't take a thread pool parameter
  anymore: a thread pool is attached later via the new `with_thread_pool()`
  method.

### Removed

- The `IntoParallelIterator` trait, in favor of the new traits
  `IntoParallelSource`, `IntoParallelRefSource` and `IntoParallelRefMutSource`.

### Fixes

- The benchmarks now correctly compute the throughput (they were underestimating
  the size by one item beforehand).

## 0.0.3 - 2024-10-22

### Added

- A parallel iterator trait implemented for slices, with 14 adaptors:
  `cloned()`, `copied()`, `filter()`, `filter_map()`, `for_each()`, `inspect()`,
  `map()`, `max()`, `max_by()`, `max_by_key()`, `min()`, `min_by()`,
  `min_by_key()` and `reduce()`.
- A thread pool configuration to control whether worker threads are pinned to
  CPUs.

### Changed

- The `num_threads` configuration in `ThreadPoolBuilder` now also has an
  `AvailableParallelism` variant to use the number of threads reported by
  `std::thread::available_parallelism()`.
- The `ThreadPool` struct doesn't have an `'env` lifetime anymore, only a
  `'scope` lifetime.
- Tests for non-`Sync` types now use `Cell` rather than a synthetic type,
  allowing them to run on stable Rust.
- Tests now use the available parallelism (rather than a fixed number of
  threads) and don't pin worker threads to CPUs. Miri tests are configured to
  use 4 CPUs. Benchmarks still use a set number of threads and CPU pinning.

### Removed

- The `ThreadPool::pipeline()` function from the public API.

### Fixes

- Clarified that the input length cannot exceed `u32::MAX` in the `WorkStealing`
  configuration. Trying to use a larger input now panics rather than processing
  a smaller subset of items.
- Fixed possible integer overflows in `WorkStealing` mode if some intermediate
  results would exceed `u32::MAX`. This could notably have happened if the input
  length multiplied by `max(2, number of threads)` would exceed `u32::MAX`.

## 0.0.2 - 2024-10-01

### Added

- Support for running different pipelines on the same thread pool.
- Support for using a local input slice, and capturing local variables in the
  pipeline functions.
- Benchmarks of summing a slice of integers, compared with serial iterators and
  with [rayon](https://docs.rs/rayon), using [divan](https://docs.rs/divan) and
  [criterion](https://docs.rs/criterion).

### Changed

- The main API to create a thread pool. There is now a `ThreadPoolBuilder` from
  which to spawn a scoped thread pool.
- The main API to run a pipeline on a thread pool. The input slice and pipeline
  functions are not captured by the whole thread pool anymore, but are local
  parameters of the `pipeline()` function.

## 0.0.1 - 2024-09-17

- Initial version.
