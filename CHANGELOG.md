# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.0.3 - 2024-10-22

### Added

- A parallel iterator trait implemented for slices, with 14 adaptors:
  `cloned()`, `copied()`, `filter()`, `filter_map()`, `for_each()`, `inspect()`,
  `map()`, `max()`, `max_by()`, `max_by_key()`, `min()`, `min_by()`,
  `min_by_key()` and `reduce`.
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

## 0.0.2 - 2024-01-01

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
