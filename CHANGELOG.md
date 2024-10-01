# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

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
