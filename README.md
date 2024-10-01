# Paralight: a lightweight parallelism library for indexed structures

[![Crate](https://img.shields.io/crates/v/paralight.svg?logo=rust)](https://crates.io/crates/paralight)
[![Documentation](https://img.shields.io/docsrs/paralight?logo=rust)](https://docs.rs/paralight)
[![Minimum Rust 1.75.0](https://img.shields.io/badge/rust-1.75.0%2B-orange.svg?logo=rust)](https://releases.rs/docs/1.75.0/)
[![Dependencies](https://deps.rs/crate/paralight/0.0.2/status.svg)](https://deps.rs/crate/paralight/0.0.2)
[![Lines of Code](https://tokei.rs/b1/github/gendx/paralight?style=flat&category=code&branch=0.0.2)](https://github.com/gendx/paralight)
[![Build Status](https://github.com/gendx/paralight/actions/workflows/build.yml/badge.svg?branch=0.0.2)](https://github.com/gendx/paralight/actions/workflows/build.yml)
[![Test Status](https://github.com/gendx/paralight/actions/workflows/tests.yml/badge.svg?branch=0.0.2)](https://github.com/gendx/paralight/actions/workflows/tests.yml)

This library allows you to distribute computation over slices among multiple
threads. Each thread processes a subset of the items, and a final step reduces
the outputs from all threads into a single result.

```rust
use paralight::{RangeStrategy, ThreadPoolBuilder};
use std::num::NonZeroUsize;

// Define thread pool parameters.
let pool_builder = ThreadPoolBuilder {
    num_threads: NonZeroUsize::try_from(4).unwrap(),
    range_strategy: RangeStrategy::WorkStealing,
};

// Create a scoped thread pool.
let sum = pool_builder.scope(
    |mut thread_pool| {
        // Compute the sum of a slice.
        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        thread_pool.pipeline(
            &input,
            /* init */ || 0u64,
            /* process_item */ |acc, _, x| *acc += *x,
            /* finalize */ |acc| acc,
            /* reduce */ |a, b| a + b,
        )
    },
);
assert_eq!(sum, 5 * 11);
```

Note: In principle, Paralight could be extended to support other inputs than
slices as long as they are *indexed*, but for now only slices are supported.
Come back to check when future versions are published!

## Work-stealing strategy

Paralight offers two strategies to distribute computation among threads:
- [`RangeStrategy::Fixed`] splits the input evenly and hands out a fixed
  sequential range of items to each thread,
- [`RangeStrategy::WorkStealing`] starts with the fixed distribution, but lets
  each thread steal items from others once it is done computing its items.

Note: In work-stealing mode, each thread processes an arbitrary subset of items
in arbitrary order, meaning that the reduction operation must be both
commutative and associative to yield a deterministic result (in contrast with
the standard library's `Iterator` trait that processes items in order).
Fortunately, a lot of common operations are commutative and associative, but be
mindful of this.

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

This software is distributed under the terms of both the [MIT
license](LICENSE-MIT) and the [Apache License (Version 2.0)](LICENSE-APACHE).

See [`LICENSE`](LICENSE) for details.
