# Paralight: a lightweight parallelism library for indexed structures

This library allows you to distribute computation over slices among multiple
threads. Each thread processes a subset of the items, and a final step reduces
the outputs from all threads into a single result.

```rust
use paralight::ThreadPool;

let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let num_threads = NonZeroUsize::try_from(4).unwrap();

let sum = std::thread::scope(|scope| {
    // Initialize a thread pool attached to the given input and accumulator (see
    // below).
    let thread_pool = ThreadPool::new(
        scope,
        num_threads,
        RangeStrategy::WorkStealing,
        &input,
        || SumAccumulator,
    );
    // Compute the sum over the inputs.
    thread_pool.process_inputs().reduce(|a, b| a + b).unwrap()
});
assert_eq!(sum, 5 * 11);

// Example of accumulator that computes a sum of integers.
struct SumAccumulator;

impl ThreadAccumulator<u64, u64> for SumAccumulator {
    type Accumulator<'a> = u64;

    fn init(&self) -> u64 {
        0
    }

    fn process_item(&self, accumulator: &mut u64, _index: usize, x: &u64) {
        *accumulator += *x;
    }

    fn finalize(&self, accumulator: u64) -> u64 {
        accumulator
    }
}
```

Note: In principle, Paralight could be extended to support other inputs than
slices as long as they are *indexed*, but for now only slices are supported.
Come back to check when future versions are published!

Paralight offers two strategies to distribute computation among threads:
- "fixed" splits the input evenly and hands out a fixed sequential range of
  items to each thread,
- "work-stealing" starts with the fixed distribution, but lets each thread steal
  items from others once it is done computing its items.

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
logged. Only the indices of the items in the input may be present. If you're
concerned about that leaking information about your data into log, you need to
make sure that you depend on Paralight with the `log` and `log_parallelism`
features disabled.

## Disclaimer

This is not an officially supported Google product.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

This software is distributed under the terms of both the MIT license and the
Apache License (Version 2.0).

See [`LICENSE`](LICENSE) for details.
