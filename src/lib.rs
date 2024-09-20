// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![doc = include_str!("../README.md")]
#![forbid(
    missing_docs,
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::multiple_unsafe_ops_per_block
)]
#![cfg_attr(not(test), forbid(clippy::undocumented_unsafe_blocks))]

mod macros;
mod range;
mod thread_pool;
mod util;

pub use thread_pool::{RangeStrategy, ThreadAccumulator, ThreadPool, ThreadPoolBuilder};

#[cfg(test)]
mod test {
    use super::*;
    use std::num::NonZeroUsize;

    /// Example of accumulator that computes a sum of integers.
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

    /// Example of accumulator that computes a sum of integers, but panics on
    /// one input.
    struct SumAccumulatorOnePanic;

    impl ThreadAccumulator<u64, u64> for SumAccumulatorOnePanic {
        type Accumulator<'a> = u64;

        fn init(&self) -> u64 {
            0
        }

        fn process_item(&self, accumulator: &mut u64, _index: usize, x: &u64) {
            if *x == 0 {
                panic!("arithmetic panic");
            } else {
                *accumulator += *x;
            }
        }

        fn finalize(&self, accumulator: u64) -> u64 {
            accumulator
        }
    }

    /// Example of accumulator that computes a sum of integers, but panics on
    /// some inputs.
    struct SumAccumulatorSomePanics;

    impl ThreadAccumulator<u64, u64> for SumAccumulatorSomePanics {
        type Accumulator<'a> = u64;

        fn init(&self) -> u64 {
            0
        }

        fn process_item(&self, accumulator: &mut u64, _index: usize, x: &u64) {
            if *x % 123 == 0 {
                panic!("arithmetic panic");
            } else {
                *accumulator += *x;
            }
        }

        fn finalize(&self, accumulator: u64) -> u64 {
            accumulator
        }
    }

    /// Example of accumulator that computes a sum of integers, but panics on
    /// many inputs.
    struct SumAccumulatorManyPanics;

    impl ThreadAccumulator<u64, u64> for SumAccumulatorManyPanics {
        type Accumulator<'a> = u64;

        fn init(&self) -> u64 {
            0
        }

        fn process_item(&self, accumulator: &mut u64, _index: usize, x: &u64) {
            if *x % 2 == 0 {
                panic!("arithmetic panic");
            } else {
                *accumulator += *x;
            }
        }

        fn finalize(&self, accumulator: u64) -> u64 {
            accumulator
        }
    }

    macro_rules! expand_tests {
        ( $range_strategy:expr, ) => {};
        ( $range_strategy:expr, $case:ident, $( $others:tt )* ) => {
            #[test]
            fn $case() {
                $crate::test::$case($range_strategy);
            }

            expand_tests!($range_strategy, $($others)*);
        };
        ( $range_strategy:expr, $case:ident => fail($msg:expr), $( $others:tt )* ) => {
            #[test]
            #[should_panic(expected = $msg)]
            fn $case() {
                $crate::test::$case($range_strategy);
            }

            expand_tests!($range_strategy, $($others)*);
        };
    }

    macro_rules! parallelism_tests {
        ( $mod:ident, $range_strategy:expr, $( $tests:tt )* ) => {
            mod $mod {
                use super::*;

                expand_tests!($range_strategy, $($tests)*);
            }
        };
    }

    macro_rules! all_parallelism_tests {
        ( $mod:ident, $range_strategy:expr ) => {
            parallelism_tests!(
                $mod,
                $range_strategy,
                test_sum_integers,
                test_sum_twice,
                test_one_panic => fail("A worker thread panicked!"),
                test_some_panics => fail("A worker thread panicked!"),
                test_many_panics => fail("A worker thread panicked!"),
                test_fn_once,
                test_local_sum,
                test_several_inputs,
            );
        };
    }

    all_parallelism_tests!(fixed, RangeStrategy::Fixed);
    all_parallelism_tests!(work_stealing, RangeStrategy::WorkStealing);

    #[cfg(not(miri))]
    const INPUT_LEN: u64 = 100_000;
    #[cfg(miri)]
    const INPUT_LEN: u64 = 1000;

    fn test_sum_integers(range_strategy: RangeStrategy) {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(
            || SumAccumulator,
            |thread_pool| {
                thread_pool
                    .process_inputs(&input)
                    .reduce(|a, b| a + b)
                    .unwrap()
            },
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_sum_twice(range_strategy: RangeStrategy) {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let (sum1, sum2) = pool_builder.scope(
            || SumAccumulator,
            |thread_pool| {
                // The same input can be processed multiple times on the thread pool.
                let sum1 = thread_pool
                    .process_inputs(&input)
                    .reduce(|a, b| a + b)
                    .unwrap();
                let sum2 = thread_pool
                    .process_inputs(&input)
                    .reduce(|a, b| a + b)
                    .unwrap();
                (sum1, sum2)
            },
        );
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_one_panic(range_strategy: RangeStrategy) {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(
            || SumAccumulatorOnePanic,
            |thread_pool| {
                thread_pool
                    .process_inputs(&input)
                    .reduce(|a, b| a + b)
                    .unwrap()
            },
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_some_panics(range_strategy: RangeStrategy) {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(
            || SumAccumulatorSomePanics,
            |thread_pool| {
                thread_pool
                    .process_inputs(&input)
                    .reduce(|a, b| a + b)
                    .unwrap()
            },
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_many_panics(range_strategy: RangeStrategy) {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(
            || SumAccumulatorManyPanics,
            |thread_pool| {
                thread_pool
                    .process_inputs(&input)
                    .reduce(|a, b| a + b)
                    .unwrap()
            },
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_fn_once(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        // The scope should accept FnOnce() parameter. We test it with a closure that
        // captures and consumes a non-Copy type.
        let token = Box::new(());
        pool_builder.scope(|| SumAccumulator, |_| drop(token));
    }

    fn test_local_sum(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(
            || SumAccumulator,
            |thread_pool| {
                // The input can be local.
                let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
                thread_pool
                    .process_inputs(&input)
                    .reduce(|a, b| a + b)
                    .unwrap()
            },
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_several_inputs(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let (sum1, sum2) = pool_builder.scope(
            || SumAccumulator,
            |thread_pool| {
                // Several inputs can be used successively.
                let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
                let sum1 = thread_pool
                    .process_inputs(&input)
                    .reduce(|a, b| a + b)
                    .unwrap();

                let input = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
                let sum2 = thread_pool
                    .process_inputs(&input)
                    .reduce(|a, b| a + b)
                    .unwrap();

                (sum1, sum2)
            },
        );
        // n(n+1)/2
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);
        // 2n(2n+1)/2
        assert_eq!(sum2, INPUT_LEN * (2 * INPUT_LEN + 1));
    }
}
