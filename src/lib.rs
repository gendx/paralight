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

pub use thread_pool::{RangeStrategy, ThreadPool, ThreadPoolBuilder};

#[cfg(test)]
mod test {
    use super::*;
    use std::num::NonZeroUsize;

    macro_rules! expand_tests {
        ( $range_strategy:expr, ) => {};
        ( $range_strategy:expr, $case:ident, $( $others:tt )* ) => {
            #[test]
            fn $case() {
                for _ in 0..ITERATIONS {
                    $crate::test::$case($range_strategy);
                }
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
                test_one_panic => fail("worker thread(s) panicked!"),
                test_some_panics => fail("worker thread(s) panicked!"),
                test_many_panics => fail("worker thread(s) panicked!"),
                test_fn_once,
                test_local_sum,
                test_several_inputs,
                test_several_functions,
                test_several_accumulators,
                test_several_input_types,
            );
        };
    }

    all_parallelism_tests!(fixed, RangeStrategy::Fixed);
    all_parallelism_tests!(work_stealing, RangeStrategy::WorkStealing);

    #[cfg(not(miri))]
    const INPUT_LEN: u64 = 100_000;
    #[cfg(miri)]
    const INPUT_LEN: u64 = 1000;

    #[cfg(not(miri))]
    const ITERATIONS: u64 = 100;
    #[cfg(miri)]
    const ITERATIONS: u64 = 1;

    fn test_sum_integers(range_strategy: RangeStrategy) {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(|thread_pool| {
            thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| *acc += *x,
                |acc| acc,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_sum_twice(range_strategy: RangeStrategy) {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let (sum1, sum2) = pool_builder.scope(|thread_pool| {
            // The same input can be processed multiple times on the thread pool.
            let sum1 = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| *acc += *x,
                |acc| acc,
                |a, b| a + b,
            );
            let sum2 = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| *acc += *x,
                |acc| acc,
                |a, b| a + b,
            );
            (sum1, sum2)
        });
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_one_panic(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(|thread_pool| {
            // The input can be local.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| {
                    if *x == 0 {
                        panic!("arithmetic panic");
                    } else {
                        *acc += *x;
                    }
                },
                |acc| acc,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_some_panics(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(|thread_pool| {
            // The input can be local.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| {
                    if *x % 123 == 0 {
                        panic!("arithmetic panic");
                    } else {
                        *acc += *x;
                    }
                },
                |acc| acc,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_many_panics(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(|thread_pool| {
            // The input can be local.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| {
                    if *x % 2 == 0 {
                        panic!("arithmetic panic");
                    } else {
                        *acc += *x;
                    }
                },
                |acc| acc,
                |a, b| a + b,
            )
        });
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
        pool_builder.scope::<(), ()>(|_| drop(token));
    }

    fn test_local_sum(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let sum = pool_builder.scope(|thread_pool| {
            // The input can be local.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| *acc += *x,
                |acc| acc,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_several_inputs(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let (sum1, sum2) = pool_builder.scope(|thread_pool| {
            // Several inputs can be used successively.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            let sum1 = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| *acc += *x,
                |acc| acc,
                |a, b| a + b,
            );

            let input = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
            let sum2 = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| *acc += *x,
                |acc| acc,
                |a, b| a + b,
            );

            (sum1, sum2)
        });
        // n(n+1)/2
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);
        // 2n(2n+1)/2
        assert_eq!(sum2, INPUT_LEN * (2 * INPUT_LEN + 1));
    }

    fn test_several_functions(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let (sum, sum_squares) = pool_builder.scope(|thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // Several functions can be computed successively.
            let sum = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| *acc += *x,
                |acc| acc,
                |a, b| a + b,
            );

            let sum_squares = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, &x| *acc += x * x,
                |acc| acc,
                |a, b| a + b,
            );

            (sum, sum_squares)
        });
        // n(n+1)/2
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
        // n(n+1)(2n+1) / 6
        assert_eq!(
            sum_squares,
            INPUT_LEN * (INPUT_LEN + 1) * (2 * INPUT_LEN + 1) / 6
        );
    }

    fn test_several_accumulators(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let (sum1, sum2) = pool_builder.scope(|thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // Several accumulator types can be used successively.
            let sum1 = thread_pool.pipeline(
                &input,
                || 0u32,
                |acc, _, &x| *acc += x as u32,
                |acc| acc as u64,
                |a, b| (a + b) & 0xffff_ffff,
            );

            let sum2 = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| *acc += *x,
                |acc| acc,
                |a, b| a + b,
            );

            (sum1, sum2)
        });
        assert_eq!(sum1, (INPUT_LEN * (INPUT_LEN + 1) / 2) & 0xffff_ffff);
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_several_input_types(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: NonZeroUsize::try_from(4).unwrap(),
            range_strategy,
        };
        let (sum, sum_lengths) = pool_builder.scope(|thread_pool| {
            // Several input types can be used successively.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            let sum = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, &x| *acc += x,
                |acc| acc,
                |a, b| a + b,
            );

            let input = (0..=INPUT_LEN)
                .map(|x| format!("{x}"))
                .collect::<Vec<String>>();
            let sum_lengths = thread_pool.pipeline(
                &input,
                || 0usize,
                |acc, _, x| *acc += x.len(),
                |acc| acc as u64,
                |a, b| a + b,
            );

            (sum, sum_lengths)
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_lengths, expected_sum_lengths(INPUT_LEN));
    }

    const fn expected_sum_lengths(max: u64) -> u64 {
        if max < 10 {
            max + 1
        } else {
            let mut expected = 10;
            let mut ipow = 10;
            let mut i = 1;
            loop {
                if max / ipow >= 10 {
                    expected += 9 * ipow * (i + 1);
                } else if max >= ipow {
                    expected += (max + 1 - ipow) * (i + 1);
                    break;
                } else {
                    break;
                }
                ipow *= 10;
                i += 1;
                if i == 10 {
                    break;
                }
            }
            expected
        }
    }

    #[test]
    fn test_expected_sum_lengths() {
        assert_eq!(expected_sum_lengths(0), 1);
        assert_eq!(expected_sum_lengths(9), 10);
        assert_eq!(expected_sum_lengths(10), 10 + 2);
        assert_eq!(expected_sum_lengths(99), 10 + 90 * 2);
        assert_eq!(expected_sum_lengths(100), 10 + 90 * 2 + 3);
        assert_eq!(expected_sum_lengths(999), 10 + 90 * 2 + 900 * 3);
        assert_eq!(expected_sum_lengths(1000), 10 + 90 * 2 + 900 * 3 + 4);
        assert_eq!(expected_sum_lengths(9999), 10 + 90 * 2 + 900 * 3 + 9000 * 4);
    }
}
