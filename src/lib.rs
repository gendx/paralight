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
#![cfg_attr(all(test, feature = "nightly_tests"), feature(negative_impls))]

mod core;
pub mod iter;
mod macros;

pub use core::{CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPool, ThreadPoolBuilder};

#[cfg(test)]
mod test {
    use super::*;
    use crate::iter::{IntoParallelIterator, ParallelIterator, ParallelIteratorExt};
    use std::cell::Cell;
    use std::collections::HashSet;
    use std::rc::Rc;
    use std::sync::Mutex;

    macro_rules! expand_tests {
        ( $range_strategy:expr, ) => {};
        ( $range_strategy:expr, $( #[ $attrs:meta ] )* $case:ident, $( $others:tt )* ) => {
            $( #[$attrs] )*
            #[test]
            fn $case() {
                for _ in 0..ITERATIONS {
                    $crate::test::$case($range_strategy);
                }
            }

            expand_tests!($range_strategy, $($others)*);
        };
        ( $range_strategy:expr, $( #[ $attrs:meta ] )* $case:ident => fail($msg:expr), $( $others:tt )* ) => {
            $( #[$attrs] )*
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
                test_empty_input,
                test_several_inputs,
                test_several_functions,
                test_several_accumulators,
                test_several_input_types,
                test_several_pipelines,
                test_capture_environment,
                #[cfg(feature = "nightly_tests")]
                test_non_send_functions,
                #[cfg(feature = "nightly_tests")]
                test_non_send_input,
                test_non_sync_output,
                #[cfg(feature = "nightly_tests")]
                test_non_send_accumulator,
                test_non_sync_accumulator,
                test_non_send_sync_accumulator,
                test_local_lifetime_functions,
                test_local_lifetime_input,
                test_local_lifetime_output,
                test_local_lifetime_accumulator,
                test_adaptor_par_iter,
                test_adaptor_filter,
                test_adaptor_filter_map,
                test_adaptor_for_each,
                test_adaptor_map,
                test_adaptor_reduce,
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
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            thread_pool.pipeline(&input, || 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b)
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_sum_twice(range_strategy: RangeStrategy) {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let (sum1, sum2) = pool_builder.scope(|mut thread_pool| {
            // The same input can be processed multiple times on the thread pool.
            let sum1 =
                thread_pool.pipeline(&input, || 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);
            let sum2 =
                thread_pool.pipeline(&input, || 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);
            (sum1, sum2)
        });
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_one_panic(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            // The input can be local.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, x| {
                    if *x == 0 {
                        panic!("arithmetic panic");
                    } else {
                        acc + *x
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
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            // The input can be local.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, x| {
                    if *x % 123 == 0 {
                        panic!("arithmetic panic");
                    } else {
                        acc + *x
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
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            // The input can be local.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, x| {
                    if *x % 2 == 0 {
                        panic!("arithmetic panic");
                    } else {
                        acc + *x
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
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        // The scope should accept FnOnce() parameter. We test it with a closure that
        // captures and consumes a non-Copy type.
        let token = Box::new(());
        pool_builder.scope(|_| drop(token));
    }

    fn test_local_sum(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            // The input can be local.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(&input, || 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b)
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_empty_input(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            // The input can be empty.
            thread_pool.pipeline(
                &[],
                || 0,
                |acc, _, x: &u64| acc + *x,
                |acc| acc,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, 0);
    }

    fn test_several_inputs(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let (sum1, sum2) = pool_builder.scope(|mut thread_pool| {
            // Several inputs can be used successively.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            let sum1 =
                thread_pool.pipeline(&input, || 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);

            let input = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
            let sum2 =
                thread_pool.pipeline(&input, || 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);

            (sum1, sum2)
        });
        // n(n+1)/2
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);
        // 2n(2n+1)/2
        assert_eq!(sum2, INPUT_LEN * (2 * INPUT_LEN + 1));
    }

    fn test_several_functions(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let (sum, sum_squares) = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // Several functions can be computed successively.
            let sum =
                thread_pool.pipeline(&input, || 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);

            let sum_squares = thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, &x| acc + x * x,
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
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let (sum1, sum2) = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // Several accumulator types can be used successively.
            let sum1 = thread_pool.pipeline(
                &input,
                || 0u32,
                |acc, _, &x| acc + x as u32,
                |acc| acc as u64,
                |a, b| (a + b) & 0xffff_ffff,
            );

            let sum2 = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, x| acc + *x,
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
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let (sum, sum_lengths) = pool_builder.scope(|mut thread_pool| {
            // Several input types can be used successively.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            let sum = thread_pool.pipeline(
                &input,
                || 0u64,
                |acc, _, &x| acc + x,
                |acc| acc,
                |a, b| a + b,
            );

            let input = (0..=INPUT_LEN)
                .map(|x| format!("{x}"))
                .collect::<Vec<String>>();
            let sum_lengths = thread_pool.pipeline(
                &input,
                || 0usize,
                |acc, _, x| acc + x.len(),
                |acc| acc as u64,
                |a, b| a + b,
            );

            (sum, sum_lengths)
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_lengths, expected_sum_lengths(INPUT_LEN));
    }

    fn test_several_pipelines(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let (sum, sum_pairs) = pool_builder.scope(|mut thread_pool| {
            // Pipelines with different types can be used successively.
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            let sum =
                thread_pool.pipeline(&input, || 0, |acc, _, &x| acc + x, |acc| acc, |a, b| a + b);

            let input = (0..=INPUT_LEN)
                .map(|i| (2 * i, 2 * i + 1))
                .collect::<Vec<(u64, u64)>>();
            let sum_pairs = thread_pool.pipeline(
                &input,
                || (0, 0),
                |(a, b), _, &(x, y)| (a + x, b + y),
                |acc| acc,
                |(a, b), (x, y)| (a + x, b + y),
            );

            (sum, sum_pairs)
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(
            sum_pairs,
            (
                INPUT_LEN * (INPUT_LEN + 1),
                (INPUT_LEN + 1) * (INPUT_LEN + 1)
            )
        );
    }

    fn test_capture_environment(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        // Pipelines can capture from the environment outside of the scoped thread pool.
        let zero = 0;
        let zero_ref = &zero;
        let one = 1;
        let one_ref = &one;
        let sum = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(
                &input,
                || *zero_ref,
                |acc, _, x| acc + *x * *one_ref,
                |acc| acc * *one_ref,
                |a, b| (a + b) * *one_ref,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly_tests")]
    struct NotSend(u64);
    #[cfg(feature = "nightly_tests")]
    impl NotSend {
        fn get(&self) -> u64 {
            self.0
        }
    }
    #[cfg(feature = "nightly_tests")]
    impl !Send for NotSend {}

    #[cfg(feature = "nightly_tests")]
    fn test_non_send_functions(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let (sum1, sum2, sum3, sum4) = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // Non-Send functions can be used in the pipeline.
            let init = NotSend(0);
            let sum1 = thread_pool.pipeline(
                &input,
                move || init.get(),
                |acc, _, x| acc + *x,
                |acc| acc,
                |a, b| a + b,
            );

            let one = NotSend(1);
            let sum2 = thread_pool.pipeline(
                &input,
                || 0,
                move |acc, _, x| acc + *x * one.get(),
                |acc| acc,
                |a, b| a + b,
            );

            let one = NotSend(1);
            let sum3 = thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, x| acc + *x,
                move |acc| acc * one.get(),
                |a, b| a + b,
            );

            let zero = NotSend(0);
            let sum4 = thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, x| acc + *x,
                |acc| acc,
                move |a, b| a + b + zero.get(),
            );

            (sum1, sum2, sum3, sum4)
        });
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum4, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly_tests")]
    fn test_non_send_input(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            // A non-Send input can be used in the pipeline.
            let input = (0..=INPUT_LEN).map(NotSend).collect::<Vec<NotSend>>();
            thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, x| acc + x.get(),
                |acc| acc,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_non_sync_output(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // A non-Sync output can be used in the pipeline.
            thread_pool
                .pipeline(
                    &input,
                    || 0,
                    |acc, _, x| acc + *x,
                    Cell::new,
                    |a, b| Cell::new(a.get() + b.get()),
                )
                .get()
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly_tests")]
    fn test_non_send_accumulator(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // A non-Send accumulator can be used in the pipeline.
            thread_pool.pipeline(
                &input,
                || NotSend(0),
                |acc, _, x| NotSend(acc.0 + *x),
                |acc| acc.0,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_non_sync_accumulator(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(move |mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // A non-Sync accumulator can be used in the pipeline.
            thread_pool.pipeline(
                &input,
                || Cell::new(0),
                |mut acc, _, x| {
                    *acc.get_mut() += *x;
                    acc
                },
                |acc| acc.get(),
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_non_send_sync_accumulator(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(move |mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // A neither Send nor Sync accumulator can be used in the pipeline.
            thread_pool.pipeline(
                &input,
                || Rc::new(0),
                |mut acc, _, x| {
                    *Rc::get_mut(&mut acc).unwrap() += *x;
                    acc
                },
                |acc| *acc,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_local_lifetime_functions(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let (sum1, sum2, sum3, sum4) = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // The pipeline functions can borrow local values (and therefore have a local
            // lifetime).
            let zero = 0;
            let one = 1;

            let sum1 = thread_pool.pipeline(
                &input,
                || zero,
                |acc, _, x| acc + *x,
                |acc| acc,
                |a, b| a + b,
            );

            let sum2 = thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, x| acc + *x * one,
                |acc| acc,
                |a, b| a + b,
            );

            let sum3 = thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, x| acc + *x,
                |acc| acc * one,
                |a, b| a + b,
            );

            let sum4 = thread_pool.pipeline(
                &input,
                || 0,
                |acc, _, x| acc + *x,
                |acc| acc,
                |a, b| a + b + zero,
            );

            (sum1, sum2, sum3, sum4)
        });
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum4, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_local_lifetime_input(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            // The pipeline input can borrow local values (and therefore have a local
            // lifetime).
            let token = ();
            let token_ref = &token;
            let input = (0..=INPUT_LEN)
                .map(|x| (x, token_ref))
                .collect::<Vec<(u64, _)>>();
            thread_pool.pipeline(&input, || 0, |acc, _, x| acc + x.0, |acc| acc, |a, b| a + b)
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_local_lifetime_output(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            // The pipeline output can borrow local values (and therefore have a local
            // lifetime).
            let token = ();
            let token_ref = &token;
            thread_pool
                .pipeline(
                    &input,
                    || 0,
                    |acc, _, x| acc + *x,
                    |acc| (acc, token_ref),
                    |a, b| (a.0 + b.0, a.1),
                )
                .0
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_local_lifetime_accumulator(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            // The pipeline accumulator can borrow local values (and therefore have a local
            // lifetime).
            let token = ();
            let token_ref = &token;
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            thread_pool.pipeline(
                &input,
                || (0, token_ref),
                |acc, _, x| (acc.0 + *x, acc.1),
                |acc| acc.0,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_par_iter(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            input.par_iter(&mut thread_pool).pipeline(
                || 0,
                |acc, _, x| acc + *x,
                |acc| acc,
                |a, b| a + b,
            )
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_filter(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            input
                .par_iter(&mut thread_pool)
                .filter(|&&x| x % 2 == 0)
                .pipeline(|| 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b)
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN / 2 + 1) / 2);
    }

    fn test_adaptor_filter_map(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            input
                .par_iter(&mut thread_pool)
                .filter_map(|&x| if x % 2 == 0 { Some(x * 3) } else { None })
                .pipeline(|| 0, |acc, _, x| acc + x, |acc| acc, |a, b| a + b)
        });
        assert_eq!(sum, 3 * INPUT_LEN * (INPUT_LEN / 2 + 1) / 2);
    }

    fn test_adaptor_for_each(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let set = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            let set = Mutex::new(HashSet::new());
            input.par_iter(&mut thread_pool).for_each(|&x| {
                set.lock().unwrap().insert(x);
            });
            set.into_inner().unwrap()
        });
        assert_eq!(set, (0..=INPUT_LEN).collect());
    }

    fn test_adaptor_map(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let (sum1, sum2, sum3) = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();

            let sum1 = input.par_iter(&mut thread_pool).map(|&x| x * 42).pipeline(
                || 0,
                |acc, _, x| acc + x,
                |acc| acc,
                |a, b| a + b,
            );

            let sum2 = input
                .par_iter(&mut thread_pool)
                .map(|&x| x * 6)
                .map(|x| x * 7)
                .pipeline(|| 0, |acc, _, x| acc + x, |acc| acc, |a, b| a + b);

            let sum3 = input
                .par_iter(&mut thread_pool)
                // Mapping to a non-Send non-Sync type is fine, as the item stays on the same thread
                // and isn't shared with other threads.
                .map(|&x| Rc::new(x))
                .pipeline(|| 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);

            (sum1, sum2, sum3)
        });
        assert_eq!(sum1, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum2, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_reduce(range_strategy: RangeStrategy) {
        let pool_builder = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::IfSupported,
        };
        let sum = pool_builder.scope(|mut thread_pool| {
            let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
            input
                .par_iter(&mut thread_pool)
                .map(|&x| x)
                .reduce(|| 0, |x, y| x + y)
        });
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
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
