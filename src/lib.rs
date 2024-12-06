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
#![cfg_attr(
    all(test, feature = "nightly_tests"),
    feature(negative_impls, coverage_attribute)
)]
#![cfg_attr(feature = "nightly", feature(step_trait, try_trait_v2))]
#![cfg_attr(docsrs, feature(doc_cfg))]

mod core;
pub mod iter;
mod macros;

pub use core::{
    Accumulator, CpuPinningPolicy, RangeStrategy, ThreadCount, ThreadPool, ThreadPoolBuilder,
};

#[cfg(test)]
mod test {
    use super::*;
    use crate::iter::{
        IntoParallelRefMutSource, IntoParallelRefSource, IntoParallelSource, ParallelIterator,
        ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    };
    use rand::Rng;
    use std::cell::Cell;
    use std::collections::HashSet;
    use std::rc::Rc;
    use std::sync::atomic::AtomicU64;
    #[cfg(all(not(miri), feature = "log"))]
    use std::sync::LazyLock;
    use std::sync::Mutex;

    #[cfg(all(not(miri), feature = "log"))]
    static ENV_LOGGER_INIT: LazyLock<()> = LazyLock::new(env_logger::init);

    macro_rules! expand_tests {
        ( $range_strategy:expr, ) => {};
        ( $range_strategy:expr, $( #[ $attrs:meta ] )* $case:ident, $( $others:tt )* ) => {
            $( #[$attrs] )*
            #[test]
            fn $case() {
                #[cfg(all(not(miri), feature = "log"))]
                LazyLock::force(&ENV_LOGGER_INIT);
                $crate::test::$case($range_strategy);
            }

            expand_tests!($range_strategy, $($others)*);
        };
        ( $range_strategy:expr, $( #[ $attrs:meta ] )* $case:ident => fail($msg:expr), $( $others:tt )* ) => {
            $( #[$attrs] )*
            #[test]
            #[should_panic(expected = $msg)]
            fn $case() {
                #[cfg(all(not(miri), feature = "log"))]
                LazyLock::force(&ENV_LOGGER_INIT);
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
                test_pipeline_sum_integers,
                test_pipeline_one_panic => fail("worker thread(s) panicked!"),
                test_pipeline_some_panics => fail("worker thread(s) panicked!"),
                test_pipeline_many_panics => fail("worker thread(s) panicked!"),
                test_pipeline_empty_input,
                test_pipelines_sum_twice,
                test_pipelines_several_inputs,
                test_pipelines_several_functions,
                test_pipelines_several_accumulators,
                test_pipelines_several_input_types,
                test_pipelines_several_types,
                #[cfg(feature = "nightly_tests")]
                test_pipeline_non_send_functions,
                #[cfg(feature = "nightly_tests")]
                test_pipeline_non_send_input,
                test_pipeline_non_sync_output,
                #[cfg(feature = "nightly_tests")]
                test_pipeline_non_send_accumulator,
                test_pipeline_non_sync_accumulator,
                test_pipeline_non_send_sync_accumulator,
                test_pipeline_local_lifetime_functions,
                test_pipeline_local_lifetime_input,
                test_pipeline_local_lifetime_output,
                test_pipeline_local_lifetime_accumulator,
                test_source_slice,
                #[cfg(feature = "nightly_tests")]
                test_source_slice_not_send,
                test_source_slice_mut,
                test_source_slice_mut_not_sync,
                test_source_range,
                test_source_range_backwards => fail("cannot iterate over a backward range"),
                #[cfg(feature = "nightly")]
                test_source_range_u64,
                #[cfg(feature = "nightly")]
                test_source_range_u128_too_large => fail("cannot iterate over a range with more than usize::MAX items"),
                test_source_range_inclusive,
                test_source_range_inclusive_backwards => fail("cannot iterate over a backward range"),
                test_source_range_inclusive_too_large => fail("cannot iterate over a range with more than usize::MAX items"),
                #[cfg(feature = "nightly")]
                test_source_range_inclusive_u64,
                #[cfg(feature = "nightly")]
                test_source_range_inclusive_u64_too_large => fail("cannot iterate over a range with more than usize::MAX items"),
                #[cfg(feature = "nightly")]
                test_source_range_inclusive_u128_too_large => fail("cannot iterate over a range with more than usize::MAX items"),
                test_source_adaptor_chain,
                test_source_adaptor_chain_overflow => fail("called chain() with sources that together produce more than usize::MAX items"),
                test_source_adaptor_enumerate,
                test_source_adaptor_rev,
                test_source_adaptor_skip,
                test_source_adaptor_skip_exact,
                test_source_adaptor_skip_exact_too_much => fail("called skip_exact() with more items than this source produces"),
                test_source_adaptor_step_by,
                test_source_adaptor_step_by_zero => fail("called step_by() with a step of zero"),
                test_source_adaptor_step_by_zero_empty => fail("called step_by() with a step of zero"),
                test_source_adaptor_take,
                test_source_adaptor_take_exact,
                test_source_adaptor_take_exact_too_much => fail("called take_exact() with more items than this source produces"),
                test_source_adaptor_zip_eq,
                test_source_adaptor_zip_eq_unequal => fail("called zip_eq() with sources of different lengths"),
                test_source_adaptor_zip_max,
                test_source_adaptor_zip_min,
                test_adaptor_all,
                test_adaptor_any,
                test_adaptor_cloned,
                test_adaptor_cmp,
                test_adaptor_cmp_by,
                test_adaptor_cmp_by_key,
                test_adaptor_cmp_by_keys,
                test_adaptor_copied,
                test_adaptor_eq,
                test_adaptor_eq_by_key,
                test_adaptor_eq_by_keys,
                test_adaptor_filter,
                test_adaptor_filter_map,
                test_adaptor_find_any,
                test_adaptor_find_first,
                test_adaptor_for_each,
                test_adaptor_for_each_init,
                test_adaptor_inspect,
                test_adaptor_map,
                test_adaptor_map_init,
                test_adaptor_max,
                test_adaptor_max_by,
                test_adaptor_max_by_key,
                test_adaptor_min,
                test_adaptor_min_by,
                test_adaptor_min_by_key,
                test_adaptor_ne,
                test_adaptor_ne_by_key,
                test_adaptor_ne_by_keys,
                test_adaptor_partial_cmp,
                test_adaptor_partial_cmp_by,
                test_adaptor_partial_cmp_by_key,
                test_adaptor_partial_cmp_by_keys,
                test_adaptor_product,
                test_adaptor_reduce,
                test_adaptor_sum,
                test_adaptor_try_for_each,
                #[cfg(feature = "nightly")]
                test_adaptor_try_for_each_option,
                test_adaptor_try_for_each_init,
                #[cfg(feature = "nightly")]
                test_adaptor_try_for_each_init_option,
            );
        };
    }

    all_parallelism_tests!(fixed, RangeStrategy::Fixed);
    all_parallelism_tests!(work_stealing, RangeStrategy::WorkStealing);

    #[cfg(not(miri))]
    const INPUT_LEN: u64 = 100_000;
    #[cfg(miri)]
    const INPUT_LEN: u64 = 1000;

    fn test_pipeline_sum_integers(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_empty_input(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // The input can be empty.
        let sum = thread_pool.pipeline(
            /* input_len = */ 0,
            || 0,
            |acc, _index| acc,
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum, 0);
    }

    fn test_pipeline_one_panic(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| {
                let x = input[index];
                if x == 0 {
                    panic!("arithmetic panic");
                } else {
                    acc + x
                }
            },
            |acc| acc,
            |a, b| a + b,
        );
    }

    fn test_pipeline_some_panics(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| {
                let x = input[index];
                if x % 123 == 0 {
                    panic!("arithmetic panic");
                } else {
                    acc + x
                }
            },
            |acc| acc,
            |a, b| a + b,
        );
    }

    fn test_pipeline_many_panics(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| {
                let x = input[index];
                if x % 2 == 1 {
                    panic!("arithmetic panic");
                } else {
                    acc + x
                }
            },
            |acc| acc,
            |a, b| a + b,
        );
    }

    fn test_pipelines_sum_twice(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // The same input can be processed multiple times on the thread pool.
        let sum1 = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum2 = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipelines_several_inputs(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // Several inputs can be used successively.
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum1 = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        // n(n+1)/2
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let input = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let sum2 = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        // 2n(2n+1)/2
        assert_eq!(sum2, INPUT_LEN * (2 * INPUT_LEN + 1));
    }

    fn test_pipelines_several_functions(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // Several functions can be computed successively.
        let sum = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        // n(n+1)/2
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum_squares = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| {
                let x = input[index];
                acc + x * x
            },
            |acc| acc,
            |a, b| a + b,
        );
        // n(n+1)(2n+1) / 6
        assert_eq!(
            sum_squares,
            INPUT_LEN * (INPUT_LEN + 1) * (2 * INPUT_LEN + 1) / 6
        );
    }

    fn test_pipelines_several_accumulators(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // Several accumulator types can be used successively.
        let sum1 = thread_pool.pipeline(
            input.len(),
            || 0u32,
            |acc, index| acc + input[index] as u32,
            |acc| acc as u64,
            |a, b| (a + b) & 0xffff_ffff,
        );
        assert_eq!(sum1, (INPUT_LEN * (INPUT_LEN + 1) / 2) & 0xffff_ffff);

        let sum2 = thread_pool.pipeline(
            input.len(),
            || 0u64,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipelines_several_input_types(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // Several input types can be used successively.
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = thread_pool.pipeline(
            input.len(),
            || 0u64,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let input = (0..=INPUT_LEN)
            .map(|x| format!("{x}"))
            .collect::<Vec<String>>();
        let sum_lengths = thread_pool.pipeline(
            input.len(),
            || 0usize,
            |acc, index| acc + input[index].len(),
            |acc| acc as u64,
            |a, b| a + b,
        );
        assert_eq!(sum_lengths, expected_sum_lengths(INPUT_LEN));
    }

    fn test_pipelines_several_types(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // Pipelines with different types can be used successively.
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let input = (0..=INPUT_LEN)
            .map(|i| (2 * i, 2 * i + 1))
            .collect::<Vec<(u64, u64)>>();
        let sum_pairs = thread_pool.pipeline(
            input.len(),
            || (0, 0),
            |(a, b), index| {
                let (x, y) = input[index];
                (a + x, b + y)
            },
            |acc| acc,
            |(a, b), (x, y)| (a + x, b + y),
        );
        assert_eq!(
            sum_pairs,
            (
                INPUT_LEN * (INPUT_LEN + 1),
                (INPUT_LEN + 1) * (INPUT_LEN + 1)
            )
        );
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
    fn test_pipeline_non_send_functions(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // Non-Send functions can be used in the pipeline.
        let init = NotSend(0);
        let sum1 = thread_pool.pipeline(
            input.len(),
            move || init.get(),
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let one = NotSend(1);
        let sum2 = thread_pool.pipeline(
            input.len(),
            || 0,
            {
                let input = &input;
                move |acc, index| acc + input[index] * one.get()
            },
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let one = NotSend(1);
        let sum3 = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            move |acc| acc * one.get(),
            |a, b| a + b,
        );
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let zero = NotSend(0);
        let sum4 = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc,
            move |a, b| a + b + zero.get(),
        );
        assert_eq!(sum4, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly_tests")]
    fn test_pipeline_non_send_input(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // A non-Send input can be used in the pipeline.
        let input = (0..=INPUT_LEN).map(NotSend).collect::<Vec<NotSend>>();
        let sum = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index].get(),
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_non_sync_output(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // A non-Sync output can be used in the pipeline.
        let sum = thread_pool
            .pipeline(
                input.len(),
                || 0,
                |acc, index| acc + input[index],
                Cell::new,
                |a, b| Cell::new(a.get() + b.get()),
            )
            .get();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly_tests")]
    fn test_pipeline_non_send_accumulator(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // A non-Send accumulator can be used in the pipeline.
        let sum = thread_pool.pipeline(
            input.len(),
            || NotSend(0),
            |acc, index| NotSend(acc.0 + input[index]),
            |acc| acc.0,
            |a, b| a + b,
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_non_sync_accumulator(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // A non-Sync accumulator can be used in the pipeline.
        let sum = thread_pool.pipeline(
            input.len(),
            || Cell::new(0),
            |mut acc, index| {
                *acc.get_mut() += input[index];
                acc
            },
            |acc| acc.get(),
            |a, b| a + b,
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_non_send_sync_accumulator(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // A neither Send nor Sync accumulator can be used in the pipeline.
        let sum = thread_pool.pipeline(
            input.len(),
            || Rc::new(0),
            |mut acc, index| {
                *Rc::get_mut(&mut acc).unwrap() += input[index];
                acc
            },
            |acc| *acc,
            |a, b| a + b,
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_local_lifetime_functions(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // The pipeline functions can borrow local values (and therefore have a local
        // lifetime).
        let zero = 0;
        let zero_ref = &zero;
        let one = 1;
        let one_ref = &one;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum1 = thread_pool.pipeline(
            input.len(),
            || *zero_ref,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum2 = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index] * *one_ref,
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum3 = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc * *one_ref,
            |a, b| a + b,
        );
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum4 = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index],
            |acc| acc,
            |a, b| a + b + *zero_ref,
        );
        assert_eq!(sum4, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_local_lifetime_input(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // The pipeline input can borrow local values (and therefore have a local
        // lifetime).
        let token = ();
        let token_ref = &token;

        let input = (0..=INPUT_LEN)
            .map(|x| (x, token_ref))
            .collect::<Vec<(u64, _)>>();
        let sum = thread_pool.pipeline(
            input.len(),
            || 0,
            |acc, index| acc + input[index].0,
            |acc| acc,
            |a, b| a + b,
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_local_lifetime_output(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // The pipeline output can borrow local values (and therefore have a local
        // lifetime).
        let token = ();
        let token_ref = &token;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = thread_pool
            .pipeline(
                input.len(),
                || 0,
                |acc, index| acc + input[index],
                |acc| (acc, token_ref),
                |a, b| (a.0 + b.0, a.1),
            )
            .0;
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_local_lifetime_accumulator(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // The pipeline accumulator can borrow local values (and therefore have a local
        // lifetime).
        let token = ();
        let token_ref = &token;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = thread_pool.pipeline(
            input.len(),
            || (0, token_ref),
            |acc, index| (acc.0 + input[index], acc.1),
            |acc| acc.0,
            |a, b| a + b,
        );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_slice(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, _, x| acc + x, |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly_tests")]
    fn test_source_slice_not_send(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).map(NotSend).collect::<Vec<NotSend>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, _, x| acc + x.get(), |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_slice_mut(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let mut values = (0..=INPUT_LEN).collect::<Vec<u64>>();
        values
            .par_iter_mut()
            .with_thread_pool(&mut thread_pool)
            .for_each(|x| *x *= 2);
        assert_eq!(values, (0..=INPUT_LEN).map(|x| x * 2).collect::<Vec<_>>());
    }

    fn test_source_slice_mut_not_sync(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let mut values = (0..=INPUT_LEN).map(Cell::new).collect::<Vec<Cell<u64>>>();
        values
            .par_iter_mut()
            .with_thread_pool(&mut thread_pool)
            .for_each(|x| x.set(x.get() * 2));
        assert_eq!(
            values,
            (0..=INPUT_LEN)
                .map(|x| Cell::new(x * 2))
                .collect::<Vec<_>>()
        );
    }

    fn test_source_range(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let sum = (0..INPUT_LEN as usize)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
        assert_eq!(sum, (INPUT_LEN as usize - 1) * INPUT_LEN as usize / 2);
    }

    #[allow(clippy::reversed_empty_ranges)]
    fn test_source_range_backwards(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        (10..0)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_u64(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let sum = (0..INPUT_LEN)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, (INPUT_LEN - 1) * INPUT_LEN / 2);
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_u128_too_large(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        (0u128..0x1_0000_0000_0000_0000)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u128>();
    }

    fn test_source_range_inclusive(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let sum = (0..=INPUT_LEN as usize)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
        assert_eq!(sum, INPUT_LEN as usize * (INPUT_LEN as usize + 1) / 2);
    }

    #[allow(clippy::reversed_empty_ranges)]
    fn test_source_range_inclusive_backwards(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        (10..=0)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
    }

    fn test_source_range_inclusive_too_large(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        (0..=usize::MAX)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_inclusive_u64(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let sum = (0..=INPUT_LEN)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_inclusive_u64_too_large(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        (0..=u64::MAX)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_inclusive_u128_too_large(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        (0u128..=0x1_0000_0000_0000_0000)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u128>();
    }

    fn test_source_adaptor_chain(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input1 = (0..INPUT_LEN / 2).collect::<Vec<u64>>();
        let input2 = (INPUT_LEN / 2..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input1
            .par_iter()
            .chain(input2.par_iter())
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_adaptor_chain_overflow(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        (0..usize::MAX)
            .into_par_iter()
            .chain((0..1).into_par_iter())
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
    }

    fn test_source_adaptor_enumerate(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum_squares = input
            .par_iter()
            .enumerate()
            .with_thread_pool(&mut thread_pool)
            .map(|(i, &x)| i as u64 * x)
            .sum::<u64>();
        assert_eq!(
            sum_squares,
            INPUT_LEN * (INPUT_LEN + 1) * (2 * INPUT_LEN + 1) / 6
        );
    }

    fn test_source_adaptor_rev(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .rev()
            .enumerate()
            .with_thread_pool(&mut thread_pool)
            .map(|(i, &x)| i as u64 * x)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN - 1) * (INPUT_LEN + 1) / 6);
    }

    fn test_source_adaptor_skip(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .skip(INPUT_LEN as usize / 2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, ((INPUT_LEN + 1) / 2) * ((3 * INPUT_LEN) / 2 + 1) / 2);

        let sum_empty = input
            .par_iter()
            .skip(2 * INPUT_LEN as usize)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum_empty, 0);
    }

    fn test_source_adaptor_skip_exact(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .skip_exact(INPUT_LEN as usize / 2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, ((INPUT_LEN + 1) / 2) * ((3 * INPUT_LEN) / 2 + 1) / 2);
    }

    fn test_source_adaptor_skip_exact_too_much(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        input
            .par_iter()
            .skip_exact(2 * INPUT_LEN as usize)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    fn test_source_adaptor_step_by(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let mut input = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let sum_by_1 = input
            .par_iter()
            .step_by(1)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum_by_1, INPUT_LEN * (2 * INPUT_LEN + 1));

        let sum_by_2 = input
            .par_iter()
            .step_by(2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum_by_2, INPUT_LEN * (INPUT_LEN + 1));

        input.truncate(2 * INPUT_LEN as usize);
        let sum_by_2 = input
            .par_iter()
            .step_by(2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum_by_2, (INPUT_LEN - 1) * INPUT_LEN);

        let sum_empty = []
            .par_iter()
            .step_by(2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum_empty, 0);
    }

    fn test_source_adaptor_step_by_zero(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        input
            .par_iter()
            .step_by(0)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    fn test_source_adaptor_step_by_zero_empty(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        [].par_iter()
            .step_by(0)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    fn test_source_adaptor_take(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .take(INPUT_LEN as usize / 2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, ((INPUT_LEN / 2) * (INPUT_LEN / 2 + 1)) / 2);

        let sum_all = input
            .par_iter()
            .take(2 * INPUT_LEN as usize)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum_all, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_adaptor_take_exact(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .take_exact(INPUT_LEN as usize / 2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, ((INPUT_LEN / 2) * (INPUT_LEN / 2 + 1)) / 2);
    }

    fn test_source_adaptor_take_exact_too_much(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        input
            .par_iter()
            .take_exact(2 * INPUT_LEN as usize)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    fn test_source_adaptor_zip_eq(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let (sum_left, sum_right) = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .map(|(&a, &b)| (a, b))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));

        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_adaptor_zip_eq_unequal(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN).collect::<Vec<u64>>();
        (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .map(|(&a, &b)| (a, b))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
    }

    fn test_source_adaptor_zip_max(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let (sum_left, sum_right) = (left.par_iter(), right.par_iter())
            .zip_max()
            .with_thread_pool(&mut thread_pool)
            .map(|(a, b)| (a.copied().unwrap(), b.copied().unwrap_or(0)))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));

        assert_eq!(sum_left, INPUT_LEN * (2 * INPUT_LEN + 1));
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_adaptor_zip_min(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let (sum_left, sum_right) = (left.par_iter(), right.par_iter())
            .zip_min()
            .with_thread_pool(&mut thread_pool)
            .map(|(&a, &b)| (a, b))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));

        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_all(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let all_even = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .all(|&x| x % 2 == 0);
        assert!(!all_even);

        let all_small = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .all(|&x| x <= INPUT_LEN);
        assert!(all_small);

        let all_large = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .all(|&x| x > INPUT_LEN);
        assert!(!all_large);

        let all_empty = []
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .all(|_: &u64| false);
        assert!(all_empty);
    }

    fn test_adaptor_any(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let any_even = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .any(|&x| x % 2 == 0);
        assert!(any_even);

        let any_small = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .any(|&x| x <= INPUT_LEN);
        assert!(any_small);

        let any_large = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .any(|&x| x > INPUT_LEN);
        assert!(!any_large);

        let any_empty = []
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .any(|_: &u64| true);
        assert!(!any_empty);
    }

    fn test_adaptor_cloned(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .cloned()
            .reduce(
                || Box::new(0u64),
                |mut x, y| {
                    *x += *y;
                    x
                },
            );
        assert_eq!(*sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_cmp(range_strategy: RangeStrategy) {
        use std::cmp::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let ordering = (input.par_iter(), input.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp();
        assert_eq!(ordering, Ordering::Equal);

        let left = (0..INPUT_LEN).collect::<Vec<u64>>();
        let right = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp();
        assert_eq!(ordering, Ordering::Less);

        let left = std::iter::once(INPUT_LEN)
            .chain(0..INPUT_LEN)
            .collect::<Vec<u64>>();
        let right = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp();
        assert_eq!(ordering, Ordering::Greater);
    }

    fn test_adaptor_cmp_by(range_strategy: RangeStrategy) {
        use std::cmp::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (0, i)).collect::<Vec<(u64, u64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp_by(|x, y| x.0.cmp(&y.1).then(x.1.cmp(&y.0)));
        assert_eq!(ordering, Ordering::Equal);

        let left = (0..=INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (1, i)).collect::<Vec<(u64, u64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp_by(|x, y| x.0.cmp(&y.1).then(x.1.cmp(&y.0)));
        assert_eq!(ordering, Ordering::Less);
    }

    fn test_adaptor_cmp_by_key(range_strategy: RangeStrategy) {
        use std::cmp::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (i, 1)).collect::<Vec<(u64, u64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp_by_key(|x| x.0);
        assert_eq!(ordering, Ordering::Equal);

        let left = (0..INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (1..=INPUT_LEN).map(|i| (i, 1)).collect::<Vec<(u64, u64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp_by_key(|x| x.0);
        assert_eq!(ordering, Ordering::Less);

        let left = std::iter::once(INPUT_LEN)
            .chain(0..INPUT_LEN)
            .map(|i| (i, 0))
            .collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (i, 1)).collect::<Vec<(u64, u64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp_by_key(|x| x.0);
        assert_eq!(ordering, Ordering::Greater);
    }

    fn test_adaptor_cmp_by_keys(range_strategy: RangeStrategy) {
        use std::cmp::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (1, i)).collect::<Vec<(u64, u64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp_by_keys(|x| x.0, |x| x.1);
        assert_eq!(ordering, Ordering::Equal);

        let left = (0..INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (1..=INPUT_LEN).map(|i| (1, i)).collect::<Vec<(u64, u64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp_by_keys(|x| x.0, |x| x.1);
        assert_eq!(ordering, Ordering::Less);

        let left = std::iter::once(INPUT_LEN)
            .chain(0..INPUT_LEN)
            .map(|i| (i, 0))
            .collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (1, i)).collect::<Vec<(u64, u64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .cmp_by_keys(|x| x.0, |x| x.1);
        assert_eq!(ordering, Ordering::Greater);
    }

    fn test_adaptor_copied(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .reduce(|| 0, |x, y| x + y);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_eq(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let equal = (input.par_iter(), input.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .eq();
        assert!(equal);

        let equal = (
            input.par_iter().take(INPUT_LEN as usize),
            input.par_iter().skip(1),
        )
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .eq();
        assert!(!equal);
    }

    fn test_adaptor_eq_by_key(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (i, 1)).collect::<Vec<(u64, u64)>>();
        let equal = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .eq_by_key(|x| x.0);
        assert!(equal);

        let equal = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .eq_by_key(|x| x.1);
        assert!(!equal);
    }

    fn test_adaptor_eq_by_keys(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (1, i)).collect::<Vec<(u64, u64)>>();
        let equal = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .eq_by_keys(|x| x.0, |y| y.1);
        assert!(equal);

        let equal = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .eq_by_keys(|x| x.1, |y| y.0);
        assert!(!equal);
    }

    fn test_adaptor_filter(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .filter(|&&x| x % 2 == 0)
            .pipeline(|| 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN / 2 + 1) / 2);
    }

    fn test_adaptor_filter_map(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .filter_map(|&x| if x % 2 == 0 { Some(x * 3) } else { None })
            .pipeline(|| 0, |acc, _, x| acc + x, |acc| acc, |a, b| a + b);
        assert_eq!(sum, 3 * INPUT_LEN * (INPUT_LEN / 2 + 1) / 2);
    }

    fn test_adaptor_find_any(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let first = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_any(|&x| x == 0);
        assert_eq!(first, Some(0));

        let last = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_any(|&x| x == INPUT_LEN);
        assert_eq!(last, Some(INPUT_LEN));

        let end = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_any(|&x| x == INPUT_LEN + 1);
        assert_eq!(end, None);

        let forty_two = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_any(|&x| x == 42);
        assert_eq!(forty_two, if INPUT_LEN >= 42 { Some(42) } else { None });

        let even = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_any(|&x| x % 2 == 0);
        assert!(even.unwrap() % 2 == 0);

        let empty = []
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_any(|_: &&u64| true);
        assert_eq!(empty, None);
    }

    fn test_adaptor_find_first(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let first = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_first(|_| true);
        assert_eq!(first, Some(0));

        let last = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_first(|&x| x >= INPUT_LEN);
        assert_eq!(last, Some(INPUT_LEN));

        let end = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_first(|&x| x > INPUT_LEN);
        assert_eq!(end, None);

        let forty_two = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_first(|&x| x >= 42);
        assert_eq!(forty_two, if INPUT_LEN >= 42 { Some(42) } else { None });

        let even = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_first(|&x| x % 2 == 0);
        assert_eq!(even, Some(0));

        let odd = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .find_first(|&x| x % 2 == 1);
        assert_eq!(odd, Some(1));

        let empty = []
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_first(|_: &&u64| true);
        assert_eq!(empty, None);
    }

    fn test_adaptor_for_each(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let set = Mutex::new(HashSet::new());
        input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .for_each(|&x| {
                set.lock().unwrap().insert(x);
            });
        let set = set.into_inner().unwrap();

        assert_eq!(set, (0..=INPUT_LEN).collect());
    }

    fn test_adaptor_for_each_init(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let mut values = (0..=INPUT_LEN).collect::<Vec<u64>>();
        values
            .par_iter_mut()
            .with_thread_pool(&mut thread_pool)
            .for_each_init(rand::thread_rng, |rng, x| {
                if rng.gen() {
                    *x *= 2
                };
            });

        let sum: u64 = values.iter().sum();
        assert!(sum >= INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert!(sum <= INPUT_LEN * (INPUT_LEN + 1));
    }

    fn test_adaptor_inspect(range_strategy: RangeStrategy) {
        use std::sync::atomic::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = AtomicU64::new(0);
        let max = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .inspect(|&x| {
                sum.fetch_add(x, Ordering::Relaxed);
            })
            .max();
        assert_eq!(max, Some(INPUT_LEN));

        let sum = sum.into_inner();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_map(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum1 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|&x| x * 42)
            .pipeline(|| 0, |acc, _, x| acc + x, |acc| acc, |a, b| a + b);
        assert_eq!(sum1, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum2 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|&x| x * 6)
            .map(|x| x * 7)
            .pipeline(|| 0, |acc, _, x| acc + x, |acc| acc, |a, b| a + b);
        assert_eq!(sum2, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum3 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            // Mapping to a non-Send non-Sync type is fine, as the item stays on the same thread
            // and isn't shared with other threads.
            .map(|&x| Rc::new(x))
            .pipeline(|| 0, |acc, _, x| acc + *x, |acc| acc, |a, b| a + b);
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_map_init(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .map_init(
                rand::thread_rng,
                |rng, &x| if rng.gen() { x * 2 } else { x * 3 },
            )
            .sum::<u64>();

        assert!(sum >= INPUT_LEN * (INPUT_LEN + 1));
        assert!(sum <= 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_max(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let mut input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let max = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .max();
        assert_eq!(max, Some(INPUT_LEN));

        input.truncate(1);
        let max_one = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .max();
        assert_eq!(max_one, Some(0));

        input.clear();
        let max_empty = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .max();
        assert_eq!(max_empty, None);
    }

    fn test_adaptor_max_by(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // Custom comparison function where even numbers are smaller than all odd
        // numbers.
        let mut input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let max = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .max_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));

        let last_odd = ((INPUT_LEN - 1) / 2) * 2 + 1;
        assert_eq!(max, Some(last_odd));

        input.truncate(1);
        let max_one = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .max_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(max_one, Some(0));

        input.clear();
        let max_empty = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .max_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(max_empty, None);
    }

    fn test_adaptor_max_by_key(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let mut input = (0..=INPUT_LEN)
            .map(|x| (x, INPUT_LEN - x))
            .collect::<Vec<(u64, u64)>>();
        let max = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .max_by_key(|pair| pair.1);
        assert_eq!(max, Some((0, INPUT_LEN)));

        input.truncate(1);
        let max_one = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .max_by_key(|pair| pair.1);
        assert_eq!(max_one, Some((0, INPUT_LEN)));

        input.clear();
        let max_empty = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .max_by_key(|pair| pair.1);
        assert_eq!(max_empty, None);
    }

    fn test_adaptor_min(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let mut input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let min = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .min();
        assert_eq!(min, Some(0));

        input.truncate(1);
        let min_one = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .min();
        assert_eq!(min_one, Some(0));

        input.clear();
        let min_empty = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .min();
        assert_eq!(min_empty, None);
    }

    fn test_adaptor_min_by(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        // Custom comparison function where even numbers are smaller than all odd
        // numbers.
        let mut input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let min = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .min_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));

        let first_even = 2;
        assert_eq!(min, Some(first_even));

        input.truncate(1);
        let min_one = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .min_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(min_one, Some(1));

        input.clear();
        let min_empty = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .min_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(min_empty, None);
    }

    fn test_adaptor_min_by_key(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let mut input = (0..=INPUT_LEN)
            .map(|x| (x, INPUT_LEN - x))
            .collect::<Vec<(u64, u64)>>();
        let min = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .min_by_key(|pair| pair.1);
        assert_eq!(min, Some((INPUT_LEN, 0)));

        input.truncate(1);
        let min_one = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .min_by_key(|pair| pair.1);
        assert_eq!(min_one, Some((0, INPUT_LEN)));

        input.clear();
        let min_empty = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .min_by_key(|pair| pair.1);
        assert_eq!(min_empty, None);
    }

    fn test_adaptor_ne(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let not_equal = (input.par_iter(), input.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .ne();
        assert!(!not_equal);

        let not_equal = (
            input.par_iter().take(INPUT_LEN as usize),
            input.par_iter().skip(1),
        )
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .ne();
        assert!(not_equal);
    }

    fn test_adaptor_ne_by_key(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (i, 1)).collect::<Vec<(u64, u64)>>();
        let not_equal = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .ne_by_key(|x| x.0);
        assert!(!not_equal);

        let not_equal = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .ne_by_key(|x| x.1);
        assert!(not_equal);
    }

    fn test_adaptor_ne_by_keys(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();
        let right = (0..=INPUT_LEN).map(|i| (1, i)).collect::<Vec<(u64, u64)>>();
        let not_equal = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .ne_by_keys(|x| x.0, |y| y.1);
        assert!(!not_equal);

        let not_equal = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .ne_by_keys(|x| x.1, |y| y.0);
        assert!(not_equal);
    }

    fn test_adaptor_partial_cmp(range_strategy: RangeStrategy) {
        use std::cmp::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).map(|x| x as f64).collect::<Vec<f64>>();
        let ordering = (input.par_iter(), input.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp();
        assert_eq!(ordering, Some(Ordering::Equal));

        let ordering = (
            input.par_iter().take(INPUT_LEN as usize),
            input.par_iter().skip(1),
        )
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp();
        assert_eq!(ordering, Some(Ordering::Less));

        let left = std::iter::once(INPUT_LEN)
            .chain(0..INPUT_LEN)
            .map(|x| x as f64)
            .collect::<Vec<f64>>();
        let right = (0..=INPUT_LEN).map(|x| x as f64).collect::<Vec<f64>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp();
        assert_eq!(ordering, Some(Ordering::Greater));

        let left = std::iter::once(f64::NAN)
            .chain((1..=INPUT_LEN).map(|x| x as f64))
            .collect::<Vec<f64>>();
        let right = (0..=INPUT_LEN).map(|x| x as f64).collect::<Vec<f64>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp();
        assert_eq!(ordering, None);

        let left = std::iter::once(INPUT_LEN as f64)
            .chain((0..INPUT_LEN).map(|_| f64::NAN))
            .collect::<Vec<f64>>();
        let right = (0..=INPUT_LEN).map(|x| x as f64).collect::<Vec<f64>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp();
        assert_eq!(ordering, Some(Ordering::Greater));
    }

    fn test_adaptor_partial_cmp_by(range_strategy: RangeStrategy) {
        use std::cmp::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN)
            .map(|i| (i as f64, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (0.0, i as f64))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by(|x, y| (x.0, x.1).partial_cmp(&(y.1, y.0)));
        assert_eq!(ordering, Some(Ordering::Equal));

        let left = (0..=INPUT_LEN)
            .map(|i| (i as f64, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (1.0, i as f64))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by(|x, y| (x.0, x.1).partial_cmp(&(y.1, y.0)));
        assert_eq!(ordering, Some(Ordering::Less));

        let left = (0..=INPUT_LEN)
            .map(|i| (if i == INPUT_LEN { f64::NAN } else { i as f64 }, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (0.0, i as f64))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by(|x, y| (x.0, x.1).partial_cmp(&(y.1, y.0)));
        assert_eq!(ordering, None);

        let left = (0..=INPUT_LEN)
            .map(|i| (if i == 0 { i as f64 } else { f64::NAN }, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (1.0, i as f64))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by(|x, y| (x.0, x.1).partial_cmp(&(y.1, y.0)));
        assert_eq!(ordering, Some(Ordering::Less));
    }

    fn test_adaptor_partial_cmp_by_key(range_strategy: RangeStrategy) {
        use std::cmp::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN)
            .map(|i| (i as f64, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (i as f64, 1.0))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_key(|x| x.0);
        assert_eq!(ordering, Some(Ordering::Equal));

        let left = (0..INPUT_LEN)
            .map(|i| (i as f64, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (1..=INPUT_LEN)
            .map(|i| (i as f64, 1.0))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_key(|x| x.0);
        assert_eq!(ordering, Some(Ordering::Less));

        let left = std::iter::once(INPUT_LEN)
            .chain(0..INPUT_LEN)
            .map(|i| (i as f64, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (i as f64, 1.0))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_key(|x| x.0);
        assert_eq!(ordering, Some(Ordering::Greater));

        let left = (0..=INPUT_LEN)
            .map(|i| (if i == INPUT_LEN { f64::NAN } else { i as f64 }, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (i as f64, 1.0))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_key(|x| x.0);
        assert_eq!(ordering, None);

        let left = (0..INPUT_LEN)
            .map(|i| (if i == 0 { i as f64 } else { f64::NAN }, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (1..=INPUT_LEN)
            .map(|i| (i as f64, 1.0))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_key(|x| x.0);
        assert_eq!(ordering, Some(Ordering::Less));
    }

    fn test_adaptor_partial_cmp_by_keys(range_strategy: RangeStrategy) {
        use std::cmp::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let left = (0..=INPUT_LEN)
            .map(|i| (i as f64, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (1.0, i as f64))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_keys(|x| x.0, |x| x.1);
        assert_eq!(ordering, Some(Ordering::Equal));

        let left = (0..INPUT_LEN)
            .map(|i| (i as f64, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (1..=INPUT_LEN)
            .map(|i| (1.0, i as f64))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_keys(|x| x.0, |x| x.1);
        assert_eq!(ordering, Some(Ordering::Less));

        let left = std::iter::once(INPUT_LEN)
            .chain(0..INPUT_LEN)
            .map(|i| (i as f64, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (1.0, i as f64))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_keys(|x| x.0, |x| x.1);
        assert_eq!(ordering, Some(Ordering::Greater));

        let left = (0..=INPUT_LEN)
            .map(|i| (if i == INPUT_LEN { f64::NAN } else { i as f64 }, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (0..=INPUT_LEN)
            .map(|i| (1.0, i as f64))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_keys(|x| x.0, |x| x.1);
        assert_eq!(ordering, None);

        let left = (0..INPUT_LEN)
            .map(|i| (if i == 0 { i as f64 } else { f64::NAN }, 0.0))
            .collect::<Vec<(f64, f64)>>();
        let right = (1..=INPUT_LEN)
            .map(|i| (1.0, i as f64))
            .collect::<Vec<(f64, f64)>>();
        let ordering = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .partial_cmp_by_keys(|x| x.0, |x| x.1);
        assert_eq!(ordering, Some(Ordering::Less));
    }

    fn test_adaptor_product(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (1..=INPUT_LEN).map(|_| -1).collect::<Vec<i32>>();
        let product = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .product::<i32>();
        assert_eq!(product, if INPUT_LEN % 2 == 0 { 1 } else { -1 });
    }

    fn test_adaptor_reduce(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .copied()
            .reduce(|| 0, |x, y| x + y);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_sum(range_strategy: RangeStrategy) {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_try_for_each(range_strategy: RangeStrategy) {
        use std::sync::atomic::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each(|&x| {
                if x <= INPUT_LEN {
                    sum.fetch_add(x, Ordering::Relaxed);
                    Ok(())
                } else {
                    Err(x)
                }
            });
        assert_eq!(result, Ok(()));
        assert_eq!(sum.into_inner(), INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each(|&x| {
                if x < INPUT_LEN {
                    sum.fetch_add(x, Ordering::Relaxed);
                    Ok(())
                } else {
                    Err(x)
                }
            });
        assert_eq!(result, Err(INPUT_LEN));

        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each(|&x| Err(x));
        assert!(result.is_err());
        assert!(result.unwrap_err() <= INPUT_LEN);
    }

    #[cfg(feature = "nightly")]
    fn test_adaptor_try_for_each_option(range_strategy: RangeStrategy) {
        use std::sync::atomic::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each(|&x| {
                if x <= INPUT_LEN {
                    sum.fetch_add(x, Ordering::Relaxed);
                    Some(())
                } else {
                    None
                }
            });
        assert_eq!(result, Some(()));
        assert_eq!(sum.into_inner(), INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each(|&x| {
                if x < INPUT_LEN {
                    sum.fetch_add(x, Ordering::Relaxed);
                    Some(())
                } else {
                    None
                }
            });
        assert_eq!(result, None);

        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each(|_| None);
        assert!(result.is_none());
    }

    fn test_adaptor_try_for_each_init(range_strategy: RangeStrategy) {
        use std::sync::atomic::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::thread_rng, |rng, &x| {
                let y = rng.gen_range(0..=x);
                sum.fetch_add(y, Ordering::Relaxed);
                if y <= INPUT_LEN {
                    Ok(())
                } else {
                    Err(y)
                }
            });
        assert_eq!(result, Ok(()));
        assert!(sum.into_inner() <= INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::thread_rng, |rng, &x| {
                let y = rng.gen_range(0..=INPUT_LEN);
                if y <= x {
                    sum.fetch_add(y, Ordering::Relaxed);
                    Ok(())
                } else {
                    Err(x)
                }
            });
        // The probability for this to fail is negligible if INPUT_LEN is large enough.
        assert!(result.is_err());

        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::thread_rng, |rng, &x| Err(x * rng.gen_range(1..=10)));
        assert!(result.is_err());
        assert!(result.unwrap_err() <= 10 * INPUT_LEN);
    }

    #[cfg(feature = "nightly")]
    fn test_adaptor_try_for_each_init_option(range_strategy: RangeStrategy) {
        use std::sync::atomic::Ordering;

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::thread_rng, |rng, &x| {
                let y = rng.gen_range(0..=x);
                sum.fetch_add(y, Ordering::Relaxed);
                if y <= INPUT_LEN {
                    Some(())
                } else {
                    None
                }
            });
        assert_eq!(result, Some(()));
        assert!(sum.into_inner() <= INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::thread_rng, |rng, &x| {
                let y = rng.gen_range(0..=INPUT_LEN);
                if y <= x {
                    sum.fetch_add(y, Ordering::Relaxed);
                    Some(())
                } else {
                    None
                }
            });
        // The probability for this to fail is negligible if INPUT_LEN is large enough.
        assert!(result.is_none());

        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::thread_rng, |_, _| None);
        assert!(result.is_none());
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
