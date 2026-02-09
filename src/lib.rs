// Copyright 2024-2026 Google LLC
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
    feature(cfg_sanitize, negative_impls)
)]
#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]
#![cfg_attr(
    feature = "nightly",
    feature(
        array_ptr_get,
        maybe_uninit_uninit_array_transpose,
        step_trait,
        try_trait_v2,
        try_trait_v2_residual
    )
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![doc(test(attr(deny(warnings))))]

#[cfg(any(feature = "rayon", feature = "default-thread-pool"))]
mod core;
pub mod iter;
#[cfg(any(feature = "rayon", feature = "default-thread-pool"))]
mod macros;
#[cfg(any(feature = "rayon", feature = "default-thread-pool"))]
pub mod threads;

/// Prelude of commonly used items from this crate.
pub mod prelude {
    pub use crate::iter::{
        ExactParallelSourceExt, GenericThreadPool, IntoExactParallelRefMutSource,
        IntoExactParallelRefSource, IntoExactParallelSource, IntoParallelRefMutSource,
        IntoParallelRefSource, IntoParallelSource, MinMaxResult, ParallelIterator,
        ParallelIteratorExt, ParallelSourceExt, ZipableSource,
    };
    #[cfg(feature = "rayon")]
    pub use crate::threads::RayonThreadPool;
    #[cfg(feature = "default-thread-pool")]
    pub use crate::threads::{CpuPinningPolicy, ThreadPool, ThreadPoolBuilder};
    #[cfg(any(feature = "rayon", feature = "default-thread-pool"))]
    pub use crate::threads::{RangeStrategy, ThreadCount};
}

#[cfg(all(test, any(feature = "rayon", feature = "default-thread-pool")))]
mod test {
    use crate::iter::MyHashSet;
    use crate::prelude::*;
    use rand::RngExt;
    use std::cell::Cell;
    use std::collections::{HashSet, VecDeque};
    use std::rc::Rc;
    use std::sync::atomic::AtomicU64;
    #[cfg(all(not(miri), feature = "log"))]
    use std::sync::LazyLock;
    use std::sync::Mutex;

    #[cfg(all(not(miri), feature = "log"))]
    static ENV_LOGGER_INIT: LazyLock<()> = LazyLock::new(env_logger::init);

    macro_rules! parallelism_tests {
        (
            $mod:ident,
            $thread_pool:expr,
            $( $( #[ $attrs:meta ] )* $case:ident $( => fail($msg:expr) )? ,)*
        ) => {
            mod $mod {
                use super::*;

                $(
                    $( #[$attrs] )*
                    #[test]
                    $( #[should_panic(expected = $msg)] )?
                    fn $case() {
                        #[cfg(all(not(miri), feature = "log"))]
                        LazyLock::force(&ENV_LOGGER_INIT);
                        $crate::test::$case($thread_pool);
                    }
                )*
            }
        };
    }

    macro_rules! parallelism_tests_select_failure_msg {
        (
            rayon,
            $thread_pool:expr,
            $( $( #[ $attrs:meta ] )* $case:ident $( => fail($msg:expr, $rayon_msg:expr) )? ,)*
        ) => {
            parallelism_tests!(
                rayon,
                $thread_pool,
                $( $( #[ $attrs ] )* $case $( => fail($rayon_msg) )? ,)*
            );
        };
        (
            rayon_local,
            $thread_pool:expr,
            $( $( #[ $attrs:meta ] )* $case:ident $( => fail($msg:expr, $rayon_msg:expr) )? ,)*
        ) => {
            parallelism_tests!(
                rayon_local,
                $thread_pool,
                $( $( #[ $attrs ] )* $case $( => fail($rayon_msg) )? ,)*
            );
        };
        (
            $mod:ident,
            $thread_pool:expr,
            $( $( #[ $attrs:meta ] )* $case:ident $( => fail($msg:expr, $rayon_msg:expr) )? ,)*
        ) => {
            parallelism_tests!(
                $mod,
                $thread_pool,
                $( $( #[ $attrs ] )* $case $( => fail($msg) )? ,)*
            );
        };
    }

    macro_rules! all_parallelism_tests {
        ( $mod:ident, $thread_pool:expr ) => {
            parallelism_tests_select_failure_msg!(
                $mod,
                $thread_pool,
                test_pipeline_sum_integers,
                test_pipeline_empty_input,
                test_pipeline_one_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_pipeline_some_panics => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_pipeline_many_panics => fail("worker thread(s) panicked!", "arithmetic panic"),
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
                #[cfg(feature = "nightly")]
                test_source_array,
                #[cfg(feature = "nightly")]
                test_source_array_boxed,
                #[cfg(feature = "nightly")]
                test_source_array_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                #[cfg(feature = "nightly")]
                test_source_array_find_any_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                #[cfg(feature = "nightly")]
                test_source_array_find_first_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_source_boxed_slice,
                test_source_hashset,
                test_source_slice,
                #[cfg(feature = "nightly_tests")]
                test_source_slice_not_send,
                test_source_slice_mut,
                test_source_slice_mut_not_sync,
                test_source_range,
                test_source_range_backwards => fail("cannot iterate over a backward range", "cannot iterate over a backward range"),
                #[cfg(feature = "nightly_tests")]
                #[cfg(not(any(miri, coverage, sanitize = "thread")))]
                test_source_range_u32max,
                #[cfg(feature = "nightly")]
                test_source_range_u64,
                #[cfg(feature = "nightly")]
                test_source_range_u128_too_large => fail("cannot iterate over a range with more than usize::MAX items", "cannot iterate over a range with more than usize::MAX items"),
                test_source_range_inclusive,
                test_source_range_inclusive_backwards => fail("cannot iterate over a backward range", "cannot iterate over a backward range"),
                test_source_range_inclusive_too_large => fail("cannot iterate over a range with more than usize::MAX items", "cannot iterate over a range with more than usize::MAX items"),
                #[cfg(feature = "nightly")]
                test_source_range_inclusive_u64,
                #[cfg(feature = "nightly")]
                test_source_range_inclusive_u64_too_large => fail("cannot iterate over a range with more than usize::MAX items", "cannot iterate over a range with more than usize::MAX items"),
                #[cfg(feature = "nightly")]
                test_source_range_inclusive_u128_too_large => fail("cannot iterate over a range with more than usize::MAX items", "cannot iterate over a range with more than usize::MAX items"),
                test_source_vec,
                test_source_vec_boxed,
                test_source_vec_find_any,
                test_source_vec_find_first,
                test_source_vec_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_source_vec_find_any_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_source_vec_find_first_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_source_vec_deque_ref,
                test_source_vec_deque_ref_mut,
                test_source_adaptor_chain,
                test_source_adaptor_cloned,
                test_source_adaptor_copied,
                test_source_adaptor_filter,
                test_source_adaptor_filter_map,
                test_source_adaptor_inspect,
                test_source_adaptor_map,
                test_source_adaptor_map_init,
                test_source_adaptor_rev,
                test_source_exact_adaptor_chain,
                test_source_exact_adaptor_chain_cleanup,
                test_source_exact_adaptor_chain_overflow => fail("called chain() with sources that together produce more than usize::MAX items", "called chain() with sources that together produce more than usize::MAX items"),
                test_source_exact_adaptor_chains_cleanup,
                test_source_exact_adaptor_cloned,
                test_source_exact_adaptor_copied,
                test_source_exact_adaptor_enumerate,
                test_source_exact_adaptor_enumerate_cleanup,
                test_source_exact_adaptor_filter,
                test_source_exact_adaptor_filter_find_first,
                test_source_exact_adaptor_filter_map,
                test_source_exact_adaptor_inspect,
                test_source_exact_adaptor_map,
                test_source_exact_adaptor_map_init,
                test_source_exact_adaptor_map_init_find_first,
                test_source_exact_adaptor_rev,
                test_source_exact_adaptor_rev_cleanup,
                test_source_exact_adaptor_skip,
                test_source_exact_adaptor_skip_cleanup,
                test_source_exact_adaptor_skip_exact,
                test_source_exact_adaptor_skip_exact_too_much => fail("called skip_exact() with more items than this source produces", "called skip_exact() with more items than this source produces"),
                test_source_exact_adaptor_step_by,
                test_source_exact_adaptor_step_by_cleanup,
                test_source_exact_adaptor_step_by_one,
                test_source_exact_adaptor_step_by_zero => fail("called step_by() with a step of zero", "called step_by() with a step of zero"),
                test_source_exact_adaptor_step_by_zero_empty => fail("called step_by() with a step of zero", "called step_by() with a step of zero"),
                test_source_exact_adaptor_take,
                test_source_exact_adaptor_take_cleanup,
                test_source_exact_adaptor_take_exact,
                test_source_exact_adaptor_take_exact_too_much => fail("called take_exact() with more items than this source produces", "called take_exact() with more items than this source produces"),
                test_source_exact_adaptor_zip_eq,
                test_source_exact_adaptor_zip_eq_cleanup,
                test_source_exact_adaptor_zip_eq_unequal_array => fail("called zip_eq() with sources of different lengths", "called zip_eq() with sources of different lengths"),
                test_source_exact_adaptor_zip_eq_unequal_tuple => fail("called zip_eq() with sources of different lengths", "called zip_eq() with sources of different lengths"),
                test_source_exact_adaptor_zip_max,
                test_source_exact_adaptor_zip_max_cleanup,
                test_source_exact_adaptor_zip_min,
                test_source_exact_adaptor_zip_min_cleanup,
                #[cfg(feature = "nightly")]
                test_sink_array,
                #[cfg(feature = "nightly")]
                test_sink_array_boxed,
                #[cfg(feature = "nightly")]
                test_sink_array_incorrect_length => fail("tried to collect an iterator into an array of the wrong length", "tried to collect an iterator into an array of the wrong length"),
                #[cfg(feature = "nightly")]
                test_sink_array_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                #[cfg(feature = "nightly")]
                test_sink_array_one_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_sink_boxed_slice,
                test_sink_boxed_slice_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_sink_vec,
                test_sink_vec_boxed,
                test_sink_vec_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_sink_vec_one_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_sink_vec_deque,
                test_sink_vec_deque_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_adaptor_all,
                test_adaptor_any,
                test_adaptor_cmp,
                test_adaptor_cmp_by,
                test_adaptor_cmp_by_key,
                test_adaptor_cmp_by_keys,
                test_adaptor_collect_per_thread,
                test_adaptor_eq,
                test_adaptor_eq_by_key,
                test_adaptor_eq_by_keys,
                test_adaptor_find_any,
                test_adaptor_find_first,
                test_adaptor_find_map_any,
                test_adaptor_find_map_first,
                test_adaptor_fold_per_thread,
                test_adaptor_for_each,
                test_adaptor_for_each_init,
                test_adaptor_map,
                test_adaptor_max,
                test_adaptor_max_by,
                test_adaptor_max_by_key,
                test_adaptor_min,
                test_adaptor_min_by,
                test_adaptor_min_by_key,
                test_adaptor_minmax,
                test_adaptor_minmax_by,
                test_adaptor_minmax_by_key,
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
                test_adaptor_try_collect,
                test_adaptor_try_collect_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                test_adaptor_try_collect_one_panic => fail("worker thread(s) panicked!", "arithmetic panic"),
                #[cfg(feature = "nightly")]
                test_adaptor_try_collect_option,
                test_adaptor_try_collect_per_thread,
                #[cfg(feature = "nightly")]
                test_adaptor_try_collect_per_thread_option,
                test_adaptor_try_fold_per_thread,
                #[cfg(feature = "nightly")]
                test_adaptor_try_fold_per_thread_option,
                test_adaptor_try_for_each,
                #[cfg(feature = "nightly")]
                test_adaptor_try_for_each_option,
                test_adaptor_try_for_each_init,
                #[cfg(feature = "nightly")]
                test_adaptor_try_for_each_init_option,
                test_adaptor_try_reduce,
                #[cfg(feature = "nightly")]
                test_adaptor_try_reduce_option,
            );
        };
    }

    #[cfg(feature = "default-thread-pool")]
    all_parallelism_tests!(
        fixed,
        ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build()
    );
    #[cfg(feature = "default-thread-pool")]
    all_parallelism_tests!(
        work_stealing,
        ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::WorkStealing,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build()
    );
    // TODO: Enable Miri once supported by Rayon and its dependencies: https://github.com/crossbeam-rs/crossbeam/issues/1181.
    #[cfg(all(feature = "rayon", not(miri)))]
    all_parallelism_tests!(
        rayon,
        RayonThreadPool::new_global(
            ThreadCount::AvailableParallelism,
            RangeStrategy::WorkStealing,
        )
    );
    // TODO: Enable Miri once supported by Rayon and its dependencies: https://github.com/crossbeam-rs/crossbeam/issues/1181.
    // TODO: Make this work with TSAN: https://github.com/rayon-rs/rayon/issues/1275.
    #[cfg(feature = "nightly_tests")]
    #[cfg(all(feature = "rayon", not(miri), not(sanitize = "thread")))]
    all_parallelism_tests!(
        rayon_local,
        RayonThreadPool::new(
            &rayon_core::ThreadPoolBuilder::new()
                .num_threads(4)
                .build()
                .unwrap(),
            ThreadCount::try_from(4).unwrap(),
            RangeStrategy::Fixed,
        )
    );

    #[cfg(not(miri))]
    const INPUT_LEN: u64 = 100_000;
    #[cfg(miri)]
    const INPUT_LEN: u64 = 200;

    // TSAN reports segmentation faults when creating too large arrays on the stack,
    // so we cap the input size accordingly.
    #[cfg(feature = "nightly")]
    const ARRAY_LEN: u64 = if INPUT_LEN < 10_000 {
        INPUT_LEN
    } else {
        10_000
    };

    fn test_pipeline_sum_integers<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_empty_input<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // The input can be empty.
        let input: [u64; 0] = [];
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        assert_eq!(sum, 0);
    }

    fn test_pipeline_one_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| {
                    let x = *item;
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

    fn test_pipeline_some_panics<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| {
                    let x = *item;
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

    fn test_pipeline_many_panics<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| {
                    let x = *item;
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

    fn test_pipelines_sum_twice<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // The same input can be processed multiple times on the thread pool.
        let sum1 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum2 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipelines_several_inputs<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // Several inputs can be used successively.
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum1 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        // n(n+1)/2
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let input = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let sum2 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        // 2n(2n+1)/2
        assert_eq!(sum2, INPUT_LEN * (2 * INPUT_LEN + 1));
    }

    fn test_pipelines_several_functions<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // Several functions can be computed successively.
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        // n(n+1)/2
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum_squares = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| {
                    let x = *item;
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

    fn test_pipelines_several_accumulators<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // Several accumulator types can be used successively.
        let sum1 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0u32,
                |acc, item| acc.wrapping_add(*item as u32),
                |acc| acc as u64,
                |a, b| (a + b) & 0xffff_ffff,
            );
        assert_eq!(sum1, (INPUT_LEN * (INPUT_LEN + 1) / 2) & 0xffff_ffff);

        let sum2 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0u64, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipelines_several_input_types<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // Several input types can be used successively.
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0u64, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let input = (0..=INPUT_LEN)
            .map(|x| format!("{x}"))
            .collect::<Vec<String>>();
        let sum_lengths = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0usize,
                |acc, item| acc + item.len(),
                |acc| acc as u64,
                |a, b| a + b,
            );
        assert_eq!(sum_lengths, expected_sum_lengths(INPUT_LEN));
    }

    fn test_pipelines_several_types<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // Pipelines with different types can be used successively.
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + *item, |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let input = (0..=INPUT_LEN)
            .map(|i| (2 * i, 2 * i + 1))
            .collect::<Vec<(u64, u64)>>();
        let sum_pairs = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || (0, 0),
                |(a, b), item| {
                    let (x, y) = *item;
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
    fn test_pipeline_non_send_functions<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // Non-Send functions can be used in the pipeline.
        let init = NotSend(0);
        let sum1 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                move || init.get(),
                |acc, item| acc + *item,
                |acc| acc,
                |a, b| a + b,
            );
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let one = NotSend(1);
        let sum2 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                move |acc, item| acc + *item * one.get(),
                |acc| acc,
                |a, b| a + b,
            );
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let one = NotSend(1);
        let sum3 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| acc + *item,
                move |acc| acc * one.get(),
                |a, b| a + b,
            );
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let zero = NotSend(0);
        let sum4 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| acc + *item,
                |acc| acc,
                move |a, b| a + b + zero.get(),
            );
        assert_eq!(sum4, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly_tests")]
    fn test_pipeline_non_send_input<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // A non-Send input can be used in the pipeline.
        let input = (0..=INPUT_LEN).map(NotSend).collect::<Vec<NotSend>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + item.get(), |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_non_sync_output<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // A non-Sync output can be used in the pipeline.
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| acc + *item,
                Cell::new,
                |a, b| Cell::new(a.get() + b.get()),
            )
            .get();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly_tests")]
    fn test_pipeline_non_send_accumulator<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // A non-Send accumulator can be used in the pipeline.
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || NotSend(0),
                |acc, item| NotSend(acc.0 + *item),
                |acc| acc.0,
                |a, b| a + b,
            );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_non_sync_accumulator<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // A non-Sync accumulator can be used in the pipeline.
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || Cell::new(0),
                |mut acc, item| {
                    *acc.get_mut() += *item;
                    acc
                },
                |acc| acc.get(),
                |a, b| a + b,
            );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_non_send_sync_accumulator<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        // A neither Send nor Sync accumulator can be used in the pipeline.
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || Rc::new(0),
                |mut acc, item| {
                    *Rc::get_mut(&mut acc).unwrap() += *item;
                    acc
                },
                |acc| *acc,
                |a, b| a + b,
            );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_local_lifetime_functions<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // The pipeline functions can borrow local values (and therefore have a local
        // lifetime).
        let zero = 0;
        let zero_ref = &zero;
        let one = 1;
        let one_ref = &one;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum1 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || *zero_ref,
                |acc, item| acc + *item,
                |acc| acc,
                |a, b| a + b,
            );
        assert_eq!(sum1, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum2 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| acc + *item * *one_ref,
                |acc| acc,
                |a, b| a + b,
            );
        assert_eq!(sum2, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum3 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| acc + *item,
                |acc| acc * *one_ref,
                |a, b| a + b,
            );
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum4 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| acc + *item,
                |acc| acc,
                |a, b| a + b + *zero_ref,
            );
        assert_eq!(sum4, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_local_lifetime_input<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // The pipeline input can borrow local values (and therefore have a local
        // lifetime).
        let token = ();
        let token_ref = &token;

        let input = (0..=INPUT_LEN)
            .map(|x| (x, token_ref))
            .collect::<Vec<(u64, _)>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, item| acc + item.0, |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_local_lifetime_output<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // The pipeline output can borrow local values (and therefore have a local
        // lifetime).
        let token = ();
        let token_ref = &token;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || 0,
                |acc, item| acc + *item,
                |acc| (acc, token_ref),
                |a, b| (a.0 + b.0, a.1),
            )
            .0;
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_pipeline_local_lifetime_accumulator<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // The pipeline accumulator can borrow local values (and therefore have a local
        // lifetime).
        let token = ();
        let token_ref = &token;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(
                || (0, token_ref),
                |acc, item| (acc.0 + *item, acc.1),
                |acc| acc.0,
                |a, b| a + b,
            );
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly")]
    fn test_source_array<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input: [u64; ARRAY_LEN as usize + 1] = std::array::from_fn(|i| i as u64);
        let sum = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, ARRAY_LEN * (ARRAY_LEN + 1) / 2);

        let input: [u64; ARRAY_LEN as usize + 1] = std::array::from_fn(|i| i as u64);
        let needle = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| *x % 10 == 9);
        assert!(needle.is_some());
        assert_eq!(needle.unwrap() % 10, 9);

        let input: [u64; ARRAY_LEN as usize + 1] = std::array::from_fn(|i| i as u64);
        let needle = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| *x % 10 == 9);
        assert_eq!(needle, Some(9));
    }

    #[cfg(feature = "nightly")]
    fn test_source_array_boxed<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input: [Box<u64>; ARRAY_LEN as usize + 1] = std::array::from_fn(|i| Box::new(i as u64));
        let sum = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum, ARRAY_LEN * (ARRAY_LEN + 1) / 2);

        let input: [Box<u64>; ARRAY_LEN as usize + 1] = std::array::from_fn(|i| Box::new(i as u64));
        let needle = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| **x % 10 == 9);
        assert!(needle.is_some());
        assert_eq!(*needle.unwrap() % 10, 9);

        let input: [Box<u64>; ARRAY_LEN as usize + 1] = std::array::from_fn(|i| Box::new(i as u64));
        let needle = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| **x % 10 == 9);
        assert_eq!(needle, Some(Box::new(9)));
    }

    #[cfg(feature = "nightly")]
    fn test_source_array_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input: [Box<u64>; ARRAY_LEN as usize + 1] = std::array::from_fn(|i| Box::new(i as u64));
        input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .for_each(|x| {
                if *x % 2 == 1 {
                    panic!("arithmetic panic");
                }
            });
    }

    #[cfg(feature = "nightly")]
    fn test_source_array_find_any_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input: [Box<u64>; ARRAY_LEN as usize + 1] = std::array::from_fn(|i| Box::new(i as u64));
        input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| {
                if **x % 2 == 0 {
                    false
                } else {
                    panic!("arithmetic panic");
                }
            });
    }

    #[cfg(feature = "nightly")]
    fn test_source_array_find_first_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input: [Box<u64>; ARRAY_LEN as usize + 1] = std::array::from_fn(|i| Box::new(i as u64));
        input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| {
                if **x % 2 == 0 {
                    false
                } else {
                    panic!("arithmetic panic");
                }
            });
    }

    fn test_source_boxed_slice<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Box<[Box<u64>]>>();
        let sum = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let input = (0..=INPUT_LEN).map(Box::new).collect::<Box<[Box<u64>]>>();
        let needle = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| **x % 10 == 9);
        assert!(needle.is_some());
        assert_eq!(*needle.unwrap() % 10, 9);

        let input = (0..=INPUT_LEN).map(Box::new).collect::<Box<[Box<u64>]>>();
        let needle = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| **x % 10 == 9);
        assert_eq!(needle, Some(Box::new(9)));
    }

    fn test_source_hashset<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, x| acc + x, |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_slice<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, x| acc + x, |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly_tests")]
    fn test_source_slice_not_send<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(NotSend).collect::<Vec<NotSend>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, x| acc + x.get(), |acc| acc, |a, b| a + b);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_slice_mut<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let mut values = (0..=INPUT_LEN).collect::<Vec<u64>>();
        values
            .par_iter_mut()
            .with_thread_pool(&mut thread_pool)
            .for_each(|x| *x *= 2);
        assert_eq!(values, (0..=INPUT_LEN).map(|x| x * 2).collect::<Vec<_>>());
    }

    fn test_source_slice_mut_not_sync<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_source_range<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let sum = (0..INPUT_LEN as usize)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
        assert_eq!(sum, (INPUT_LEN as usize - 1) * INPUT_LEN as usize / 2);
    }

    #[allow(clippy::reversed_empty_ranges)]
    fn test_source_range_backwards<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        (10..0)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
    }

    #[cfg(feature = "nightly_tests")]
    #[cfg(not(any(miri, coverage, sanitize = "thread")))]
    fn test_source_range_u32max<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let sum = (0..u32::MAX as usize)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|x| x as u64)
            .sum::<u64>();
        assert_eq!(sum, (u32::MAX as u64 - 1) * u32::MAX as u64 / 2);
    }

    #[cfg(feature = "nightly_tests")]
    #[cfg(not(any(miri, coverage, sanitize = "thread")))]
    #[test]
    fn test_source_range_u33max_fixed() {
        #[cfg(all(not(miri), feature = "log"))]
        LazyLock::force(&ENV_LOGGER_INIT);

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let sum = (0..u32::MAX as usize * 2)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|x| x as u64)
            .reduce(|| 0, |a, b| a.wrapping_add(b));
        assert_eq!(sum, (u32::MAX as u64 * 2 - 1).wrapping_mul(u32::MAX as u64));
    }

    #[cfg(feature = "nightly_tests")]
    #[cfg(not(any(miri, coverage, sanitize = "thread")))]
    #[test]
    #[should_panic(
        expected = "cannot process range of 8589934590 elements: only ranges of up to 4294967295 elements (2^32 - 1) are supported"
    )]
    fn test_source_range_u33max_work_stealing() {
        #[cfg(all(not(miri), feature = "log"))]
        LazyLock::force(&ENV_LOGGER_INIT);

        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::WorkStealing,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let sum = (0..u32::MAX as usize * 2)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|x| x as u64)
            .reduce(|| 0, |a, b| a.wrapping_add(b));
        assert_eq!(sum, (u32::MAX as u64 * 2 - 1).wrapping_mul(u32::MAX as u64));
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_u64<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let sum = (0..INPUT_LEN)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, (INPUT_LEN - 1) * INPUT_LEN / 2);
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_u128_too_large<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        (0u128..0x1_0000_0000_0000_0000)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u128>();
    }

    fn test_source_range_inclusive<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let sum = (0..=INPUT_LEN as usize)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
        assert_eq!(sum, INPUT_LEN as usize * (INPUT_LEN as usize + 1) / 2);
    }

    #[allow(clippy::reversed_empty_ranges)]
    fn test_source_range_inclusive_backwards<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        (10..=0)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
    }

    fn test_source_range_inclusive_too_large<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        (0..=usize::MAX)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_inclusive_u64<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let sum = (0..=INPUT_LEN)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_inclusive_u64_too_large<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        (0..=u64::MAX)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    #[cfg(feature = "nightly")]
    fn test_source_range_inclusive_u128_too_large<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        (0u128..=0x1_0000_0000_0000_0000)
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u128>();
    }

    fn test_source_vec<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_vec_boxed<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let sum = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_vec_find_any<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let needle = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| **x % 10 == 9);
        assert!(needle.is_some());
        assert_eq!(*needle.unwrap() % 10, 9);
    }

    fn test_source_vec_find_first<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let needle = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| **x % 10 == 9);
        assert_eq!(needle, Some(Box::new(9)));
    }

    fn test_source_vec_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .for_each(|x| {
                if *x % 2 == 1 {
                    panic!("arithmetic panic");
                }
            });
    }

    fn test_source_vec_find_any_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| {
                if **x % 2 == 0 {
                    false
                } else {
                    panic!("arithmetic panic");
                }
            });
    }

    fn test_source_vec_find_first_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| {
                if **x % 2 == 0 {
                    false
                } else {
                    panic!("arithmetic panic");
                }
            });
    }

    fn test_source_vec_deque_ref<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // Simple contiguous VecDeque.
        let input = (0..=INPUT_LEN).collect::<VecDeque<u64>>();
        assert!(vec_deque_is_contiguous(&input));

        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        // VecDeque split in 2 parts.
        let mut input = (1..=INPUT_LEN).collect::<VecDeque<u64>>();
        input.push_front(0);
        assert!(!vec_deque_is_contiguous(&input));

        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_vec_deque_ref_mut<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // Simple contiguous VecDeque.
        let mut values = (0..=INPUT_LEN).collect::<VecDeque<u64>>();
        assert!(vec_deque_is_contiguous(&values));

        values
            .par_iter_mut()
            .with_thread_pool(&mut thread_pool)
            .for_each(|x| *x *= 2);
        assert_eq!(
            values,
            (0..=INPUT_LEN).map(|x| x * 2).collect::<VecDeque<_>>()
        );

        // VecDeque split in 2 parts.
        let mut values = (1..=INPUT_LEN).collect::<VecDeque<u64>>();
        values.push_front(0);
        assert!(!vec_deque_is_contiguous(&values));

        values
            .par_iter_mut()
            .with_thread_pool(&mut thread_pool)
            .for_each(|x| *x *= 2);
        assert_eq!(
            values,
            (0..=INPUT_LEN).map(|x| x * 2).collect::<VecDeque<_>>()
        );
    }

    fn test_source_adaptor_chain<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input1 = (0..INPUT_LEN / 2).collect::<MyHashSet<u64>>();
        let input2 = (INPUT_LEN / 2..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let sum = input1
            .par_iter()
            .chain(input2.par_iter())
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        // Even though each set yields elements in arbitrary order, this specific needle
        // is guaranteed to be in the first half of the chained iterator.
        let needle = input1
            .par_iter()
            .chain(input2.par_iter())
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| *x >= INPUT_LEN / 2 - 1);
        assert_eq!(needle, Some(INPUT_LEN / 2 - 1));
    }

    fn test_source_adaptor_cloned<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN)
            .map(Box::new)
            .collect::<MyHashSet<Box<u64>>>();
        let sum = input
            .par_iter()
            .cloned()
            .with_thread_pool(&mut thread_pool)
            .reduce(
                || Box::new(0u64),
                |mut x, y| {
                    *x += *y;
                    x
                },
            );
        assert_eq!(*sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_adaptor_copied<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let sum = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .reduce(|| 0, |x, y| x + y);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_adaptor_filter<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let sum = input
            .par_iter()
            .filter(|&&x| x % 2 == 0)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, (INPUT_LEN / 2) * (INPUT_LEN / 2 + 1));
    }

    fn test_source_adaptor_filter_map<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let sum = input
            .par_iter()
            .filter_map(|&x| if x % 2 == 0 { Some(x * 3) } else { None })
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, 3 * (INPUT_LEN / 2) * (INPUT_LEN / 2 + 1));
    }

    fn test_source_adaptor_inspect<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::sync::atomic::Ordering;

        let input = (0..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let sum = AtomicU64::new(0);
        let max = input
            .par_iter()
            .copied()
            .inspect(|&x| {
                sum.fetch_add(x, Ordering::Relaxed);
            })
            .with_thread_pool(&mut thread_pool)
            .max();
        assert_eq!(max, Some(INPUT_LEN));

        let sum = sum.into_inner();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_adaptor_map<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let sum1 = input
            .par_iter()
            .map(|&x| x * 42)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum1, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum2 = input
            .par_iter()
            .map(|&x| x * 6)
            .map(|x| x * 7)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum2, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum3 = input
            .par_iter()
            // Mapping to a non-Send non-Sync type is fine, as the item stays on the same thread
            // and isn't shared with other threads.
            .map(|&x| Rc::new(x))
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, x| acc + *x, |acc| acc, |a, b| a + b);
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_adaptor_map_init<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let sum = input
            .par_iter()
            .map_init(
                rand::rng,
                |rng, &x| if rng.random() { x * 2 } else { x * 3 },
            )
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();

        assert!(sum >= INPUT_LEN * (INPUT_LEN + 1));
        assert!(sum <= 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_adaptor_rev<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let sum = input
            .par_iter()
            .rev()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        // Even though each set yields elements in arbitrary order, this specific needle
        // is guaranteed to be in the first half of the chained and reversed iterator.
        let input1 = (0..INPUT_LEN / 2).collect::<MyHashSet<u64>>();
        let input2 = (INPUT_LEN / 2..=INPUT_LEN).collect::<MyHashSet<u64>>();
        let needle = input1
            .par_iter()
            .chain(input2.par_iter())
            .rev()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| *x <= INPUT_LEN / 2);
        assert_eq!(needle, Some(INPUT_LEN / 2));
    }

    fn test_source_exact_adaptor_chain<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input1 = (0..INPUT_LEN / 2).collect::<Vec<u64>>();
        let input2 = (INPUT_LEN / 2..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input1
            .par_iter()
            .chain(input2.par_iter())
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_exact_adaptor_chain_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input1 = (0..INPUT_LEN / 2).map(Box::new).collect::<Vec<Box<u64>>>();
        let input2 = (INPUT_LEN / 2..=INPUT_LEN)
            .map(Box::new)
            .collect::<Vec<Box<u64>>>();

        let sum = input1
            .clone()
            .into_par_iter()
            .chain(input2.clone().into_par_iter())
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let needle = input1
            .clone()
            .into_par_iter()
            .chain(input2.clone().into_par_iter())
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| **x % 10 == 9);
        assert!(needle.is_some());
        assert_eq!(*needle.unwrap() % 10, 9);

        let needle = input1
            .into_par_iter()
            .chain(input2.into_par_iter())
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| **x % 10 == 9);
        assert_eq!(needle, Some(Box::new(9)));
    }

    fn test_source_exact_adaptor_chain_overflow<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        (0..usize::MAX)
            .into_par_iter()
            .chain((0..1).into_par_iter())
            .with_thread_pool(&mut thread_pool)
            .sum::<usize>();
    }

    fn test_source_exact_adaptor_chains_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let make_chained_iters = || {
            // Make a binary tree of chained iterators.
            let inputs: [Vec<Box<u64>>; 16] = std::array::from_fn(|chunk| {
                ((chunk as u64 * INPUT_LEN) / 16..((chunk as u64 + 1) * INPUT_LEN) / 16)
                    .map(Box::new)
                    .collect()
            });

            let [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15] =
                inputs.map(|chunk| chunk.into_par_iter());
            let (y0, y1, y2, y3, y4, y5, y6, y7) = (
                x0.chain(x1),
                x2.chain(x3),
                x4.chain(x5),
                x6.chain(x7),
                x8.chain(x9),
                x10.chain(x11),
                x12.chain(x13),
                x14.chain(x15),
            );
            let (z0, z1, z2, z3) = (y0.chain(y1), y2.chain(y3), y4.chain(y5), y6.chain(y7));
            let (t0, t1) = (z0.chain(z1), z2.chain(z3));
            t0.chain(t1)
        };

        let sum = make_chained_iters()
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN - 1) / 2);

        let needle = make_chained_iters()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| **x % 10 == 9);
        assert!(needle.is_some());
        assert_eq!(*needle.unwrap() % 10, 9);

        let needle = make_chained_iters()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| **x % 10 == 9);
        assert_eq!(needle, Some(Box::new(9)));
    }

    fn test_source_exact_adaptor_cloned<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let sum = input
            .par_iter()
            .cloned()
            .with_thread_pool(&mut thread_pool)
            .reduce(
                || Box::new(0u64),
                |mut x, y| {
                    *x += *y;
                    x
                },
            );
        assert_eq!(*sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_exact_adaptor_copied<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .reduce(|| 0, |x, y| x + y);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_exact_adaptor_enumerate<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_source_exact_adaptor_enumerate_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let sum_squares = input
            .clone()
            .into_par_iter()
            .enumerate()
            .with_thread_pool(&mut thread_pool)
            .map(|(i, x)| i as u64 * *x)
            .sum::<u64>();
        assert_eq!(
            sum_squares,
            INPUT_LEN * (INPUT_LEN + 1) * (2 * INPUT_LEN + 1) / 6
        );

        let needle = input
            .clone()
            .into_par_iter()
            .enumerate()
            .with_thread_pool(&mut thread_pool)
            .find_any(|(_, x)| **x % 10 == 9);
        let needle = needle.unwrap();
        assert_eq!(*needle.1 % 10, 9);
        assert_eq!(needle.0 as u64, *needle.1);

        let needle = input
            .clone()
            .into_par_iter()
            .enumerate()
            .with_thread_pool(&mut thread_pool)
            .find_first(|(_, x)| **x % 10 == 9);
        assert_eq!(needle, Some((9, Box::new(9))));
    }

    fn test_source_exact_adaptor_filter<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .filter(|&&x| x % 2 == 0)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, (INPUT_LEN / 2) * (INPUT_LEN / 2 + 1));
    }

    fn test_source_exact_adaptor_filter_find_first<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let needle = input
            .par_iter()
            .filter(|&&x| x % 6 == 5)
            .with_thread_pool(&mut thread_pool)
            .find_first(|&x| x % 7 == 6);
        assert_eq!(needle, Some(&41));
    }

    fn test_source_exact_adaptor_filter_map<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .filter_map(|&x| if x % 2 == 0 { Some(x * 3) } else { None })
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, 3 * (INPUT_LEN / 2) * (INPUT_LEN / 2 + 1));
    }

    fn test_source_exact_adaptor_inspect<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::sync::atomic::Ordering;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = AtomicU64::new(0);
        let max = input
            .par_iter()
            .copied()
            .inspect(|&x| {
                sum.fetch_add(x, Ordering::Relaxed);
            })
            .with_thread_pool(&mut thread_pool)
            .max();
        assert_eq!(max, Some(INPUT_LEN));

        let sum = sum.into_inner();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_exact_adaptor_map<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum1 = input
            .par_iter()
            .map(|&x| x * 42)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum1, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum2 = input
            .par_iter()
            .map(|&x| x * 6)
            .map(|x| x * 7)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum2, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum3 = input
            .par_iter()
            // Mapping to a non-Send non-Sync type is fine, as the item stays on the same thread
            // and isn't shared with other threads.
            .map(|&x| Rc::new(x))
            .with_thread_pool(&mut thread_pool)
            .pipeline(|| 0, |acc, x| acc + *x, |acc| acc, |a, b| a + b);
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_exact_adaptor_map_init<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .map_init(
                rand::rng,
                |rng, &x| if rng.random() { x * 2 } else { x * 3 },
            )
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();

        assert!(sum >= INPUT_LEN * (INPUT_LEN + 1));
        assert!(sum <= 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_exact_adaptor_map_init_find_first<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let needle = input
            .par_iter()
            .map_init(
                rand::rng,
                |rng, &x| if rng.random() { 2 * x } else { 2 * x + 1 },
            )
            .with_thread_pool(&mut thread_pool)
            .find_first(|&x| x >= 10);
        let needle = needle.unwrap();
        assert!(needle == 10 || needle == 11);
    }

    fn test_source_exact_adaptor_rev<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_source_exact_adaptor_rev_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();

        let sum = input
            .clone()
            .into_par_iter()
            .rev()
            .enumerate()
            .with_thread_pool(&mut thread_pool)
            .map(|(i, x)| i as u64 * *x)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN - 1) * (INPUT_LEN + 1) / 6);

        let needle = input
            .clone()
            .into_par_iter()
            .rev()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| **x % 10 == 9);
        assert!(needle.is_some());
        assert_eq!(*needle.unwrap() % 10, 9);

        let needle = input
            .into_par_iter()
            .rev()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| **x % 10 == 9);
        let expected = ((INPUT_LEN + 1) / 10) * 10 - 1;
        assert_eq!(needle, Some(Box::new(expected)));
    }

    fn test_source_exact_adaptor_skip<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .skip(INPUT_LEN as usize / 2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN.div_ceil(2) * ((3 * INPUT_LEN) / 2 + 1) / 2);

        let sum_empty = input
            .par_iter()
            .skip(2 * INPUT_LEN as usize)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum_empty, 0);
    }

    fn test_source_exact_adaptor_skip_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (1..=2 * INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();

        // Skip less than half of the items.
        let sum = input
            .clone()
            .into_par_iter()
            .skip(INPUT_LEN as usize / 2)
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(
            sum,
            (3 * INPUT_LEN).div_ceil(2) * ((5 * INPUT_LEN) / 2 + 1) / 2
        );

        // Skip more than half of the items.
        let sum = input
            .clone()
            .into_par_iter()
            .skip(3 * INPUT_LEN as usize / 2)
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN.div_ceil(2) * ((7 * INPUT_LEN) / 2 + 1) / 2);

        // Skip less than half or more than half of the items.
        for skip in [INPUT_LEN / 2, 3 * INPUT_LEN / 2] {
            let input = (0..2 * INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();

            let needle = input
                .clone()
                .into_par_iter()
                .skip(skip as usize)
                .with_thread_pool(&mut thread_pool)
                .find_any(|x| **x % 10 == 9);
            assert!(needle.is_some());
            assert_eq!(*needle.unwrap() % 10, 9);

            let needle = input
                .into_par_iter()
                .skip(skip as usize)
                .with_thread_pool(&mut thread_pool)
                .find_first(|x| **x % 10 == 9);
            let expected = (skip / 10) * 10 + 9;
            assert_eq!(needle, Some(Box::new(expected)));
        }
    }

    fn test_source_exact_adaptor_skip_exact<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .skip_exact(INPUT_LEN as usize / 2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN.div_ceil(2) * ((3 * INPUT_LEN) / 2 + 1) / 2);
    }

    fn test_source_exact_adaptor_skip_exact_too_much<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        input
            .par_iter()
            .skip_exact(2 * INPUT_LEN as usize)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    fn test_source_exact_adaptor_step_by<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_source_exact_adaptor_step_by_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=3 * INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();

        let sum_by_3 = input
            .clone()
            .into_par_iter()
            .step_by(3)
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum_by_3, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum_by_3 = input
            .clone()
            .into_par_iter()
            .take(3 * INPUT_LEN as usize)
            .step_by(3)
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum_by_3, 3 * INPUT_LEN * (INPUT_LEN - 1) / 2);

        let sum_by_3 = input
            .clone()
            .into_par_iter()
            .take(3 * INPUT_LEN as usize - 1)
            .step_by(3)
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum_by_3, 3 * INPUT_LEN * (INPUT_LEN - 1) / 2);

        let needle = input
            .clone()
            .into_par_iter()
            .step_by(7)
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| **x % 10 == 9);
        assert!(needle.is_some());
        assert_eq!(*needle.unwrap() % 10, 9);

        let needle = input
            .into_par_iter()
            .step_by(7)
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| **x % 10 == 9);
        assert_eq!(needle, Some(Box::new(49)));
    }

    fn test_source_exact_adaptor_step_by_one<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();

        let sum = input
            .clone()
            .into_par_iter()
            .step_by(1)
            .with_thread_pool(&mut thread_pool)
            .map(|x| *x)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);

        let needle = input
            .clone()
            .into_par_iter()
            .step_by(1)
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| **x % 10 == 9);
        assert!(needle.is_some());
        assert_eq!(*needle.unwrap() % 10, 9);

        let needle = input
            .into_par_iter()
            .step_by(1)
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| **x % 10 == 9);
        assert_eq!(needle, Some(Box::new(9)));
    }

    fn test_source_exact_adaptor_step_by_zero<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        input
            .par_iter()
            .step_by(0)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    fn test_source_exact_adaptor_step_by_zero_empty<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        [].par_iter()
            .step_by(0)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    fn test_source_exact_adaptor_take<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_source_exact_adaptor_take_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // Take less than half or more than half of the items.
        for take in [INPUT_LEN / 2, 3 * INPUT_LEN / 2] {
            let input = (1..=2 * INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
            let sum = input
                .into_par_iter()
                .take(take as usize)
                .with_thread_pool(&mut thread_pool)
                .map(|x| *x)
                .sum::<u64>();
            assert_eq!(sum, (take * (take + 1)) / 2);

            let input = (0..2 * INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();

            let needle = input
                .clone()
                .into_par_iter()
                .take(take as usize)
                .with_thread_pool(&mut thread_pool)
                .find_any(|x| **x % 10 == 9);
            assert!(needle.is_some());
            assert_eq!(*needle.unwrap() % 10, 9);

            let needle = input
                .into_par_iter()
                .take(take as usize)
                .with_thread_pool(&mut thread_pool)
                .find_first(|x| **x % 10 == 9);
            assert_eq!(needle, Some(Box::new(9)));
        }
    }

    fn test_source_exact_adaptor_take_exact<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .take_exact(INPUT_LEN as usize / 2)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, ((INPUT_LEN / 2) * (INPUT_LEN / 2 + 1)) / 2);
    }

    fn test_source_exact_adaptor_take_exact_too_much<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        input
            .par_iter()
            .take_exact(2 * INPUT_LEN as usize)
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
    }

    fn test_source_exact_adaptor_zip_eq<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let left = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN).collect::<Vec<u64>>();

        // Tuples.
        let (sum_left, sum_right) = (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .map(|(&a, &b)| (a, b))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        // Arrays.
        let [sum_left, sum_right] = [left.par_iter(), right.par_iter()]
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .map(|[&a, &b]| [a, b])
            .reduce(|| [0, 0], |[a, b], [c, d]| [a + c, b + d]);
        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_exact_adaptor_zip_eq_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let left = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN)
            .map(Box::new)
            .collect::<Vec<Box<u64>>>();

        // Tuples.
        let (sum_left, sum_right) = (left.clone().into_par_iter(), right.clone().into_par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .map(|(a, b)| (*a, *b))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let needle = (left.clone().into_par_iter(), right.clone().into_par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .find_any(|(x, _)| **x % 10 == 9);
        let needle = needle.unwrap();
        assert_eq!(*needle.0 % 10, 9);
        assert_eq!(*needle.1, *needle.0 + INPUT_LEN);

        let needle = (left.clone().into_par_iter(), right.clone().into_par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .find_first(|(x, _)| **x % 10 == 9);
        assert_eq!(needle, Some((Box::new(9), Box::new(INPUT_LEN + 9))));

        // Arrays.
        let [sum_left, sum_right] = [left.clone().into_par_iter(), right.clone().into_par_iter()]
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .map(|[a, b]| [*a, *b])
            .reduce(|| [0, 0], |[a, b], [c, d]| [a + c, b + d]);
        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let needle = [left.clone().into_par_iter(), right.clone().into_par_iter()]
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| *x[0] % 10 == 9);
        let needle = needle.unwrap();
        assert_eq!(*needle[0] % 10, 9);
        assert_eq!(*needle[1], *needle[0] + INPUT_LEN);

        let needle = [left.into_par_iter(), right.into_par_iter()]
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| *x[0] % 10 == 9);
        assert_eq!(needle, Some([Box::new(9), Box::new(INPUT_LEN + 9)]));
    }

    fn test_source_exact_adaptor_zip_eq_unequal_array<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let left = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN).collect::<Vec<u64>>();
        [left.par_iter(), right.par_iter()]
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .map(|[&a, &b]| [a, b])
            .reduce(|| [0, 0], |[a, b], [c, d]| [a + c, b + d]);
    }

    fn test_source_exact_adaptor_zip_eq_unequal_tuple<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let left = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN).collect::<Vec<u64>>();
        (left.par_iter(), right.par_iter())
            .zip_eq()
            .with_thread_pool(&mut thread_pool)
            .map(|(&a, &b)| (a, b))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
    }

    fn test_source_exact_adaptor_zip_max<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let left = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN).collect::<Vec<u64>>();

        // Tuples.
        let (sum_left, sum_right) = (left.par_iter(), right.par_iter())
            .zip_max()
            .with_thread_pool(&mut thread_pool)
            .map(|(a, b)| (a.copied().unwrap(), b.copied().unwrap_or(0)))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
        assert_eq!(sum_left, INPUT_LEN * (2 * INPUT_LEN + 1));
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        // Arrays.
        let [sum_left, sum_right] = [left.par_iter(), right.par_iter()]
            .zip_max()
            .with_thread_pool(&mut thread_pool)
            .map(|[a, b]| [a.copied().unwrap(), b.copied().unwrap_or(0)])
            .reduce(|| [0, 0], |[a, b], [c, d]| [a + c, b + d]);
        assert_eq!(sum_left, INPUT_LEN * (2 * INPUT_LEN + 1));
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_exact_adaptor_zip_max_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let left = (0..=2 * INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN)
            .map(Box::new)
            .collect::<Vec<Box<u64>>>();

        // Tuples.
        let (sum_left, sum_right) = (left.clone().into_par_iter(), right.clone().into_par_iter())
            .zip_max()
            .with_thread_pool(&mut thread_pool)
            .map(|(a, b)| (a.map(|x| *x).unwrap(), b.map(|x| *x).unwrap_or(0)))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
        assert_eq!(sum_left, INPUT_LEN * (2 * INPUT_LEN + 1));
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let needle = (left.clone().into_par_iter(), right.clone().into_par_iter())
            .zip_max()
            .with_thread_pool(&mut thread_pool)
            .find_any(|(x, _)| **x.as_ref().unwrap() % 10 == 9);
        let needle: (Option<Box<u64>>, Option<Box<u64>>) = needle.unwrap();
        assert_eq!(*needle.0.unwrap() % 10, 9);

        let needle = (left.clone().into_par_iter(), right.clone().into_par_iter())
            .zip_max()
            .with_thread_pool(&mut thread_pool)
            .find_first(|(x, _)| **x.as_ref().unwrap() % 10 == 9);
        assert_eq!(
            needle,
            Some((Some(Box::new(9)), Some(Box::new(INPUT_LEN + 9))))
        );

        // Arrays.
        let [sum_left, sum_right] = [left.clone().into_par_iter(), right.clone().into_par_iter()]
            .zip_max()
            .with_thread_pool(&mut thread_pool)
            .map(|[a, b]| [a.map(|x| *x).unwrap(), b.map(|x| *x).unwrap_or(0)])
            .reduce(|| [0, 0], |[a, b], [c, d]| [a + c, b + d]);
        assert_eq!(sum_left, INPUT_LEN * (2 * INPUT_LEN + 1));
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let needle = [left.clone().into_par_iter(), right.clone().into_par_iter()]
            .zip_max()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| **x[0].as_ref().unwrap() % 10 == 9);
        let needle: [Option<Box<u64>>; 2] = needle.unwrap();
        assert_eq!(**needle[0].as_ref().unwrap() % 10, 9);

        let needle = [left.into_par_iter(), right.into_par_iter()]
            .zip_max()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| **x[0].as_ref().unwrap() % 10 == 9);
        assert_eq!(
            needle,
            Some([Some(Box::new(9)), Some(Box::new(INPUT_LEN + 9))])
        );
    }

    fn test_source_exact_adaptor_zip_min<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let left = (0..=2 * INPUT_LEN).collect::<Vec<u64>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN).collect::<Vec<u64>>();

        // Tuples.
        let (sum_left, sum_right) = (left.par_iter(), right.par_iter())
            .zip_min()
            .with_thread_pool(&mut thread_pool)
            .map(|(&a, &b)| (a, b))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        // Arrays.
        let [sum_left, sum_right] = [left.par_iter(), right.par_iter()]
            .zip_min()
            .with_thread_pool(&mut thread_pool)
            .map(|[&a, &b]| [a, b])
            .reduce(|| [0, 0], |[a, b], [c, d]| [a + c, b + d]);
        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_source_exact_adaptor_zip_min_cleanup<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let left = (0..=2 * INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let right = (INPUT_LEN..=2 * INPUT_LEN)
            .map(Box::new)
            .collect::<Vec<Box<u64>>>();

        // Tuples.
        let (sum_left, sum_right) = (left.clone().into_par_iter(), right.clone().into_par_iter())
            .zip_min()
            .with_thread_pool(&mut thread_pool)
            .map(|(a, b)| (*a, *b))
            .reduce(|| (0, 0), |(a, b), (c, d)| (a + c, b + d));
        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let needle = (left.clone().into_par_iter(), right.clone().into_par_iter())
            .zip_min()
            .with_thread_pool(&mut thread_pool)
            .find_any(|(x, _)| **x % 10 == 9);
        let needle = needle.unwrap();
        assert_eq!(*needle.0 % 10, 9);
        assert_eq!(*needle.1, *needle.0 + INPUT_LEN);

        let needle = (left.clone().into_par_iter(), right.clone().into_par_iter())
            .zip_min()
            .with_thread_pool(&mut thread_pool)
            .find_first(|(x, _)| **x % 10 == 9);
        assert_eq!(needle, Some((Box::new(9), Box::new(INPUT_LEN + 9))));

        // Arrays.
        let [sum_left, sum_right] = [left.clone().into_par_iter(), right.clone().into_par_iter()]
            .zip_min()
            .with_thread_pool(&mut thread_pool)
            .map(|[a, b]| [*a, *b])
            .reduce(|| [0, 0], |[a, b], [c, d]| [a + c, b + d]);
        assert_eq!(sum_left, INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert_eq!(sum_right, 3 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let needle = [left.clone().into_par_iter(), right.clone().into_par_iter()]
            .zip_min()
            .with_thread_pool(&mut thread_pool)
            .find_any(|x| *x[0] % 10 == 9);
        let needle = needle.unwrap();
        assert_eq!(*needle[0] % 10, 9);
        assert_eq!(*needle[1], *needle[0] + INPUT_LEN);

        let needle = [left.into_par_iter(), right.into_par_iter()]
            .zip_min()
            .with_thread_pool(&mut thread_pool)
            .find_first(|x| *x[0] % 10 == 9);
        assert_eq!(needle, Some([Box::new(9), Box::new(INPUT_LEN + 9)]));
    }

    #[cfg(feature = "nightly")]
    fn test_sink_array<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=ARRAY_LEN).collect::<Vec<u64>>();
        let collection: [u64; ARRAY_LEN as usize + 1] = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .collect();
        for (i, x) in collection.iter().enumerate() {
            assert_eq!(i as u64, *x);
        }
    }

    #[cfg(feature = "nightly")]
    fn test_sink_array_boxed<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=ARRAY_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let collection: [Box<u64>; ARRAY_LEN as usize + 1] = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .collect();
        for (i, x) in collection.iter().enumerate() {
            assert_eq!(i as u64, **x);
        }
    }

    #[cfg(feature = "nightly")]
    fn test_sink_array_incorrect_length<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=ARRAY_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let _: [Box<u64>; ARRAY_LEN as usize] = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .collect();
    }

    #[cfg(feature = "nightly")]
    fn test_sink_array_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=ARRAY_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let _: [Box<u64>; ARRAY_LEN as usize + 1] = input
            .into_par_iter()
            .inspect(|x| {
                if **x % 10 == 9 {
                    panic!("arithmetic panic")
                }
            })
            .with_thread_pool(&mut thread_pool)
            .collect();
    }

    #[cfg(feature = "nightly")]
    fn test_sink_array_one_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=ARRAY_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let _: [Box<u64>; ARRAY_LEN as usize + 1] = input
            .into_par_iter()
            .inspect(|x| {
                if **x == 1 {
                    panic!("arithmetic panic")
                }
            })
            .with_thread_pool(&mut thread_pool)
            .collect();
    }

    fn test_sink_boxed_slice<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let collection = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .collect::<Box<[Box<u64>]>>();
        for (i, x) in collection.iter().enumerate() {
            assert_eq!(i as u64, **x);
        }
    }

    fn test_sink_boxed_slice_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        input
            .into_par_iter()
            .inspect(|x| {
                if **x % 10 == 9 {
                    panic!("arithmetic panic")
                }
            })
            .with_thread_pool(&mut thread_pool)
            .collect::<Box<[Box<u64>]>>();
    }

    fn test_sink_vec<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let collection = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .collect::<Vec<u64>>();
        for (i, x) in collection.iter().enumerate() {
            assert_eq!(i as u64, *x);
        }
    }

    fn test_sink_vec_boxed<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let collection = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .collect::<Vec<Box<u64>>>();
        for (i, x) in collection.iter().enumerate() {
            assert_eq!(i as u64, **x);
        }
    }

    fn test_sink_vec_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        input
            .into_par_iter()
            .inspect(|x| {
                if **x % 10 == 9 {
                    panic!("arithmetic panic")
                }
            })
            .with_thread_pool(&mut thread_pool)
            .collect::<Vec<Box<u64>>>();
    }

    fn test_sink_vec_one_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        input
            .into_par_iter()
            .inspect(|x| {
                if **x == 1 {
                    panic!("arithmetic panic")
                }
            })
            .with_thread_pool(&mut thread_pool)
            .collect::<Vec<Box<u64>>>();
    }

    fn test_sink_vec_deque<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let collection = input
            .into_par_iter()
            .with_thread_pool(&mut thread_pool)
            .collect::<VecDeque<Box<u64>>>();
        for (i, x) in collection.iter().enumerate() {
            assert_eq!(i as u64, **x);
        }
    }

    fn test_sink_vec_deque_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        input
            .into_par_iter()
            .inspect(|x| {
                if **x % 10 == 9 {
                    panic!("arithmetic panic")
                }
            })
            .with_thread_pool(&mut thread_pool)
            .collect::<VecDeque<Box<u64>>>();
    }

    fn test_adaptor_all<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_adaptor_any<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_adaptor_cmp<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::cmp::Ordering;

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

    fn test_adaptor_cmp_by<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::cmp::Ordering;

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

    fn test_adaptor_cmp_by_key<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::cmp::Ordering;

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

    fn test_adaptor_cmp_by_keys<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::cmp::Ordering;

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

    fn test_adaptor_collect_per_thread<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let collection: Vec<Vec<u64>> = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .collect_per_thread();

        let mut values: Vec<u64> = collection.into_iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, input);

        // The inner type can be any collection.
        let collection: Vec<HashSet<u64>> = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .collect_per_thread();

        let mut values: Vec<u64> = collection.into_iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, input);
    }

    fn test_adaptor_eq<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_adaptor_eq_by_key<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_adaptor_eq_by_keys<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_adaptor_find_any<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let first = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_any(|&x| x == 0);
        assert_eq!(first, Some(0));

        let last = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_any(|&x| x == INPUT_LEN);
        assert_eq!(last, Some(INPUT_LEN));

        let end = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_any(|&x| x == INPUT_LEN + 1);
        assert_eq!(end, None);

        let forty_two = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_any(|&x| x == 42);
        assert_eq!(forty_two, if INPUT_LEN >= 42 { Some(42) } else { None });

        let even = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_any(|&x| x % 2 == 0);
        assert!(even.unwrap() % 2 == 0);

        let empty = []
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_any(|_: &&u64| true);
        assert_eq!(empty, None);
    }

    fn test_adaptor_find_first<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let first = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_first(|_| true);
        assert_eq!(first, Some(0));

        let last = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_first(|&x| x >= INPUT_LEN);
        assert_eq!(last, Some(INPUT_LEN));

        let end = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_first(|&x| x > INPUT_LEN);
        assert_eq!(end, None);

        let forty_two = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_first(|&x| x >= 42);
        assert_eq!(forty_two, if INPUT_LEN >= 42 { Some(42) } else { None });

        let even = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_first(|&x| x % 2 == 0);
        assert_eq!(even, Some(0));

        let odd = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_first(|&x| x % 2 == 1);
        assert_eq!(odd, Some(1));

        let empty = []
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_first(|_: &&u64| true);
        assert_eq!(empty, None);
    }

    fn test_adaptor_find_map_any<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let first = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_any(|x| if x == 0 { Some(2 * x) } else { None });
        assert_eq!(first, Some(0));

        let last = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_any(|x| if x == INPUT_LEN { Some(2 * x) } else { None });
        assert_eq!(last, Some(2 * INPUT_LEN));

        let end = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_any(|x| if x > INPUT_LEN { Some(2 * x) } else { None });
        assert_eq!(end, None);

        let forty_two = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_any(|x| if x == 42 { Some(2 * x) } else { None });
        assert_eq!(forty_two, if INPUT_LEN >= 42 { Some(84) } else { None });

        let even = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_any(|x| if x % 2 == 0 { Some(2 * x) } else { None });
        assert!(even.unwrap() % 4 == 0);

        let empty = []
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_map_any(|_: &u64| Some(42));
        assert_eq!(empty, None);
    }

    fn test_adaptor_find_map_first<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let first = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_first(|x| Some(2 * x));
        assert_eq!(first, Some(0));

        let last = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_first(|x| if x >= INPUT_LEN { Some(2 * x) } else { None });
        assert_eq!(last, Some(2 * INPUT_LEN));

        let end = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_first(|x| if x > INPUT_LEN { Some(2 * x) } else { None });
        assert_eq!(end, None);

        let forty_two = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_first(|x| if x >= 42 { Some(2 * x) } else { None });
        assert_eq!(forty_two, if INPUT_LEN >= 42 { Some(84) } else { None });

        let even = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_first(|x| if x % 2 == 0 { Some(2 * x) } else { None });
        assert_eq!(even, Some(0));

        let odd = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .find_map_first(|x| if x % 2 == 1 { Some(2 * x) } else { None });
        assert_eq!(odd, Some(2));

        let empty = []
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .find_map_first(|_: &u64| Some(42));
        assert_eq!(empty, None);
    }

    fn test_adaptor_fold_per_thread<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let collection: Vec<Vec<u64>> = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .fold_per_thread(
                Vec::new,
                |mut vec, x| {
                    vec.push(x);
                    vec
                },
                Vec::with_capacity,
                |mut vecvec, vec| {
                    vecvec.push(vec);
                    vecvec
                },
            );

        let mut values: Vec<u64> = collection.into_iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, input);
    }

    fn test_adaptor_for_each<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_adaptor_for_each_init<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let mut values = (0..=INPUT_LEN).collect::<Vec<u64>>();
        values
            .par_iter_mut()
            .with_thread_pool(&mut thread_pool)
            .for_each_init(rand::rng, |rng, x| {
                if rng.random() {
                    *x *= 2
                };
            });

        let sum: u64 = values.iter().sum();
        assert!(sum >= INPUT_LEN * (INPUT_LEN + 1) / 2);
        assert!(sum <= INPUT_LEN * (INPUT_LEN + 1));
    }

    fn test_adaptor_map<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum1 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|&x| x * 42)
            .sum::<u64>();
        assert_eq!(sum1, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum2 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .map(|&x| x * 6)
            .map(|x| x * 7)
            .sum::<u64>();
        assert_eq!(sum2, 42 * INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum3 = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            // Mapping to a non-Send non-Sync type is fine, as the item stays on the same thread
            // and isn't shared with other threads.
            .map(|&x| Rc::new(x))
            .pipeline(|| 0, |acc, x| acc + *x, |acc| acc, |a, b| a + b);
        assert_eq!(sum3, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_max<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let mut input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let max = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max();
        assert_eq!(max, Some(INPUT_LEN));

        input.truncate(1);
        let max_one = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max();
        assert_eq!(max_one, Some(0));

        input.clear();
        let max_empty = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max();
        assert_eq!(max_empty, None);
    }

    fn test_adaptor_max_by<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // Custom comparison function where even numbers are smaller than all odd
        // numbers.
        let mut input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let max = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));

        let last_odd = ((INPUT_LEN - 1) / 2) * 2 + 1;
        assert_eq!(max, Some(last_odd));

        input.truncate(1);
        let max_one = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(max_one, Some(0));

        input.clear();
        let max_empty = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(max_empty, None);
    }

    fn test_adaptor_max_by_key<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let mut input = (0..=INPUT_LEN)
            .map(|x| (x, INPUT_LEN - x))
            .collect::<Vec<(u64, u64)>>();
        let max = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max_by_key(|pair| pair.1);
        assert_eq!(max, Some((0, INPUT_LEN)));

        input.truncate(1);
        let max_one = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max_by_key(|pair| pair.1);
        assert_eq!(max_one, Some((0, INPUT_LEN)));

        input.clear();
        let max_empty = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max_by_key(|pair| pair.1);
        assert_eq!(max_empty, None);
    }

    fn test_adaptor_min<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let mut input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let min = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min();
        assert_eq!(min, Some(0));

        input.truncate(1);
        let min_one = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min();
        assert_eq!(min_one, Some(0));

        input.clear();
        let min_empty = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min();
        assert_eq!(min_empty, None);
    }

    fn test_adaptor_min_by<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // Custom comparison function where even numbers are smaller than all odd
        // numbers.
        let mut input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let min = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));

        let first_even = 2;
        assert_eq!(min, Some(first_even));

        input.truncate(1);
        let min_one = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(min_one, Some(1));

        input.clear();
        let min_empty = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(min_empty, None);
    }

    fn test_adaptor_min_by_key<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let mut input = (0..=INPUT_LEN)
            .map(|x| (x, INPUT_LEN - x))
            .collect::<Vec<(u64, u64)>>();
        let min = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min_by_key(|pair| pair.1);
        assert_eq!(min, Some((INPUT_LEN, 0)));

        input.truncate(1);
        let min_one = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min_by_key(|pair| pair.1);
        assert_eq!(min_one, Some((0, INPUT_LEN)));

        input.clear();
        let min_empty = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min_by_key(|pair| pair.1);
        assert_eq!(min_empty, None);
    }

    fn test_adaptor_minmax<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let mut input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let minmax = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax();
        assert_eq!(
            minmax,
            MinMaxResult::MinMax {
                min: 0,
                max: INPUT_LEN
            }
        );

        input.truncate(1);
        let minmax_one = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax();
        assert_eq!(minmax_one, MinMaxResult::OneElement(0));

        input.clear();
        let minmax_empty = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax();
        assert_eq!(minmax_empty, MinMaxResult::NoElements);
    }

    fn test_adaptor_minmax_by<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        // Custom comparison function where even numbers are smaller than all odd
        // numbers.
        let mut input = (1..=INPUT_LEN).collect::<Vec<u64>>();
        let minmax = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));

        let first_even = 2;
        let last_odd = ((INPUT_LEN - 1) / 2) * 2 + 1;
        assert_eq!(
            minmax,
            MinMaxResult::MinMax {
                min: first_even,
                max: last_odd
            }
        );

        input.truncate(1);
        let minmax_one = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(minmax_one, MinMaxResult::OneElement(1));

        input.clear();
        let minmax_empty = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax_by(|x, y| (*x % 2).cmp(&(*y % 2)).then(x.cmp(y)));
        assert_eq!(minmax_empty, MinMaxResult::NoElements);
    }

    fn test_adaptor_minmax_by_key<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let mut input = (0..=INPUT_LEN)
            .map(|x| (x, INPUT_LEN - x))
            .collect::<Vec<(u64, u64)>>();
        let minmax = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax_by_key(|pair| pair.1);
        assert_eq!(
            minmax,
            MinMaxResult::MinMax {
                min: (INPUT_LEN, 0),
                max: (0, INPUT_LEN)
            }
        );

        input.truncate(1);
        let minmax_one = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax_by_key(|pair| pair.1);
        assert_eq!(minmax_one, MinMaxResult::OneElement((0, INPUT_LEN)));

        input.clear();
        let minmax_empty = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax_by_key(|pair| pair.1);
        assert_eq!(minmax_empty, MinMaxResult::NoElements);
    }

    #[cfg(feature = "default-thread-pool")]
    #[test]
    fn test_min_max_stability() {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..INPUT_LEN).map(|_| 0).collect::<Vec<u64>>();

        let min = input.par_iter().with_thread_pool(&mut thread_pool).min();
        assert_eq!(min, Some(&0));
        assert_eq!(
            min.unwrap() as *const u64,
            input.first().unwrap() as *const u64
        );

        let max = input.par_iter().with_thread_pool(&mut thread_pool).max();
        assert_eq!(max, Some(&0));
        assert_eq!(
            max.unwrap() as *const u64,
            input.last().unwrap() as *const u64
        );

        let minmax = input.par_iter().with_thread_pool(&mut thread_pool).minmax();
        assert_eq!(minmax, MinMaxResult::MinMax { min: &0, max: &0 });
        if let MinMaxResult::MinMax { min, max } = minmax {
            assert_eq!(min as *const u64, input.first().unwrap() as *const u64);
            assert_eq!(max as *const u64, input.last().unwrap() as *const u64);
        }
    }

    #[cfg(feature = "default-thread-pool")]
    #[test]
    fn test_min_max_by_stability() {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..INPUT_LEN).map(|_| 0).collect::<Vec<u64>>();
        let cmp = |x: &&u64, y: &&u64| (**x % 2).cmp(&(**y % 2)).then(x.cmp(y));

        let min = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .min_by(cmp);
        assert_eq!(min, Some(&0));
        assert_eq!(
            min.unwrap() as *const u64,
            input.first().unwrap() as *const u64
        );

        let max = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .max_by(cmp);
        assert_eq!(max, Some(&0));
        assert_eq!(
            max.unwrap() as *const u64,
            input.last().unwrap() as *const u64
        );

        let minmax = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .minmax_by(cmp);
        assert_eq!(minmax, MinMaxResult::MinMax { min: &0, max: &0 });
        if let MinMaxResult::MinMax { min, max } = minmax {
            assert_eq!(min as *const u64, input.first().unwrap() as *const u64);
            assert_eq!(max as *const u64, input.last().unwrap() as *const u64);
        }
    }

    #[cfg(feature = "default-thread-pool")]
    #[test]
    fn test_min_max_by_key_stability() {
        let mut thread_pool = ThreadPoolBuilder {
            num_threads: ThreadCount::AvailableParallelism,
            range_strategy: RangeStrategy::Fixed,
            cpu_pinning: CpuPinningPolicy::No,
        }
        .build();

        let input = (0..=INPUT_LEN).map(|i| (i, 0)).collect::<Vec<(u64, u64)>>();

        let min = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .min_by_key(|pair| pair.1);
        assert_eq!(min, Some((0, 0)));

        let max = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .max_by_key(|pair| pair.1);
        assert_eq!(max, Some((INPUT_LEN, 0)));

        let minmax = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .minmax_by_key(|pair| pair.1);
        assert_eq!(
            minmax,
            MinMaxResult::MinMax {
                min: (0, 0),
                max: (INPUT_LEN, 0)
            }
        );
    }

    fn test_adaptor_ne<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_adaptor_ne_by_key<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_adaptor_ne_by_keys<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
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

    fn test_adaptor_partial_cmp<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::cmp::Ordering;

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

    fn test_adaptor_partial_cmp_by<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::cmp::Ordering;

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

    fn test_adaptor_partial_cmp_by_key<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::cmp::Ordering;

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

    fn test_adaptor_partial_cmp_by_keys<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::cmp::Ordering;

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

    fn test_adaptor_product<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (1..=INPUT_LEN).map(|_| -1).collect::<Vec<i32>>();
        let product = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .product::<i32>();
        assert_eq!(product, if INPUT_LEN % 2 == 0 { 1 } else { -1 });
    }

    fn test_adaptor_reduce<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .reduce(|| 0, |x, y| x + y);
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_sum<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .sum::<u64>();
        assert_eq!(sum, INPUT_LEN * (INPUT_LEN + 1) / 2);
    }

    fn test_adaptor_try_collect<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let collection: Result<Vec<Box<u64>>, ()> = input
            .par_iter()
            .cloned()
            .map(Ok)
            .with_thread_pool(&mut thread_pool)
            .try_collect();
        assert_eq!(collection.unwrap(), input);

        let collection: Result<Vec<Box<u64>>, _> = input
            .par_iter()
            .cloned()
            .map(|x| if *x % 2 == 0 { Ok(x) } else { Err(x) })
            .with_thread_pool(&mut thread_pool)
            .try_collect();
        assert!(collection.is_err());
        assert!(*collection.unwrap_err() % 2 == 1);

        let collection: Result<Vec<Box<u64>>, _> = input
            .par_iter()
            .cloned()
            .map(|x| if *x != 7 { Ok(x) } else { Err(x) })
            .with_thread_pool(&mut thread_pool)
            .try_collect();
        assert!(collection.is_err());
        assert!(*collection.unwrap_err() == 7);
    }

    fn test_adaptor_try_collect_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN)
            .map(|x| Ok(Box::new(x)))
            .collect::<Vec<Result<Box<u64>, ()>>>();
        let _: Result<Vec<Box<u64>>, _> = input
            .into_par_iter()
            .inspect(|x| {
                if x.as_ref().is_ok_and(|x| **x % 10 == 9) {
                    panic!("arithmetic panic")
                }
            })
            .with_thread_pool(&mut thread_pool)
            .try_collect();
    }

    fn test_adaptor_try_collect_one_panic<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN)
            .map(|x| Ok(Box::new(x)))
            .collect::<Vec<Result<Box<u64>, ()>>>();
        let _: Result<Vec<Box<u64>>, _> = input
            .into_par_iter()
            .inspect(|x| {
                if x.as_ref().is_ok_and(|x| **x == 1) {
                    panic!("arithmetic panic")
                }
            })
            .with_thread_pool(&mut thread_pool)
            .try_collect();
    }

    #[cfg(feature = "nightly")]
    fn test_adaptor_try_collect_option<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).map(Box::new).collect::<Vec<Box<u64>>>();
        let collection: Option<Vec<Box<u64>>> = input
            .par_iter()
            .cloned()
            .map(Some)
            .with_thread_pool(&mut thread_pool)
            .try_collect();
        assert_eq!(collection.unwrap(), input);

        let collection: Option<Vec<Box<u64>>> = input
            .par_iter()
            .cloned()
            .map(|x| if *x % 2 == 0 { Some(x) } else { None })
            .with_thread_pool(&mut thread_pool)
            .try_collect();
        assert!(collection.is_none());

        let collection: Option<Vec<Box<u64>>> = input
            .par_iter()
            .cloned()
            .map(|x| if *x != 7 { Some(x) } else { None })
            .with_thread_pool(&mut thread_pool)
            .try_collect();
        assert!(collection.is_none());
    }

    fn test_adaptor_try_collect_per_thread<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let collection: Result<Vec<Vec<u64>>, ()> = input
            .par_iter()
            .map(|&x| Ok(x))
            .with_thread_pool(&mut thread_pool)
            .try_collect_per_thread();

        let mut values: Vec<u64> = collection.unwrap().into_iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, input);

        // The inner type can be any collection.
        let collection: Result<Vec<HashSet<u64>>, ()> = input
            .par_iter()
            .map(|&x| Ok(x))
            .with_thread_pool(&mut thread_pool)
            .try_collect_per_thread();

        let mut values: Vec<u64> = collection.unwrap().into_iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, input);

        let collection: Result<Vec<Vec<u64>>, u64> = input
            .par_iter()
            .map(|&x| if x % 2 == 0 { Ok(x) } else { Err(x) })
            .with_thread_pool(&mut thread_pool)
            .try_collect_per_thread();
        assert!(collection.is_err());
        assert!(collection.unwrap_err() % 2 == 1);
    }

    #[cfg(feature = "nightly")]
    fn test_adaptor_try_collect_per_thread_option<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let collection: Option<Vec<Vec<u64>>> = input
            .par_iter()
            .map(|&x| Some(x))
            .with_thread_pool(&mut thread_pool)
            .try_collect_per_thread();

        let mut values: Vec<u64> = collection.unwrap().into_iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, input);

        // The inner type can be any collection.
        let collection: Option<Vec<HashSet<u64>>> = input
            .par_iter()
            .map(|&x| Some(x))
            .with_thread_pool(&mut thread_pool)
            .try_collect_per_thread();

        let mut values: Vec<u64> = collection.unwrap().into_iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, input);

        let collection: Option<Vec<Vec<u64>>> = input
            .par_iter()
            .map(|&x| if x % 2 == 0 { Some(x) } else { None })
            .with_thread_pool(&mut thread_pool)
            .try_collect_per_thread();
        assert!(collection.is_none());
    }

    fn test_adaptor_try_fold_per_thread<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let collection: Result<Vec<Vec<u64>>, ()> = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_fold_per_thread(
                Vec::new,
                |mut vec, x| {
                    vec.push(x);
                    Ok(vec)
                },
                Vec::with_capacity,
                |mut vecvec, vec| {
                    vecvec.push(vec?);
                    Ok(vecvec)
                },
            );

        let mut values: Vec<u64> = collection.unwrap().into_iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, input);

        let collection: Result<Vec<Vec<u64>>, ()> = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_fold_per_thread(
                Vec::new,
                |mut vec, x| {
                    if x == INPUT_LEN {
                        Err(())
                    } else {
                        vec.push(x);
                        Ok(vec)
                    }
                },
                Vec::with_capacity,
                |mut vecvec, vec| {
                    vecvec.push(vec?);
                    Ok(vecvec)
                },
            );
        assert_eq!(collection, Err(()));

        let collection: Result<Vec<Vec<u64>>, ()> = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_fold_per_thread(
                Vec::new,
                |mut vec, x| {
                    if x % 2 == 1 {
                        Err(())
                    } else {
                        vec.push(x);
                        Ok(vec)
                    }
                },
                Vec::with_capacity,
                |mut vecvec, vec| {
                    vecvec.push(vec?);
                    Ok(vecvec)
                },
            );
        assert_eq!(collection, Err(()));
    }

    #[cfg(feature = "nightly")]
    fn test_adaptor_try_fold_per_thread_option<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let collection: Option<Vec<Vec<u64>>> = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_fold_per_thread(
                Vec::new,
                |mut vec, x| {
                    vec.push(x);
                    Some(vec)
                },
                Vec::with_capacity,
                |mut vecvec, vec| {
                    vecvec.push(vec?);
                    Some(vecvec)
                },
            );

        let mut values: Vec<u64> = collection.unwrap().into_iter().flatten().collect();
        values.sort_unstable();
        assert_eq!(values, input);

        let collection: Option<Vec<Vec<u64>>> = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_fold_per_thread(
                Vec::new,
                |mut vec, x| {
                    if x == INPUT_LEN {
                        None
                    } else {
                        vec.push(x);
                        Some(vec)
                    }
                },
                Vec::with_capacity,
                |mut vecvec, vec| {
                    vecvec.push(vec?);
                    Some(vecvec)
                },
            );
        assert_eq!(collection, None);

        let collection: Option<Vec<Vec<u64>>> = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_fold_per_thread(
                Vec::new,
                |mut vec, x| {
                    if x % 2 == 1 {
                        None
                    } else {
                        vec.push(x);
                        Some(vec)
                    }
                },
                Vec::with_capacity,
                |mut vecvec, vec| {
                    vecvec.push(vec?);
                    Some(vecvec)
                },
            );
        assert_eq!(collection, None);
    }

    fn test_adaptor_try_for_each<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::sync::atomic::Ordering;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each(|&x| -> Result<_, ()> {
                sum.fetch_add(x, Ordering::Relaxed);
                Ok(())
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
    fn test_adaptor_try_for_each_option<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::sync::atomic::Ordering;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each(|&x| {
                sum.fetch_add(x, Ordering::Relaxed);
                Some(())
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

    fn test_adaptor_try_for_each_init<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::sync::atomic::Ordering;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::rng, |rng, &x| -> Result<_, ()> {
                let y = rng.random_range(0..=x);
                sum.fetch_add(y, Ordering::Relaxed);
                Ok(())
            });
        assert_eq!(result, Ok(()));
        assert!(sum.into_inner() <= INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::rng, |rng, &x| {
                let y = rng.random_range(0..=INPUT_LEN);
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
            .try_for_each_init(rand::rng, |rng, &x| Err(x * rng.random_range(1..=10)));
        assert!(result.is_err());
        assert!(result.unwrap_err() <= 10 * INPUT_LEN);
    }

    #[cfg(feature = "nightly")]
    fn test_adaptor_try_for_each_init_option<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        use std::sync::atomic::Ordering;

        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::rng, |rng, &x| {
                let y = rng.random_range(0..=x);
                sum.fetch_add(y, Ordering::Relaxed);
                Some(())
            });
        assert_eq!(result, Some(()));
        assert!(sum.into_inner() <= INPUT_LEN * (INPUT_LEN + 1) / 2);

        let sum = AtomicU64::new(0);
        let result = input
            .par_iter()
            .with_thread_pool(&mut thread_pool)
            .try_for_each_init(rand::rng, |rng, &x| {
                let y = rng.random_range(0..=INPUT_LEN);
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
            .try_for_each_init(rand::rng, |_, _| None);
        assert!(result.is_none());
    }

    fn test_adaptor_try_reduce<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_reduce(|| 0, |x, y| x.checked_add(y).ok_or("integer overflow"));
        assert_eq!(sum, Ok(INPUT_LEN * (INPUT_LEN + 1) / 2));

        let product = input
            .par_iter()
            .skip(1)
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_reduce(|| 1, |x, y| x.checked_mul(y).ok_or("integer overflow"));
        assert_eq!(product, Err("integer overflow"));
    }

    #[cfg(feature = "nightly")]
    fn test_adaptor_try_reduce_option<T>(mut thread_pool: T)
    where
        for<'a> &'a mut T: GenericThreadPool,
    {
        let input = (0..=INPUT_LEN).collect::<Vec<u64>>();
        let sum = input
            .par_iter()
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_reduce(|| 0, |x, y| x.checked_add(y));
        assert_eq!(sum, Some(INPUT_LEN * (INPUT_LEN + 1) / 2));

        let product = input
            .par_iter()
            .skip(1)
            .copied()
            .with_thread_pool(&mut thread_pool)
            .try_reduce(|| 1, |x, y| x.checked_mul(y));
        assert_eq!(product, None);
    }

    /* Helper functions */
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

    fn vec_deque_is_contiguous<T>(v: &VecDeque<T>) -> bool {
        let (left, right) = v.as_slices();
        left.is_empty() || right.is_empty()
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
