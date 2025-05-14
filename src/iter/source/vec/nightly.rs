// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! TODO: This file should be merged with the parent module. However, using
//! unstable syntax from stable Rust [is currently
//! disallowed](https://github.com/rust-lang/rust/issues/140960), even when put
//! behind a feature flag. However, this restriction doesn't apply when the
//! unstable syntax is in a separate file that is cfg-ed away on stable Rust.

use super::super::{RewindableSource, SourceCleanup};
use super::{VecParallelSource, VecSourceDescriptor};

// TODO: This compiles but using it is effectively blocked on
// https://github.com/rust-lang/rust/issues/137813.
// SAFETY: TODO
unsafe impl<T: Send> RewindableSource for VecParallelSource<T> where
    VecSourceDescriptor<T>: SourceCleanup<NEEDS_CLEANUP = false> /* TODO: Should the constraint
                                                                  * be stronger (e.g. T: Copy)? */
{
}
