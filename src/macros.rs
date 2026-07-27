// Copyright 2024 The Paralight Authors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Internal macros, to swap the logging macros implementation based on whether
//! the `log` feature is enabled or not.

#![allow(unused_imports, unused_macros)]

#[cfg(feature = "log")]
macro_rules! log_debug {
    ( $($args:tt)* ) => {
        log::debug!( $($args)* )
    }
}

#[cfg(feature = "log")]
macro_rules! log_error {
    ( $($args:tt)* ) => {
        log::error!( $($args)* )
    }
}

#[cfg(feature = "log")]
macro_rules! log_info {
    ( $($args:tt)* ) => {
        log::info!( $($args)* )
    }
}

#[cfg(feature = "log")]
macro_rules! log_trace {
    ( $($args:tt)* ) => {
        log::trace!( $($args)* )
    }
}

#[cfg(feature = "log")]
macro_rules! log_warn {
    ( $($args:tt)* ) => {
        log::warn!( $($args)* )
    }
}

#[cfg(not(feature = "log"))]
macro_rules! log_debug {
    ( $($args:tt)* ) => {
        ()
    };
}

#[cfg(not(feature = "log"))]
macro_rules! log_error {
    ( $($args:tt)* ) => {
        ()
    };
}

#[cfg(not(feature = "log"))]
macro_rules! log_info {
    ( $($args:tt)* ) => {
        ()
    };
}

#[cfg(not(feature = "log"))]
macro_rules! log_trace {
    ( $($args:tt)* ) => {
        ()
    };
}

#[cfg(not(feature = "log"))]
macro_rules! log_warn {
    ( $($args:tt)* ) => {
        ()
    };
}

pub(crate) use log_debug;
pub(crate) use log_error;
pub(crate) use log_info;
pub(crate) use log_trace;
pub(crate) use log_warn;
