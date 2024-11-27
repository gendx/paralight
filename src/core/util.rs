// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::ptr::NonNull;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};

/// An ergonomic wrapper around a [`Mutex`]-[`Condvar`] pair.
pub struct Status<T> {
    mutex: Mutex<T>,
    condvar: Condvar,
}

impl<T> Status<T> {
    /// Creates a new status initialized with the given value.
    pub fn new(t: T) -> Self {
        Self {
            mutex: Mutex::new(t),
            condvar: Condvar::new(),
        }
    }

    /// Attempts to set the status to the given value and notifies one waiting
    /// thread.
    ///
    /// Fails if the [`Mutex`] is poisoned.
    pub fn try_notify_one(&self, t: T) -> Result<(), PoisonError<MutexGuard<'_, T>>> {
        *self.mutex.lock()? = t;
        self.condvar.notify_one();
        Ok(())
    }

    /// Sets the status to the given value and notifies all waiting threads.
    pub fn notify_all(&self, t: T) {
        *self.mutex.lock().unwrap() = t;
        self.condvar.notify_all();
    }

    /// Waits until the predicate is true on this status.
    ///
    /// This returns a [`MutexGuard`], allowing to further inspect or modify the
    /// status.
    pub fn wait_while(&self, predicate: impl FnMut(&mut T) -> bool) -> MutexGuard<T> {
        self.condvar
            .wait_while(self.mutex.lock().unwrap(), predicate)
            .unwrap()
    }
}

/// A Proxy trait for types that have a lifetime parameter.
///
/// Because Rust doesn't directly support higher-kinded types, we use a generic
/// associated type with a lifetime parameter to represent that.
pub trait LifetimeParameterized {
    type T<'a>: ?Sized;
}

/// A lifetime-erased reference, where the underlying type is generic over a
/// lifetime. This acts as a [`&'a T<'a>`](reference) but whose lifetime can be
/// adjusted via the `unsafe` function [`get()`](Self::get).
pub struct DynLifetimeView<T: LifetimeParameterized> {
    ptr: Option<NonNull<T::T<'static>>>,
}

impl<T: LifetimeParameterized> DynLifetimeView<T> {
    /// Creates a new empty reference.
    pub fn empty() -> Self {
        Self { ptr: None }
    }

    /// Sets the underlying value to the given reference. Subsequent calls to
    /// [`get()`](Self::get) must ensure that the obtained reference doesn't
    /// outlive the reference that was set here.
    // The cast is necessary because the lifetime is coerced to 'static.
    #[allow(clippy::unnecessary_cast)]
    pub fn set(&mut self, value: &T::T<'_>) {
        self.ptr = NonNull::new(NonNull::from(value).as_ptr() as *mut T::T<'static>);
    }

    /// Clears the underlying reference. Subsequent calls to
    /// [`get()`](Self::get) will obtain [`None`].
    pub fn clear(&mut self) {
        self.ptr = None;
    }

    /// Returns the reference that was previously set with [`set()`](Self::set),
    /// or [`None`] if no reference was set or if the last reference was
    /// erased by a call to [`clear()`](Self::clear).
    ///
    /// # Safety
    ///
    /// The underlying object must be valid and not mutated during the whole
    /// output lifetime.
    // The cast is necessary because the lifetime is coerced to 'a.
    #[allow(clippy::unnecessary_cast)]
    pub unsafe fn get<'a>(&self) -> Option<&'a T::T<'a>> {
        self.ptr.map(|static_ptr| {
            let ptr = static_ptr.as_ptr() as *mut T::T<'a>;
            // SAFETY:
            // - This pointer points to a valid initialized `T`, as previously set via
            //   `set()`.
            // - The underlying `T` outlives the output lifetime, as ensured by the caller.
            // - The underlying `T` isn't mutated during the whole output lifetime, as
            //   ensured by the caller.
            unsafe { &*ptr }
        })
    }
}

/// SAFETY:
///
/// A [`DynLifetimeView`] acts as a [`&'a T<'a>`](reference). Therefore it is
/// [`Send`] if and only if `T<'_>` is [`Sync`].
unsafe impl<T: LifetimeParameterized> Send for DynLifetimeView<T> where for<'a> T::T<'a>: Sync {}
/// SAFETY:
///
/// A [`DynLifetimeView`] acts as a [`&'a T<'a>`](reference). Therefore it is
/// [`Sync`] if and only if `T<'_>` is [`Sync`].
unsafe impl<T: LifetimeParameterized> Sync for DynLifetimeView<T> where for<'a> T::T<'a>: Sync {}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::{Arc, Barrier, RwLock};

    // A type that doesn't have a lifetime parameter trivially implements
    // `LifetimeParameterized`.
    impl LifetimeParameterized for i32 {
        type T<'a> = Self;
    }

    #[test]
    fn view_basic_usage() {
        let mut view = DynLifetimeView::<i32>::empty();

        let mut foo = 42;
        view.set(&foo);
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(*bar, 42);

        foo = 1;
        view.set(&foo);
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(*bar, 1);

        let abc = 123;
        view.set(&abc);
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(*bar, 123);
    }

    #[test]
    fn view_multi_threaded() {
        const NUM_THREADS: usize = 2;

        let view = Arc::new(RwLock::new(DynLifetimeView::<i32>::empty()));
        let steps: Arc<[_; 6]> = Arc::new(std::array::from_fn(|_| Barrier::new(NUM_THREADS + 1)));

        let main = std::thread::spawn({
            let view = view.clone();
            let steps = steps.clone();
            move || {
                let mut foo = 42;
                view.write().unwrap().set(&foo);

                steps[0].wait();

                steps[1].wait();

                foo = 1;
                view.write().unwrap().set(&foo);

                steps[2].wait();

                steps[3].wait();

                let abc = 123;
                view.write().unwrap().set(&abc);

                steps[4].wait();

                steps[5].wait();
            }
        });

        let threads: [_; NUM_THREADS] = std::array::from_fn(move |_| {
            std::thread::spawn({
                let view = view.clone();
                let steps = steps.clone();
                move || {
                    steps[0].wait();

                    let guard = view.read().unwrap();
                    let reference = unsafe { guard.get().unwrap() };
                    assert_eq!(*reference, 42);
                    drop(guard);

                    steps[1].wait();

                    steps[2].wait();

                    let guard = view.read().unwrap();
                    let reference = unsafe { guard.get().unwrap() };
                    assert_eq!(*reference, 1);
                    drop(guard);

                    steps[3].wait();

                    steps[4].wait();

                    let guard = view.read().unwrap();
                    let reference = unsafe { guard.get().unwrap() };
                    assert_eq!(*reference, 123);
                    drop(guard);

                    steps[5].wait();
                }
            })
        });

        main.join().unwrap();
        for t in threads {
            t.join().unwrap();
        }
    }

    // This ignored test showcases how to misuse the unsafe API by mutating a value
    // while it is referenced. Running it under Miri returns a failure.
    #[ignore]
    #[cfg_attr(feature = "nightly_tests", coverage(off))]
    #[test]
    #[allow(unused_assignments)]
    fn view_bad_mut() {
        let mut view = DynLifetimeView::<i32>::empty();
        let mut foo = 42;
        view.set(&foo);
        let bar = unsafe { view.get().unwrap() };
        // Undefined behavior: This mutates `foo` while a reference to it `bar` is
        // active.
        foo = 1;
        assert_eq!(*bar, 1);
    }

    // This ignored test showcases how to misuse the unsafe API by obtaining a
    // reference whose lifetime extends beyond the underlying value's. Running it
    // under Miri returns a failure.
    #[ignore]
    #[cfg_attr(feature = "nightly_tests", coverage(off))]
    #[test]
    fn view_bad_lifetime() {
        let mut view = DynLifetimeView::<i32>::empty();
        {
            let foo = 42;
            view.set(&foo);
        }
        // Undefined behavior: This obtains a reference to `foo` which isn't live
        // anymore.
        let bar = unsafe { view.get().unwrap() };
        assert_ne!(*bar, 42);
    }

    impl LifetimeParameterized for &i32 {
        type T<'a> = &'a i32;
    }

    #[test]
    fn dyn_lifetime_view_basic_usage() {
        let mut view = DynLifetimeView::<&i32>::empty();

        let x = 42;
        let mut foo = &x;
        view.set(&foo);
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(**bar, 42);

        let y = 1;
        foo = &y;
        view.set(&foo);
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(**bar, 1);

        let z = 123;
        let abc = &z;
        view.set(&abc);
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(**bar, 123);
    }

    #[test]
    fn dyn_lifetime_view_multi_threaded() {
        const NUM_THREADS: usize = 2;

        let view = Arc::new(RwLock::new(DynLifetimeView::<&i32>::empty()));
        let steps: Arc<[_; 6]> = Arc::new(std::array::from_fn(|_| Barrier::new(NUM_THREADS + 1)));

        let main = std::thread::spawn({
            let view = view.clone();
            let steps = steps.clone();
            move || {
                let x = 42;
                let mut foo = &x;
                view.write().unwrap().set(&foo);

                steps[0].wait();

                steps[1].wait();

                let y = 1;
                foo = &y;
                view.write().unwrap().set(&foo);

                steps[2].wait();

                steps[3].wait();

                let z = 123;
                let abc = &z;
                view.write().unwrap().set(&abc);

                steps[4].wait();

                steps[5].wait();
            }
        });

        let threads: [_; NUM_THREADS] = std::array::from_fn(move |_| {
            std::thread::spawn({
                let view = view.clone();
                let steps = steps.clone();
                move || {
                    steps[0].wait();

                    let guard = view.read().unwrap();
                    let reference = unsafe { guard.get().unwrap() };
                    assert_eq!(**reference, 42);
                    drop(guard);

                    steps[1].wait();

                    steps[2].wait();

                    let guard = view.read().unwrap();
                    let reference = unsafe { guard.get().unwrap() };
                    assert_eq!(**reference, 1);
                    drop(guard);

                    steps[3].wait();

                    steps[4].wait();

                    let guard = view.read().unwrap();
                    let reference = unsafe { guard.get().unwrap() };
                    assert_eq!(**reference, 123);
                    drop(guard);

                    steps[5].wait();
                }
            })
        });

        main.join().unwrap();
        for t in threads {
            t.join().unwrap();
        }
    }

    // This ignored test showcases how to misuse the unsafe API by mutating a value
    // while it is referenced. Running it under Miri returns a failure.
    #[ignore]
    #[cfg_attr(feature = "nightly_tests", coverage(off))]
    #[test]
    #[allow(unused_assignments)]
    fn dyn_lifetime_view_bad_mut() {
        let mut view = DynLifetimeView::<&i32>::empty();
        let x = 42;
        let mut foo = &x;
        view.set(&foo);
        let bar = unsafe { view.get().unwrap() };
        let y = 1;
        // Undefined behavior: This mutates `foo` while a reference to it `bar` is
        // active.
        foo = &y;
        assert_eq!(**bar, 1);
    }

    // This ignored test showcases how to misuse the unsafe API by obtaining a
    // reference whose lifetime extends beyond the underlying value's. Running it
    // under Miri returns a failure.
    #[ignore]
    #[cfg_attr(feature = "nightly_tests", coverage(off))]
    #[test]
    fn dyn_lifetime_view_bad_lifetime() {
        let x = 42;
        let mut view = DynLifetimeView::<&i32>::empty();
        {
            let foo = &x;
            view.set(&foo);
        }
        // Undefined behavior: This obtains a reference to `foo` which isn't live
        // anymore.
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(**bar, 42);
    }
}
