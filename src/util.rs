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

/// A lifetime-erased reference. This acts as a [`&T`](reference) but whose
/// lifetime can be adjusted via the `unsafe` function [`get()`](Self::get).
pub struct View<T: ?Sized> {
    ptr: Option<NonNull<T>>,
}

impl<T: ?Sized> View<T> {
    /// Creates a new empty reference.
    pub fn empty() -> Self {
        Self { ptr: None }
    }

    /// Sets the underlying value to the given reference. Subsequent calls to
    /// [`get()`](Self::get) must ensure that the obtained reference doesn't
    /// outlive the reference that was set here.
    pub fn set(&mut self, value: &T) {
        self.ptr = Some(NonNull::from(value));
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
    pub unsafe fn get(&self) -> Option<&T> {
        self.ptr.map(|ptr| {
            // SAFETY:
            // - This pointer points to a valid initialized `T`, as previously set via
            //   `set()`.
            // - The underlying `T` outlives the output lifetime outlives, as ensured by the
            //   caller.
            // - The underlying `T` isn't mutated during the whole output lifetime, as
            //   ensured by the caller.
            unsafe { ptr.as_ref() }
        })
    }
}

/// SAFETY:
///
/// A [`View`] acts as a [`&T`](reference). Therefore it is [`Send`] if and only
/// if `T` is [`Sync`].
unsafe impl<T: ?Sized + Sync> Send for View<T> {}
/// SAFETY:
///
/// A [`View`] acts as a [`&T`](reference). Therefore it is [`Sync`] if and only
/// if `T` is [`Sync`].
unsafe impl<T: ?Sized + Sync> Sync for View<T> {}

/// A lifetime-erased slice. This acts as a [`&[T]`](slice) but whose lifetime
/// can be adjusted via the `unsafe` function [`get()`](Self::get).
pub struct SliceView<T> {
    ptr: *const T,
    len: usize,
}

impl<T> SliceView<T> {
    /// Creates a new empty slice.
    #[cfg(test)]
    pub fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    /// Creates a new view set to the given slice. Like with
    /// [`set()`](Self::set), subsequent calls to [`get()`](Self::get) must
    /// ensure that the obtained slice doesn't outlive the slice that was set
    /// here.
    pub fn new(slice: &[T]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }

    /// Sets the underlying value to the given slice. Subsequent calls to
    /// [`get()`](Self::get) must ensure that the obtained slice doesn't
    /// outlive the slice that was set here.
    #[cfg(test)]
    pub fn set(&mut self, slice: &[T]) {
        self.ptr = slice.as_ptr();
        self.len = slice.len();
    }

    /// Resets the underlying value to the empty slice. Subsequent calls to
    /// [`get()`](Self::get) will obtain [`None`].
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.ptr = std::ptr::null();
        self.len = 0;
    }

    /// Returns the slice that was previously set with [`set()`](Self::set), or
    /// [`None`] if no slice was set or if the last slice was erased by a
    /// call to [`clear()`](Self::clear).
    ///
    /// # Safety
    ///
    /// The underlying slice must be valid and not mutated during the whole
    /// output lifetime.
    pub unsafe fn get(&self) -> Option<&[T]> {
        if self.ptr.is_null() {
            None
        } else {
            // SAFETY:
            //
            // - The pointer-length pair points to valid, initialized memory, as previously
            //   obtained via the set() function (the only way for the pointer to be
            //   non-null).
            // - The memory referenced by the returned slice isn't mutated during its
            //   lifetime, which follows from this function's safety requirements.
            unsafe { Some(std::slice::from_raw_parts(self.ptr, self.len)) }
        }
    }
}

/// SAFETY:
///
/// A [`SliceView`] acts as a [`&[T]`](slice). Therefore it is [`Send`] if and
/// only if [`[T]`](slice) is [`Sync`], which is if and only if `T` is [`Sync`].
unsafe impl<T: Sync> Send for SliceView<T> {}
/// SAFETY:
///
/// A [`SliceView`] acts as a [`&[T]`](slice). Therefore it is [`Sync`] if and
/// only if [`[T]`](slice) is [`Sync`], which is if and only if `T` is [`Sync`].
unsafe impl<T: Sync> Sync for SliceView<T> {}

#[cfg(test)]
mod test {
    use super::*;
    use std::sync::{Arc, Barrier, RwLock};

    #[test]
    fn view_basic_usage() {
        let mut view = View::empty();

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

        let view = Arc::new(RwLock::new(View::empty()));
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
    #[test]
    #[allow(unused_assignments)]
    fn view_bad_mut() {
        let mut view = View::empty();
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
    #[test]
    fn view_bad_lifetime() {
        let mut view = View::empty();
        {
            let foo = 42;
            view.set(&foo);
        }
        // Undefined behavior: This obtains a reference to `foo` which isn't live
        // anymore.
        let bar = unsafe { view.get().unwrap() };
        assert_ne!(*bar, 42);
    }

    #[test]
    fn slice_view_basic_usage() {
        let mut view = SliceView::empty();

        let mut foo = [42; 5];
        view.set(&foo);
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(bar, [42; 5]);

        foo[0] = 1;
        view.set(&foo);
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(bar, [1, 42, 42, 42, 42]);

        let abc = [123; 10];
        view.set(&abc);
        let bar = unsafe { view.get().unwrap() };
        assert_eq!(bar, [123; 10]);
    }

    #[test]
    fn slice_view_multi_threaded() {
        const NUM_THREADS: usize = 2;

        let view = Arc::new(RwLock::new(SliceView::empty()));
        let steps: Arc<[_; 6]> = Arc::new(std::array::from_fn(|_| Barrier::new(NUM_THREADS + 1)));

        let main = std::thread::spawn({
            let view = view.clone();
            let steps = steps.clone();
            move || {
                let mut foo = [42; 5];
                view.write().unwrap().set(&foo);

                steps[0].wait();

                steps[1].wait();

                foo[0] = 1;
                view.write().unwrap().set(&foo);

                steps[2].wait();

                steps[3].wait();

                let abc = [123; 10];
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
                    let slice = unsafe { guard.get().unwrap() };
                    assert_eq!(slice, [42; 5]);
                    drop(guard);

                    steps[1].wait();

                    steps[2].wait();

                    let guard = view.read().unwrap();
                    let slice = unsafe { guard.get().unwrap() };
                    assert_eq!(slice, [1, 42, 42, 42, 42]);
                    drop(guard);

                    steps[3].wait();

                    steps[4].wait();

                    let guard = view.read().unwrap();
                    let slice = unsafe { guard.get().unwrap() };
                    assert_eq!(slice, [123; 10]);
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

    // This ignored test showcases how to misuse the unsafe API by mutating a slice
    // while it is referenced. Running it under Miri returns a failure.
    #[ignore]
    #[test]
    fn slice_view_bad_mut() {
        let mut view = SliceView::empty();
        let mut foo = [42; 5];
        view.set(&foo);
        let bar = unsafe { view.get().unwrap() };
        // Undefined behavior: This mutates the array `foo` while a reference to it
        // `bar` is active.
        foo[0] = 1;
        assert_eq!(bar, [1, 42, 42, 42, 42]);
    }

    // This ignored test showcases how to misuse the unsafe API by obtaining a
    // reference whose lifetime extends beyond the underlying slice's. Running it
    // under Miri returns a failure.
    #[ignore]
    #[test]
    fn slice_view_bad_lifetime() {
        let mut view = SliceView::empty();
        {
            let foo = [42; 5];
            view.set(&foo);
        }
        // Undefined behavior: This obtains a reference to `foo` which isn't live
        // anymore.
        let bar = unsafe { view.get().unwrap() };
        assert_ne!(bar, [42; 5]);
    }
}
