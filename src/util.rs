// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// A lifetime-erased slice. This acts as a [`&[T]`](slice) but whose lifetime
/// can be adjusted via the `unsafe` function [`get()`](Self::get).
pub struct SliceView<T> {
    ptr: *const T,
    len: usize,
}

impl<T> SliceView<T> {
    /// Creates a new empty slice.
    pub fn new() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    /// Sets the underlying value to the given slice. Subsequent calls to
    /// [`get()`](Self::get) must ensure that the obtained slice doesn't
    /// outlive the slice that was set here.
    pub fn set(&mut self, slice: &[T]) {
        self.ptr = slice.as_ptr();
        self.len = slice.len();
    }

    /// Resets the underlying value to the empty slice. Subsequent calls to
    /// [`get()`](Self::get) will obtain [`None`].
    pub fn clear(&mut self) {
        self.ptr = std::ptr::null();
        self.len = 0;
    }

    /// Return the slice that was previously set with [`set()`](Self::set), or
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
    fn slice_view_basic_usage() {
        let mut view = SliceView::new();

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

        let view = Arc::new(RwLock::new(SliceView::new()));
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
        let mut view = SliceView::new();
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
        let mut view = SliceView::new();
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
