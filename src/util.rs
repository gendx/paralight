// Copyright 2026 The Paralight Authors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generally useful functions.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, BitAnd, Div, Mul, Rem, Shl, Shr, Sub};

/// A divider stores pre-computed values to speed up divisions and modulos by a
/// given constant.
///
/// ```ignore
/// let x = Divider::new(7);
/// assert_eq!(42 / x, 6);
/// assert_eq!(42 % x, 0);
/// ```
///
/// # References
///
/// - <https://en.wikipedia.org/wiki/Division_algorithm#Division_by_a_constant>
/// - <https://rubenvannieuwpoort.nl/posts/division-by-constant-unsigned-integers>
/// - <https://ridiculousfish.com/blog/posts/labor-of-division-episode-i.html>
/// - <https://ridiculousfish.com/blog/posts/labor-of-division-episode-iii.html>
/// - <https://oeis.org/A346495>
/// - <https://oeis.org/A346496>
#[derive(Debug, Default, Clone, Copy)]
pub struct Divider<T> {
    /// Original divisor.
    divisor: T,
    /// Magic multiplier to compute divisions by this divisor. Set to zero for
    /// powers of 2 and non-zero otherwise.
    multiplier: T,
    /// Mask to compute remainders modulo this divisor. Only used for powers of
    /// 2.
    mask: T,
    /// Shift to compute divisions by this divisor.
    /// - For powers of 2, this is simply applied to the input.
    /// - For non-powers of 2, this is applied to the high word of the magic
    ///   multiplication, after adjusting for carries.
    shift: u32,
}

impl<T> Divider<T>
where
    T: Copy
        + Eq
        + Ord
        + Debug
        + Add<Output = T>
        + Sub<Output = T>
        + AddAssign
        + Shl<u32, Output = T>
        + Shr<u32, Output = T>
        + Arithmetic,
{
    pub fn new(divisor: T) -> Option<Self> {
        // Division by 0 is caught here.
        let log2 = divisor.checked_ilog2()?;

        // Power of 2.
        if divisor == T::one() << log2 {
            return Some(Divider {
                divisor,
                multiplier: T::zero(),
                mask: divisor - T::one(),
                shift: log2,
            });
        }

        // See https://rubenvannieuwpoort.nl/posts/division-by-constant-unsigned-integers:
        //  multiplier = 2^(N+shift) / d
        //  rem = 2^(N+shift) % d
        let (multiplier, rem) = T::narrowing_div_rem((T::zero(), T::one() << log2), divisor);
        debug_assert_ne!(multiplier, T::zero());
        debug_assert!(rem > T::zero());
        debug_assert!(rem < divisor);

        // At this point the highest bit of multiplier is set (because we divided
        // 2^(N+log2(d)) / d), therefore shifting discards it:
        //  multiplier = 2 * (2^(N+shift) / d) - 2^N
        debug_assert_eq!(multiplier >> (T::BITS - 1), T::one());
        let mut multiplier = multiplier << 1;
        let twice_rem = rem << 1;
        // Use the remainder to adjust the multiplier to:
        //  multiplier = 2^(N+shift+1) / d - 2^N
        if twice_rem >= divisor || twice_rem < rem {
            multiplier += T::one();
        }

        // Lastly, we compute the ceiling of that:
        //  multiplier = 2^(N+shift+1) / d + 1 - 2^N
        // Because d isn't a power of 2 (and therefore doesn't divide 2^(N+shift+1)),
        // this gives:
        //  multiplier = ceil(2^(N+shift+1) / d) - 2^N
        Some(Divider {
            divisor,
            multiplier: multiplier + T::one(),
            mask: T::zero(),
            shift: log2,
        })
    }
}

impl<T> Divider<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Shr<u32, Output = T> + Arithmetic,
{
    // Not suitable for powers of 2 (as the shift is adjusted differently).
    #[inline(always)]
    fn div_non_power_of_two(&self, x: T) -> T {
        // See https://rubenvannieuwpoort.nl/posts/division-by-constant-unsigned-integers:
        //  multiplier = m - 2^N
        //  hi = (x*m - x*2^N) >> N = x*m / 2^N - x
        //  y = (2x - x*m / 2^N) / 2 + x*m / 2^N - x
        //    = x*m / 2^(N+1)
        //  y >> shift = x*m / 2^(N+shift+1)
        let (_, hi) = x.widening_mul(self.multiplier);
        let y = ((x - hi) >> 1) + hi;
        y >> self.shift
    }
}

impl<T> Divider<T>
where
    T: Copy
        + Eq
        + Add<Output = T>
        + Sub<Output = T>
        + Mul<Output = T>
        + BitAnd<Output = T>
        + Shr<u32, Output = T>
        + Arithmetic,
{
    #[inline(always)]
    pub fn div_rem(&self, x: T) -> (T, T) {
        if self.multiplier == T::zero() {
            (x >> self.shift, x & self.mask)
        } else {
            let q = self.div_non_power_of_two(x);
            let r = x - q * self.divisor;
            (q, r)
        }
    }
}

// Note: The orphan rule prevents us from implementing this generically.
// For more information about this error, try `rustc --explain E0210`.
impl Div<Divider<Self>> for usize {
    type Output = Self;

    #[inline(always)]
    fn div(self, divider: Divider<Self>) -> Self::Output {
        if divider.multiplier == 0 {
            self >> divider.shift
        } else {
            divider.div_non_power_of_two(self)
        }
    }
}

// Note: The orphan rule prevents us from implementing this generically.
// For more information about this error, try `rustc --explain E0210`.
impl Rem<Divider<Self>> for usize {
    type Output = Self;

    #[inline(always)]
    fn rem(self, divider: Divider<Self>) -> Self::Output {
        if divider.multiplier == 0 {
            self & divider.mask
        } else {
            let q = divider.div_non_power_of_two(self);
            self - q * divider.divisor
        }
    }
}

pub trait Arithmetic: Sized {
    const BITS: u32;

    /// Returns zero.
    fn zero() -> Self;

    /// Returns one.
    fn one() -> Self;

    /// Returns the base-2 logarithm or `None` if `self` is zero.
    fn checked_ilog2(self) -> Option<u32>;

    /// Returns (low, high) of multiplying `self` by `other`.
    fn widening_mul(self, other: Self) -> (Self, Self);

    /// Divides (num_lo, num_hi) by denom, returning (quotient, remainder),
    /// assuming that the quotient fits in Self.
    // TODO: Make this unsafe and optimize to a DIV instruction on x86.
    fn narrowing_div_rem(num: (Self, Self), denom: Self) -> (Self, Self);
}

impl Arithmetic for u16 {
    const BITS: u32 = u16::BITS;

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn checked_ilog2(self) -> Option<u32> {
        self.checked_ilog2()
    }

    #[inline(always)]
    fn widening_mul(self, other: Self) -> (Self, Self) {
        self.carrying_mul(other, 0)
    }

    #[inline(always)]
    fn narrowing_div_rem((num_lo, num_hi): (Self, Self), denom: Self) -> (Self, Self) {
        let a = ((num_hi as u32) << 16) | (num_lo as u32);
        let b = denom as u32;
        let quo = a / b;
        let rem = a - quo * b;
        (quo as u16, rem as u16)
    }
}

impl Arithmetic for u32 {
    const BITS: u32 = u32::BITS;

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn checked_ilog2(self) -> Option<u32> {
        self.checked_ilog2()
    }

    #[inline(always)]
    fn widening_mul(self, other: Self) -> (Self, Self) {
        self.carrying_mul(other, 0)
    }

    #[inline(always)]
    fn narrowing_div_rem((num_lo, num_hi): (Self, Self), denom: Self) -> (Self, Self) {
        let a = ((num_hi as u64) << 32) | (num_lo as u64);
        let b = denom as u64;
        let quo = a / b;
        let rem = a - quo * b;
        (quo as u32, rem as u32)
    }
}

impl Arithmetic for u64 {
    const BITS: u32 = u64::BITS;

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn checked_ilog2(self) -> Option<u32> {
        self.checked_ilog2()
    }

    #[inline(always)]
    fn widening_mul(self, other: Self) -> (Self, Self) {
        self.carrying_mul(other, 0)
    }

    #[inline(always)]
    fn narrowing_div_rem((num_lo, num_hi): (Self, Self), denom: Self) -> (Self, Self) {
        let a = ((num_hi as u128) << 64) | (num_lo as u128);
        let b = denom as u128;
        let quo = a / b;
        let rem = a - quo * b;
        (quo as u64, rem as u64)
    }
}

impl Arithmetic for usize {
    const BITS: u32 = usize::BITS;

    #[inline(always)]
    fn zero() -> Self {
        0
    }

    #[inline(always)]
    fn one() -> Self {
        1
    }

    #[inline(always)]
    fn checked_ilog2(self) -> Option<u32> {
        self.checked_ilog2()
    }

    #[inline(always)]
    fn widening_mul(self, other: Self) -> (Self, Self) {
        self.carrying_mul(other, 0)
    }

    #[inline(always)]
    fn narrowing_div_rem(num: (Self, Self), denom: Self) -> (Self, Self) {
        // TODO(MSRV >= 1.95.0): Use cfg_select! macro.
        #[cfg(target_pointer_width = "16")]
        {
            let (quo, rem) = u16::narrowing_div_rem((num.0 as u16, num.1 as u16), denom as u16);
            (quo as usize, rem as usize)
        }
        #[cfg(target_pointer_width = "32")]
        {
            let (quo, rem) = u32::narrowing_div_rem((num.0 as u32, num.1 as u32), denom as u32);
            (quo as usize, rem as usize)
        }
        #[cfg(target_pointer_width = "64")]
        {
            let (quo, rem) = u64::narrowing_div_rem((num.0 as u64, num.1 as u64), denom as u64);
            (quo as usize, rem as usize)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_divider_usize_div() {
        for d in (1..100)
            .chain(usize::MAX - 100..=usize::MAX)
            .chain((0..usize::BITS).map(|i| 1 << i))
            .chain((1..usize::BITS).map(|i| (1 << i) - 1))
        {
            let divider = Divider::new(d).unwrap();
            for x in (0..100)
                .chain(usize::MAX - 100..=usize::MAX)
                .chain((0..usize::BITS).map(|i| 1 << i))
                .chain((0..usize::BITS).map(|i| (1 << i) - 1))
            {
                assert_eq!(
                    x / divider,
                    x / d,
                    "mismatch for {x} / {d} | {x:x?} / {d:x?}, divider = {divider:?} | {divider:x?}"
                );
            }
        }
    }

    #[test]
    fn test_divider_usize_rem() {
        for d in (1..100)
            .chain(usize::MAX - 100..=usize::MAX)
            .chain((0..usize::BITS).map(|i| 1 << i))
            .chain((1..usize::BITS).map(|i| (1 << i) - 1))
        {
            let divider = Divider::new(d).unwrap();
            for x in (0..100)
                .chain(usize::MAX - 100..=usize::MAX)
                .chain((0..usize::BITS).map(|i| 1 << i))
                .chain((0..usize::BITS).map(|i| (1 << i) - 1))
            {
                assert_eq!(
                    x % divider,
                    x % d,
                    "mismatch for {x} % {d} | {x:x?} % {d:x?}, divider = {divider:?} | {divider:x?}"
                );
            }
        }
    }

    #[test]
    fn test_divider_usize_div_rem() {
        for d in (1..100)
            .chain(usize::MAX - 100..=usize::MAX)
            .chain((0..usize::BITS).map(|i| 1 << i))
            .chain((1..usize::BITS).map(|i| (1 << i) - 1))
        {
            let divider = Divider::new(d).unwrap();
            for x in (0..100)
                .chain(usize::MAX - 100..=usize::MAX)
                .chain((0..usize::BITS).map(|i| 1 << i))
                .chain((0..usize::BITS).map(|i| (1 << i) - 1))
            {
                assert_eq!(
                    divider.div_rem(x),
                    (x / d, x % d),
                    "mismatch for {x} div_rem {d} | {x:x?} div_rem {d:x?}, divider = {divider:?} | {divider:x?}"
                );
            }
        }
    }

    #[test]
    fn test_divider_u16_div_rem() {
        for d in (1..100)
            .chain(u16::MAX - 100..=u16::MAX)
            .chain((0..u16::BITS).map(|i| 1 << i))
            .chain((1..u16::BITS).map(|i| (1 << i) - 1))
        {
            let divider = Divider::new(d).unwrap();
            for x in (0..100)
                .chain(u16::MAX - 100..=u16::MAX)
                .chain((0..u16::BITS).map(|i| 1 << i))
                .chain((0..u16::BITS).map(|i| (1 << i) - 1))
            {
                assert_eq!(
                    divider.div_rem(x),
                    (x / d, x % d),
                    "mismatch for {x} div_rem {d} | {x:x?} div_rem {d:x?}, divider = {divider:?} | {divider:x?}"
                );
            }
        }
    }

    #[test]
    fn test_divider_u32_div_rem() {
        for d in (1..100)
            .chain(u32::MAX - 100..=u32::MAX)
            .chain((0..u32::BITS).map(|i| 1 << i))
            .chain((1..u32::BITS).map(|i| (1 << i) - 1))
        {
            let divider = Divider::new(d).unwrap();
            for x in (0..100)
                .chain(u32::MAX - 100..=u32::MAX)
                .chain((0..u32::BITS).map(|i| 1 << i))
                .chain((0..u32::BITS).map(|i| (1 << i) - 1))
            {
                assert_eq!(
                    divider.div_rem(x),
                    (x / d, x % d),
                    "mismatch for {x} div_rem {d} | {x:x?} div_rem {d:x?}, divider = {divider:?} | {divider:x?}"
                );
            }
        }
    }

    #[test]
    fn test_divider_u64_div_rem() {
        for d in (1..100)
            .chain(u64::MAX - 100..=u64::MAX)
            .chain((0..u64::BITS).map(|i| 1 << i))
            .chain((1..u64::BITS).map(|i| (1 << i) - 1))
        {
            let divider = Divider::new(d).unwrap();
            for x in (0..100)
                .chain(u64::MAX - 100..=u64::MAX)
                .chain((0..u64::BITS).map(|i| 1 << i))
                .chain((0..u64::BITS).map(|i| (1 << i) - 1))
            {
                assert_eq!(
                    divider.div_rem(x),
                    (x / d, x % d),
                    "mismatch for {x} div_rem {d} | {x:x?} div_rem {d:x?}, divider = {divider:?} | {divider:x?}"
                );
            }
        }
    }
}
