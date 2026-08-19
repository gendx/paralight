// Copyright 2026 The Paralight Authors
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// https://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or https://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Generally useful functions.

use std::fmt::Debug;
use std::ops::{Add, BitAnd, Div, Rem, Shl, Shr, Sub};

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
    /// Adjusting term added to the magic multiplication.
    add: T,
    /// Mask to compute remainders modulo this divisor. Only used for powers of
    /// 2.
    mask: T,
    /// Shift to compute divisions by this divisor, applied to the high word of
    /// the magic multiply-and-add operation.
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
        + Shl<u32, Output = T>
        + Arithmetic,
{
    pub fn new(divisor: T) -> Option<Self> {
        // Division by 0 is caught here.
        let log2 = divisor.checked_ilog2()?;

        // Power of 2.
        if divisor == T::ONE << log2 {
            return Some(Divider {
                divisor,
                multiplier: T::ZERO,
                add: T::ZERO,
                mask: divisor - T::ONE,
                shift: log2,
            });
        }

        // See https://rubenvannieuwpoort.nl/posts/division-by-constant-unsigned-integers:
        //   mul_down = floor(2^(N+shift) / d)
        let (mul_down, _) = T::narrowing_div_rem((T::ZERO, T::ONE << log2), divisor);
        debug_assert_ne!(mul_down, T::ZERO);

        // Given that d isn't a power of 2, it doesn't divide 2^(N+shift), therefore:
        //   mul_up = ceil(2^(N+shift) / d)
        let mul_up = mul_down + T::ONE;
        debug_assert_ne!(mul_up, T::ZERO);

        // Lemma 11 in https://rubenvannieuwpoort.nl/posts/division-by-constant-unsigned-integers:
        if mul_up.wrapping_mul(divisor) <= T::ONE << log2 {
            // Use round-up method.
            Some(Divider {
                divisor,
                multiplier: mul_up,
                add: T::ZERO,
                mask: T::ZERO,
                shift: log2,
            })
        } else {
            // Use round-down method.
            Some(Divider {
                divisor,
                multiplier: mul_down,
                add: mul_down,
                mask: T::ZERO,
                shift: log2,
            })
        }
    }
}

impl<T> Divider<T>
where
    T: Copy + Eq + BitAnd<Output = T> + Shr<u32, Output = T> + Arithmetic,
{
    #[inline(always)]
    pub fn div_rem(&self, x: T) -> (T, T) {
        if self.multiplier == T::ZERO {
            (x >> self.shift, x & self.mask)
        } else {
            let (_, high) = x.mul_add(self.multiplier, self.add);
            let q = high >> self.shift;
            let r = x.wrapping_sub(q.wrapping_mul(self.divisor));
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
            let (_, high) = self.mul_add(divider.multiplier, divider.add);
            high >> divider.shift
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
            let (_, high) = self.mul_add(divider.multiplier, divider.add);
            let q = high >> divider.shift;
            self.wrapping_sub(q.wrapping_mul(divider.divisor))
        }
    }
}

pub trait Arithmetic: Sized {
    const BITS: u32;
    const ZERO: Self;
    const ONE: Self;

    /// Returns the base-2 logarithm or `None` if `self` is zero.
    fn checked_ilog2(self) -> Option<u32>;

    fn wrapping_sub(self, other: Self) -> Self;

    fn wrapping_mul(self, other: Self) -> Self;

    /// Returns (low, high) of `self * mul + add`.
    fn mul_add(self, mul: Self, add: Self) -> (Self, Self);

    /// Divides (num_lo, num_hi) by denom, returning (quotient, remainder),
    /// assuming that the quotient fits in Self.
    // TODO: Make this unsafe and optimize to a DIV instruction on x86.
    fn narrowing_div_rem(num: (Self, Self), denom: Self) -> (Self, Self);
}

impl Arithmetic for u8 {
    const BITS: u32 = u8::BITS;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline(always)]
    fn checked_ilog2(self) -> Option<u32> {
        self.checked_ilog2()
    }

    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }

    #[inline(always)]
    fn wrapping_mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }

    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> (Self, Self) {
        self.carrying_mul(mul, add)
    }

    #[inline(always)]
    fn narrowing_div_rem((num_lo, num_hi): (Self, Self), denom: Self) -> (Self, Self) {
        let a = ((num_hi as u16) << 8) | (num_lo as u16);
        let b = denom as u16;
        let quo = a / b;
        let rem = a.wrapping_sub(quo.wrapping_mul(b));
        (quo as u8, rem as u8)
    }
}

impl Arithmetic for u16 {
    const BITS: u32 = u16::BITS;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline(always)]
    fn checked_ilog2(self) -> Option<u32> {
        self.checked_ilog2()
    }

    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }

    #[inline(always)]
    fn wrapping_mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }

    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> (Self, Self) {
        self.carrying_mul(mul, add)
    }

    #[inline(always)]
    fn narrowing_div_rem((num_lo, num_hi): (Self, Self), denom: Self) -> (Self, Self) {
        let a = ((num_hi as u32) << 16) | (num_lo as u32);
        let b = denom as u32;
        let quo = a / b;
        let rem = a.wrapping_sub(quo.wrapping_mul(b));
        (quo as u16, rem as u16)
    }
}

impl Arithmetic for u32 {
    const BITS: u32 = u32::BITS;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline(always)]
    fn checked_ilog2(self) -> Option<u32> {
        self.checked_ilog2()
    }

    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }

    #[inline(always)]
    fn wrapping_mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }

    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> (Self, Self) {
        self.carrying_mul(mul, add)
    }

    #[inline(always)]
    fn narrowing_div_rem((num_lo, num_hi): (Self, Self), denom: Self) -> (Self, Self) {
        let a = ((num_hi as u64) << 32) | (num_lo as u64);
        let b = denom as u64;
        let quo = a / b;
        let rem = a.wrapping_sub(quo.wrapping_mul(b));
        (quo as u32, rem as u32)
    }
}

impl Arithmetic for u64 {
    const BITS: u32 = u64::BITS;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline(always)]
    fn checked_ilog2(self) -> Option<u32> {
        self.checked_ilog2()
    }

    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }

    #[inline(always)]
    fn wrapping_mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }

    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> (Self, Self) {
        self.carrying_mul(mul, add)
    }

    #[inline(always)]
    fn narrowing_div_rem((num_lo, num_hi): (Self, Self), denom: Self) -> (Self, Self) {
        let a = ((num_hi as u128) << 64) | (num_lo as u128);
        let b = denom as u128;
        let quo = a / b;
        let rem = a.wrapping_sub(quo.wrapping_mul(b));
        (quo as u64, rem as u64)
    }
}

impl Arithmetic for usize {
    const BITS: u32 = usize::BITS;
    const ZERO: Self = 0;
    const ONE: Self = 1;

    #[inline(always)]
    fn checked_ilog2(self) -> Option<u32> {
        self.checked_ilog2()
    }

    #[inline(always)]
    fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
    }

    #[inline(always)]
    fn wrapping_mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
    }

    #[inline(always)]
    fn mul_add(self, mul: Self, add: Self) -> (Self, Self) {
        self.carrying_mul(mul, add)
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

#[cfg(not(miri))]
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
    fn test_divider_u8_div_rem() {
        for d in 1..=u8::MAX {
            let divider = Divider::new(d).unwrap();
            for x in 0..=u8::MAX {
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
