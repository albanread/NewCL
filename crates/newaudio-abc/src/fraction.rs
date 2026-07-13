//! Rational duration used for ABC note lengths. Mirrors `Fraction` in the
//! original Zig source: durations are stored in whole-note units and are
//! never reduced — preserving the unreduced form is what lets the MIDI
//! writer produce byte-identical output to the Zig reference.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Fraction {
    pub num: i32,
    pub denom: i32,
}

impl Fraction {
    pub const fn new(num: i32, denom: i32) -> Self {
        Self { num, denom }
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.denom as f64
    }

    pub fn mul(self, other: Self) -> Self {
        Self {
            num: self.num.wrapping_mul(other.num),
            denom: self.denom.wrapping_mul(other.denom),
        }
    }

    pub fn add(self, other: Self) -> Self {
        Self {
            num: self.num.wrapping_mul(other.denom).wrapping_add(other.num.wrapping_mul(self.denom)),
            denom: self.denom.wrapping_mul(other.denom),
        }
    }

    pub fn mul_int(self, m: i32) -> Self {
        Self {
            num: self.num.wrapping_mul(m),
            denom: self.denom,
        }
    }

    pub fn div_int(self, d: i32) -> Self {
        Self {
            num: self.num,
            denom: self.denom.wrapping_mul(d),
        }
    }
}

impl Default for Fraction {
    fn default() -> Self {
        Self::new(1, 8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quarter_note_value() {
        let q = Fraction::new(1, 4);
        assert_eq!(q.to_f64(), 0.25);
    }

    #[test]
    fn mul_preserves_unreduced_form() {
        // 1/2 * 1/2 = 1/4 stored as 1/4 (numerator * numerator).
        let half = Fraction::new(1, 2);
        let q = half.mul(half);
        assert_eq!((q.num, q.denom), (1, 4));
    }

    #[test]
    fn add_uses_cross_multiplication() {
        // 1/4 + 1/8 = 12/32 (unreduced cross product), not 3/8.
        let a = Fraction::new(1, 4);
        let b = Fraction::new(1, 8);
        let s = a.add(b);
        assert_eq!((s.num, s.denom), (12, 32));
        assert!((s.to_f64() - 0.375).abs() < 1e-9);
    }

    #[test]
    fn mul_int_only_scales_numerator() {
        let f = Fraction::new(1, 8).mul_int(3);
        assert_eq!((f.num, f.denom), (3, 8));
    }

    #[test]
    fn div_int_only_scales_denominator() {
        let f = Fraction::new(1, 8).div_int(2);
        assert_eq!((f.num, f.denom), (1, 16));
    }
}
