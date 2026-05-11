//! Arbitrary-precision integers — Tier 1.D.
//!
//! Storage: a heap object tagged `Tag::Vector` with header type
//! `HeapType::Bignum`. Layout:
//!
//!   cell 0: HeapHeader { ty: Bignum, length_cells: 4 + n_limbs }
//!   cell 1: marker — the symbol `%BIGNUM` (printer + typep recogniser)
//!   cell 2: sign — fixnum +1 or -1
//!   cell 3: n_limbs — fixnum
//!   cell 4: reserved (cached fixnum-equivalent / hash)
//!   cell 5..5+n_limbs: raw u64 limbs, little-endian magnitude
//!
//! Arithmetic kernel: we delegate to `num_bigint::BigInt`. On every
//! operation we marshal limbs in (one allocation per operand), do
//! the arithmetic, marshal the result back into a fresh GC heap
//! object. No persistent Rust-side allocations means the GC owns
//! 100% of bignum memory and never needs finalisers.
//!
//! Normalisation: every result is run through `bigint_to_word`
//! which demotes to fixnum if the magnitude fits in 61-bit signed.

use crate::heap::{HeapHeader, HeapType};
use crate::mutator::MutatorState;
use crate::word::{Tag, Word, FIXNUM_MAX, FIXNUM_MIN};
use num_bigint::{BigInt, Sign};
use num_traits::{Signed, Zero};

/// Cells before the limb data: header + marker + sign + n_limbs + reserved.
pub const BIGNUM_HEADER_CELLS: usize = 5;

/// Allocate a bignum heap object holding the given little-endian
/// magnitude limbs and sign. NEVER call this directly with a value
/// that fits in fixnum range — go through `bigint_to_word`, which
/// demotes correctly.
fn alloc_bignum_raw(m: &mut MutatorState, sign: i8, limbs: &[u64]) -> Word {
    debug_assert!(sign == 1 || sign == -1);
    debug_assert!(!limbs.is_empty());
    debug_assert!(*limbs.last().unwrap() != 0);

    let total_cells = BIGNUM_HEADER_CELLS + limbs.len();
    let bignum_marker = m.coord().intern("%BIGNUM");

    // alloc_vector_with_header writes the heap header and returns a
    // Vector-tagged Word; we then fill in the rest of the cells.
    let w = m.alloc_typed_vector(HeapType::Bignum, total_cells as u32);
    let p = w.as_mut_ptr::<u64>(Tag::Vector).expect("just-allocated vector");

    unsafe {
        // p[0] = header (already written by alloc_typed_vector)
        *p.add(1) = bignum_marker.raw();
        *p.add(2) = Word::fixnum(sign as i64).raw();
        *p.add(3) = Word::fixnum(limbs.len() as i64).raw();
        *p.add(4) = Word::NIL.raw(); // reserved
        for (i, &limb) in limbs.iter().enumerate() {
            *p.add(BIGNUM_HEADER_CELLS + i) = limb;
        }
    }
    w
}

/// True iff WORD is a bignum heap object.
pub fn is_bignum(w: Word) -> bool {
    if w.tag() != Tag::Vector {
        return false;
    }
    let p = match w.as_ptr::<u64>(Tag::Vector) {
        Some(p) => p,
        None => return false,
    };
    let header = HeapHeader::from_raw(unsafe { *p });
    header.ty() == HeapType::Bignum
}

/// True iff WORD is any kind of integer — fixnum or bignum.
pub fn is_integer(w: Word) -> bool {
    w.tag() == Tag::Fixnum || is_bignum(w)
}

/// Read a bignum's contents into a `BigInt`. Caller must have
/// already verified `is_bignum(w)`.
pub fn bignum_to_bigint(w: Word) -> BigInt {
    let p = w.as_ptr::<u64>(Tag::Vector).expect("bignum is a vector");
    let sign_word = Word::from_raw(unsafe { *p.add(2) });
    let n_limbs_word = Word::from_raw(unsafe { *p.add(3) });
    let sign = sign_word.as_fixnum().expect("bignum sign is fixnum") as i8;
    let n_limbs = n_limbs_word.as_fixnum().expect("bignum n_limbs is fixnum") as usize;
    let limbs: Vec<u64> = (0..n_limbs)
        .map(|i| unsafe { *p.add(BIGNUM_HEADER_CELLS + i) })
        .collect();

    // num-bigint takes u32 chunks for from_slice; we can also use
    // BigUint::new which accepts u32 limbs. Convert our u64 limbs
    // to u32 little-endian first.
    let mut u32_limbs = Vec::with_capacity(n_limbs * 2);
    for &limb in &limbs {
        u32_limbs.push((limb & 0xFFFF_FFFF) as u32);
        u32_limbs.push((limb >> 32) as u32);
    }
    let mag = num_bigint::BigUint::new(u32_limbs);
    let s = if sign > 0 { Sign::Plus } else { Sign::Minus };
    if mag.is_zero() {
        BigInt::from(0)
    } else {
        BigInt::from_biguint(s, mag)
    }
}

/// Read any integer Word (fixnum or bignum) into a `BigInt`.
pub fn integer_to_bigint(w: Word) -> Option<BigInt> {
    if let Some(n) = w.as_fixnum() {
        return Some(BigInt::from(n));
    }
    if is_bignum(w) {
        return Some(bignum_to_bigint(w));
    }
    None
}

/// Convert a BigInt back to a Word, demoting to fixnum if the
/// magnitude fits. Allocates only if a real bignum is required.
pub fn bigint_to_word(m: &mut MutatorState, n: &BigInt) -> Word {
    // Fast path: try as i64. num-bigint's `to_i64` covers anything
    // within `i64::MIN..=i64::MAX`. We further restrict to fixnum
    // range (61-bit signed).
    if let Some(small) = num_traits::ToPrimitive::to_i64(n) {
        if small >= FIXNUM_MIN && small <= FIXNUM_MAX {
            return Word::fixnum(small);
        }
    }
    let sign: i8 = if n.is_negative() { -1 } else { 1 };
    let mag = n.magnitude();
    // Convert to u64 little-endian limbs.
    let u32_limbs: Vec<u32> = mag.to_u32_digits();
    let mut u64_limbs: Vec<u64> = Vec::with_capacity((u32_limbs.len() + 1) / 2);
    let mut i = 0;
    while i < u32_limbs.len() {
        let lo = u32_limbs[i] as u64;
        let hi = if i + 1 < u32_limbs.len() {
            u32_limbs[i + 1] as u64
        } else {
            0
        };
        u64_limbs.push((hi << 32) | lo);
        i += 2;
    }
    // Trim any trailing-zero limb introduced by the u32→u64 pack
    // (only happens when u32_limbs has odd length AND the high half
    // was zero — impossible since num-bigint's to_u32_digits doesn't
    // emit a leading zero. Defensive trim anyway.)
    while u64_limbs.len() > 1 && *u64_limbs.last().unwrap() == 0 {
        u64_limbs.pop();
    }
    if u64_limbs.is_empty() {
        return Word::fixnum(0);
    }
    alloc_bignum_raw(m, sign, &u64_limbs)
}

// ─── ABI helpers — promotion at fixnum overflow ────────────────────────────

/// `(+ a b)` slow path. Called from the JIT when the inline fixnum
/// add overflowed, OR when either operand is a bignum.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_add_promote(
    mutator: *mut MutatorState,
    a_raw: u64,
    b_raw: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let a = Word::from_raw(a_raw);
    let b = Word::from_raw(b_raw);
    let na = match integer_to_bigint(a) {
        Some(n) => n,
        None => panic!("+: non-integer operand: {a:?}"),
    };
    let nb = match integer_to_bigint(b) {
        Some(n) => n,
        None => panic!("+: non-integer operand: {b:?}"),
    };
    bigint_to_word(m, &(na + nb)).raw()
}

#[unsafe(no_mangle)]
pub extern "C" fn ncl_sub_promote(
    mutator: *mut MutatorState,
    a_raw: u64,
    b_raw: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let a = Word::from_raw(a_raw);
    let b = Word::from_raw(b_raw);
    let na = integer_to_bigint(a).unwrap_or_else(|| panic!("-: non-integer: {a:?}"));
    let nb = integer_to_bigint(b).unwrap_or_else(|| panic!("-: non-integer: {b:?}"));
    bigint_to_word(m, &(na - nb)).raw()
}

#[unsafe(no_mangle)]
pub extern "C" fn ncl_mul_promote(
    mutator: *mut MutatorState,
    a_raw: u64,
    b_raw: u64,
) -> u64 {
    let m = unsafe { &mut *mutator };
    let a = Word::from_raw(a_raw);
    let b = Word::from_raw(b_raw);
    let na = integer_to_bigint(a).unwrap_or_else(|| panic!("*: non-integer: {a:?}"));
    let nb = integer_to_bigint(b).unwrap_or_else(|| panic!("*: non-integer: {b:?}"));
    bigint_to_word(m, &(na * nb)).raw()
}

/// Cross-type integer comparison. Returns -1, 0, or +1 (as i64).
/// Called when the inline both-fixnum compare path can't handle
/// the operand types.
///
/// Permissive on non-integer arguments: pointer-bits comparison
/// (raw word value, signed) so e.g. `(eq sym1 sym2)`-style tests
/// via `=` still work the way they did under the pre-bignum
/// inline-compare. Strict integer-only comparison would break a
/// lot of pre-existing library code that uses `=` opportunistically
/// on tagged words.
#[unsafe(no_mangle)]
pub extern "C" fn ncl_cmp_int(
    a_raw: u64,
    b_raw: u64,
) -> i64 {
    let a = Word::from_raw(a_raw);
    let b = Word::from_raw(b_raw);
    let na = integer_to_bigint(a);
    let nb = integer_to_bigint(b);
    use std::cmp::Ordering::*;
    match (na, nb) {
        (Some(na), Some(nb)) => match na.cmp(&nb) {
            Less => -1,
            Equal => 0,
            Greater => 1,
        },
        _ => {
            // Fallback: signed compare of raw words. Matches the
            // pre-bignum inline behaviour. The only reason this
            // path fires today is that `(= sym1 sym2)` and similar
            // shapes show up in library code; treating them like
            // raw-word eq via cmp keeps the legacy idiom working.
            let ai = a_raw as i64;
            let bi = b_raw as i64;
            if ai < bi { -1 } else if ai > bi { 1 } else { 0 }
        }
    }
}

/// Render a bignum as a base-10 decimal string. Used by the
/// printer.
pub fn bignum_to_decimal(w: Word) -> String {
    bignum_to_bigint(w).to_str_radix(10)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutator::{GcConfig, GcCoordinator};

    fn small_config() -> GcConfig {
        GcConfig {
            young_bytes: 64 * 1024,
            old_bytes: 64 * 1024,
            static_bytes: 16 * 1024,
            tlab_cells: 256,
        }
    }

    #[test]
    fn fixnum_in_fixnum_out_for_small_bigints() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let n = BigInt::from(42i64);
        let w = bigint_to_word(&mut m, &n);
        assert_eq!(w.as_fixnum(), Some(42));
        assert!(!is_bignum(w));
    }

    #[test]
    fn round_trip_through_bignum() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let n: BigInt = "12345678901234567890123456789012345".parse().unwrap();
        let w = bigint_to_word(&mut m, &n);
        assert!(is_bignum(w));
        let back = bignum_to_bigint(w);
        assert_eq!(back, n);
    }

    #[test]
    fn negative_bignum_round_trip() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let n: BigInt = "-99999999999999999999999999999999999".parse().unwrap();
        let w = bigint_to_word(&mut m, &n);
        assert!(is_bignum(w));
        assert_eq!(bignum_to_bigint(w), n);
    }
}
