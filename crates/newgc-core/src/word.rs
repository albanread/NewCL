//! Tagged 64-bit Lisp values — what compiled code carries in registers
//! and on the stack.
//!
//! See `docs/GC.md` for the full design. Briefly: the low 3 bits of
//! every `Word` classify it; the upper 61 bits hold either the value
//! (fixnums, immediates) or an 8-byte-aligned heap pointer with the
//! tag masked off.
//!
//! `nil` is bit pattern `0` exactly — a fixnum-tagged zero — so
//! `(eq x nil)` is one compare.
//!
//! `Forward` (tag 111) is the GC-internal "this slot has been moved"
//! marker. Detecting a stale slot is one mask-and-compare during a
//! collection pass.
//!
//! Phase 2.5 step 1: this file is the foundation. No allocator yet.

use std::fmt;

/// A tagged Lisp value.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Word(u64);

/// Number of low bits used for the tag.
pub const TAG_BITS: u32 = 3;
/// Mask for the tag bits.
pub const TAG_MASK: u64 = 0b111;
/// Mask for the payload (everything but the tag bits).
pub const PAYLOAD_MASK: u64 = !TAG_MASK;

/// Inclusive bounds for a fixnum payload (61-bit signed).
pub const FIXNUM_MIN: i64 = -(1_i64 << 60);
pub const FIXNUM_MAX: i64 = (1_i64 << 60) - 1;

/// Tag categories. Values match the bit patterns; never reorder.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tag {
    Fixnum = 0b000,
    Cons = 0b001,
    Symbol = 0b010,
    Vector = 0b011,
    Function = 0b100,
    String = 0b101,
    Immediate = 0b110,
    Forward = 0b111,
}

impl Tag {
    pub fn from_bits(bits: u64) -> Tag {
        match bits & TAG_MASK {
            0b000 => Tag::Fixnum,
            0b001 => Tag::Cons,
            0b010 => Tag::Symbol,
            0b011 => Tag::Vector,
            0b100 => Tag::Function,
            0b101 => Tag::String,
            0b110 => Tag::Immediate,
            0b111 => Tag::Forward,
            _ => unreachable!(),
        }
    }
}

// -- Immediate sub-tags -------------------------------------------------------
//
// Tag `Immediate` (110) carries a 5-bit sub-tag in bits 3..8 plus a
// payload in the upper 56 bits. Sub-tag 0 has empty payload — that's
// `T`, the canonical truth value. Sub-tag 1 carries a Unicode scalar.
// More sub-tags as we need them (unbound marker, end-of-file, etc.).

const SUBTAG_BITS: u32 = 5;
#[allow(dead_code)]
const SUBTAG_MASK: u64 = 0b11111 << TAG_BITS;

const SUBTAG_T: u64 = 0;
const SUBTAG_CHAR: u64 = 1;
const SUBTAG_UNBOUND: u64 = 2;
const SUBTAG_NIL: u64 = 3;

const fn immediate(subtag: u64, payload: u64) -> u64 {
    (payload << (TAG_BITS + SUBTAG_BITS)) | (subtag << TAG_BITS) | (Tag::Immediate as u64)
}

impl Word {
    /// `nil` — the empty list, the only false value, also the symbol
    /// `COMMON-LISP:NIL`. Represented as immediate sub-tag 3 so it
    /// is distinguishable from `fixnum 0` (CL says `(eq nil 0)` is
    /// false). `(eq x nil)` is still one `cmp x, NIL_RAW`.
    pub const NIL: Word = Word(immediate(SUBTAG_NIL, 0));

    /// `T` — the canonical truth value.
    pub const T: Word = Word(immediate(SUBTAG_T, 0));

    /// Marker for an unbound cell (symbol value or function cell).
    /// Distinct from `nil` so `(boundp 'foo)` can distinguish.
    pub const UNBOUND: Word = Word(immediate(SUBTAG_UNBOUND, 0));

    /// Construct a fixnum. Panics in debug if `n` is out of range;
    /// callers that need to handle overflow should use `try_fixnum`.
    pub fn fixnum(n: i64) -> Word {
        debug_assert!(
            (FIXNUM_MIN..=FIXNUM_MAX).contains(&n),
            "fixnum out of range: {n}"
        );
        Word((n as u64) << TAG_BITS)
    }

    /// Construct a fixnum if it fits in 61 bits, else `None`.
    pub fn try_fixnum(n: i64) -> Option<Word> {
        if (FIXNUM_MIN..=FIXNUM_MAX).contains(&n) {
            Some(Word((n as u64) << TAG_BITS))
        } else {
            None
        }
    }

    /// Construct a character literal.
    pub fn char(c: char) -> Word {
        Word(immediate(SUBTAG_CHAR, c as u64))
    }

    /// Construct a tagged pointer. Pointer must be 8-byte aligned —
    /// debug-asserted but not enforced in release. Used by allocators
    /// only; user code goes through typed constructors.
    pub fn from_ptr<T>(ptr: *const T, tag: Tag) -> Word {
        let p = ptr as u64;
        debug_assert!(p & TAG_MASK == 0, "pointer not 8-byte aligned: {p:#x}");
        debug_assert!(
            !matches!(tag, Tag::Fixnum | Tag::Immediate),
            "from_ptr called with non-pointer tag {tag:?}"
        );
        Word(p | (tag as u64))
    }

    /// Construct a forwarding pointer (GC-internal).
    pub fn forward(new_addr: *const ()) -> Word {
        let p = new_addr as u64;
        debug_assert!(p & TAG_MASK == 0, "forward target not aligned: {p:#x}");
        Word(p | (Tag::Forward as u64))
    }

    /// Raw 64-bit representation. Use for debugging; the tag method is
    /// the supported way to ask "what kind of value is this?"
    pub const fn raw(self) -> u64 { self.0 }

    /// Construct from a raw 64-bit pattern. Used by GC internals and
    /// when reading values out of the heap.
    pub const fn from_raw(bits: u64) -> Word { Word(bits) }

    pub fn tag(self) -> Tag { Tag::from_bits(self.0) }

    pub fn is_nil(self) -> bool { self.0 == Word::NIL.0 }
    pub fn is_t(self) -> bool { self.0 == Word::T.0 }
    pub fn is_unbound(self) -> bool { self.0 == Word::UNBOUND.0 }
    pub fn is_fixnum(self) -> bool { self.tag() == Tag::Fixnum }
    pub fn is_cons(self) -> bool { self.tag() == Tag::Cons }
    pub fn is_symbol(self) -> bool { self.tag() == Tag::Symbol }
    pub fn is_forward(self) -> bool { self.tag() == Tag::Forward }
    pub fn is_immediate(self) -> bool { self.tag() == Tag::Immediate }

    /// Recover the integer value of a fixnum. Returns `None` for
    /// non-fixnum tags.
    pub fn as_fixnum(self) -> Option<i64> {
        if self.is_fixnum() {
            // Arithmetic shift right preserves sign.
            Some((self.0 as i64) >> TAG_BITS)
        } else {
            None
        }
    }

    /// Recover the character value of a character immediate.
    pub fn as_char(self) -> Option<char> {
        if (self.0 & ((1 << (TAG_BITS + SUBTAG_BITS)) - 1))
            == (Tag::Immediate as u64) | (SUBTAG_CHAR << TAG_BITS)
        {
            char::from_u32((self.0 >> (TAG_BITS + SUBTAG_BITS)) as u32)
        } else {
            None
        }
    }

    /// Untag and return the raw heap pointer if the tag matches.
    pub fn as_ptr<T>(self, expected: Tag) -> Option<*const T> {
        if self.tag() == expected {
            Some((self.0 & PAYLOAD_MASK) as *const T)
        } else {
            None
        }
    }

    /// Same as `as_ptr` but mutable.
    pub fn as_mut_ptr<T>(self, expected: Tag) -> Option<*mut T> {
        if self.tag() == expected {
            Some((self.0 & PAYLOAD_MASK) as *mut T)
        } else {
            None
        }
    }

    /// Untagged forwarding-pointer target. For GC use only.
    pub fn forward_target(self) -> Option<*const ()> {
        if self.is_forward() {
            Some((self.0 & PAYLOAD_MASK) as *const ())
        } else {
            None
        }
    }
}

// Custom Debug shows the structural meaning rather than the bits.
impl fmt::Debug for Word {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_nil() {
            return write!(f, "nil");
        }
        if self.is_t() {
            return write!(f, "T");
        }
        if self.is_unbound() {
            return write!(f, "<unbound>");
        }
        match self.tag() {
            Tag::Fixnum => write!(f, "{}", self.as_fixnum().unwrap()),
            Tag::Immediate => match self.as_char() {
                Some(c) => write!(f, "#\\{}", c),
                None => write!(f, "<imm:{:#x}>", self.0),
            },
            Tag::Forward => write!(f, "<forward:{:p}>", self.forward_target().unwrap()),
            tag @ (Tag::Cons | Tag::Symbol | Tag::Vector | Tag::Function | Tag::String) => {
                write!(f, "<{tag:?}:{:#x}>", self.0 & PAYLOAD_MASK)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nil_distinct_from_fixnum_zero() {
        assert!(Word::NIL.is_nil());
        assert!(!Word::NIL.is_fixnum());
        assert_eq!(Word::NIL.as_fixnum(), None);
        assert_eq!(Word::NIL.tag(), Tag::Immediate);

        let zero = Word::fixnum(0);
        assert!(zero.is_fixnum());
        assert!(!zero.is_nil());
        assert_eq!(zero.as_fixnum(), Some(0));
        assert_ne!(zero.raw(), Word::NIL.raw());
    }

    #[test]
    fn fixnum_round_trip() {
        for n in [0_i64, 1, -1, 42, -42, FIXNUM_MIN, FIXNUM_MAX, 1 << 30, -(1 << 30)] {
            let w = Word::fixnum(n);
            assert!(w.is_fixnum(), "{n}");
            assert_eq!(w.as_fixnum(), Some(n), "{n}");
        }
    }

    #[test]
    fn fixnum_range_bounds() {
        assert!(Word::try_fixnum(FIXNUM_MAX).is_some());
        assert!(Word::try_fixnum(FIXNUM_MIN).is_some());
        assert!(Word::try_fixnum(FIXNUM_MAX + 1).is_none());
        assert!(Word::try_fixnum(FIXNUM_MIN - 1).is_none());
        assert!(Word::try_fixnum(i64::MAX).is_none());
        assert!(Word::try_fixnum(i64::MIN).is_none());
    }

    #[test]
    fn fixnum_arithmetic_friendly() {
        // a+b on tagged words equals (a+b) on the values, because
        // both have low bits 0 — the additions don't carry into
        // the tag region. This is the SBCL/Allegro fixnum trick.
        let a = Word::fixnum(123);
        let b = Word::fixnum(456);
        let sum_raw = Word::from_raw(a.raw().wrapping_add(b.raw()));
        assert_eq!(sum_raw.as_fixnum(), Some(123 + 456));
    }

    #[test]
    fn t_is_distinct_from_nil_and_zero() {
        assert!(Word::T.is_t());
        assert!(!Word::T.is_nil());
        assert!(!Word::T.is_fixnum());
        assert_eq!(Word::T.tag(), Tag::Immediate);
    }

    #[test]
    fn unbound_is_distinct_from_nil() {
        assert!(Word::UNBOUND.is_unbound());
        assert!(!Word::UNBOUND.is_nil());
        assert!(!Word::UNBOUND.is_t());
    }

    #[test]
    fn char_round_trip() {
        for c in ['a', 'Z', ' ', '\n', '∀', '🦀'] {
            let w = Word::char(c);
            assert_eq!(w.tag(), Tag::Immediate);
            assert_eq!(w.as_char(), Some(c), "{c:?}");
            assert!(!w.is_t());
        }
    }

    #[test]
    fn char_not_confused_with_t() {
        assert_eq!(Word::T.as_char(), None);
        assert_eq!(Word::UNBOUND.as_char(), None);
    }

    #[test]
    fn from_ptr_round_trip() {
        // Use a stack-aligned dummy. The value at the address is
        // irrelevant — we're just checking encoding.
        let dummy: u64 = 0;
        let p = (&dummy as *const u64) as *const u8;
        let w = Word::from_ptr(p, Tag::Cons);
        assert_eq!(w.tag(), Tag::Cons);
        assert_eq!(w.as_ptr::<u8>(Tag::Cons), Some(p));
        assert_eq!(w.as_ptr::<u8>(Tag::Symbol), None);
    }

    #[test]
    fn forward_round_trip() {
        // Sweep across multiple 8-byte alignments inside one 64-byte
        // window. The NewOpenDylan GC port (NCL_GC_FEEDBACK.md,
        // 2026-05-17) caught a forwarding-encoding bug that had
        // slipped past a single-pointer round-trip test — the address
        // they happened to pick was 256-byte aligned and the encoding
        // was lossy below that. A sweep across `0x..00`, `0x..08`,
        // `0x..10`, … `0x..38` would have caught it on the first
        // run. We have no equivalent lossy case (Tag::Forward leaves
        // 61 bits for the payload — no compression), but sweep
        // anyway so the test pattern doesn't lull a future encoding
        // change into the same trap.
        let base: usize = 0x0000_7FF8_DEAD_BE00;
        for off in (0..64).step_by(8) {
            let p = (base + off) as *const ();
            let w = Word::forward(p);
            assert!(w.is_forward(), "lost forward tag at offset {off:#x}");
            assert_eq!(
                w.forward_target(),
                Some(p),
                "lossy at base+{off:#x} (raw {:#018x})",
                w.raw(),
            );
            assert_eq!(w.as_fixnum(), None);
        }
        // Sanity: stack-resident pointer still works (the historical
        // test shape).
        let dummy: u64 = 0;
        let p = (&dummy as *const u64) as *const ();
        let w = Word::forward(p);
        assert!(w.is_forward());
        assert_eq!(w.forward_target(), Some(p));
    }

    #[test]
    fn tag_dispatch_is_total() {
        // Every 3-bit pattern maps to a Tag.
        for bits in 0u64..8 {
            let _ = Tag::from_bits(bits);
        }
    }

    #[test]
    fn debug_format_is_helpful() {
        assert_eq!(format!("{:?}", Word::NIL), "nil");
        assert_eq!(format!("{:?}", Word::T), "T");
        assert_eq!(format!("{:?}", Word::UNBOUND), "<unbound>");
        assert_eq!(format!("{:?}", Word::fixnum(42)), "42");
        assert_eq!(format!("{:?}", Word::fixnum(-7)), "-7");
        assert_eq!(format!("{:?}", Word::char('a')), r"#\a");
    }

    #[test]
    fn word_is_eight_bytes() {
        // Vital invariant: a Word fits in a register and on a stack
        // slot at exactly 8 bytes.
        assert_eq!(std::mem::size_of::<Word>(), 8);
        assert_eq!(std::mem::align_of::<Word>(), 8);
    }

    #[test]
    fn payload_mask_is_correct() {
        assert_eq!(TAG_MASK, 0b111);
        assert_eq!(PAYLOAD_MASK, !0b111);
        assert_eq!(TAG_MASK | PAYLOAD_MASK, !0u64);
    }
}
