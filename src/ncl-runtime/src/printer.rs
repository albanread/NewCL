//! Pretty-printer for `Word` values.
//!
//! Used by the driver's `--eval` output and by tests that want a
//! human-readable rendering of a value. Walks heap pointers, so
//! must be called from the Lisp thread (or after stop-the-world).
//! Cycles in cons-graphs would loop forever in this v1 printer —
//! `*print-circle*` and friends land later.

use crate::word::{Tag, Word};

/// Format `w` as a CL-style printed value.
pub fn format_word(w: Word) -> String {
    let mut out = String::new();
    write_word(&mut out, w);
    out
}

fn write_word(out: &mut String, w: Word) {
    if w.is_nil() {
        out.push_str("nil");
        return;
    }
    if w.is_t() {
        out.push('T');
        return;
    }
    if w.is_unbound() {
        out.push_str("<unbound>");
        return;
    }
    match w.tag() {
        Tag::Fixnum => out.push_str(&w.as_fixnum().unwrap().to_string()),
        Tag::Cons => write_list(out, w),
        Tag::Immediate => {
            if let Some(c) = w.as_char() {
                out.push_str("#\\");
                out.push(c);
            } else {
                out.push_str(&format!("<imm {:#x}>", w.raw()));
            }
        }
        Tag::Forward => out.push_str("<forward>"),
        Tag::Symbol => match crate::sym_names::lookup(w.raw()) {
            Some(name) => out.push_str(&name),
            None => out.push_str("<symbol>"),
        },
        Tag::Vector => out.push_str("<vector>"),
        Tag::Function => out.push_str("<function>"),
        Tag::String => out.push_str("<string>"),
    }
}

/// Print a cons cell as a CL list: `(a b c)` if proper, `(a b . c)`
/// if dotted at the end.
fn write_list(out: &mut String, head: Word) {
    out.push('(');
    let mut cur = head;
    loop {
        let p = cur.as_ptr::<u64>(Tag::Cons).expect("cons");
        let car = Word::from_raw(unsafe { *p });
        let cdr = Word::from_raw(unsafe { *p.add(1) });
        write_word(out, car);
        if cdr.is_nil() {
            out.push(')');
            return;
        }
        if !cdr.is_cons() {
            out.push_str(" . ");
            write_word(out, cdr);
            out.push(')');
            return;
        }
        out.push(' ');
        cur = cdr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mutator::{GcConfig, GcCoordinator};

    fn small_config() -> GcConfig {
        GcConfig {
            young_bytes: 16 * 1024,
            old_bytes: 16 * 1024,
            static_bytes: 8 * 1024,
            tlab_cells: 64,
        }
    }

    #[test]
    fn primitives_print() {
        assert_eq!(format_word(Word::NIL), "nil");
        assert_eq!(format_word(Word::T), "T");
        assert_eq!(format_word(Word::UNBOUND), "<unbound>");
        assert_eq!(format_word(Word::fixnum(0)), "0");
        assert_eq!(format_word(Word::fixnum(42)), "42");
        assert_eq!(format_word(Word::fixnum(-7)), "-7");
        assert_eq!(format_word(Word::char('a')), "#\\a");
    }

    #[test]
    fn dotted_cons_prints_with_dot() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let c = m.alloc_cons(Word::fixnum(1), Word::fixnum(2));
        assert_eq!(format_word(c), "(1 . 2)");
    }

    #[test]
    fn proper_list_prints_without_dots() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let c3 = m.alloc_cons(Word::fixnum(3), Word::NIL);
        let c2 = m.alloc_cons(Word::fixnum(2), c3);
        let c1 = m.alloc_cons(Word::fixnum(1), c2);
        assert_eq!(format_word(c1), "(1 2 3)");
    }

    #[test]
    fn list_with_dotted_tail() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        // (1 2 . 3)
        let c2 = m.alloc_cons(Word::fixnum(2), Word::fixnum(3));
        let c1 = m.alloc_cons(Word::fixnum(1), c2);
        assert_eq!(format_word(c1), "(1 2 . 3)");
    }

    #[test]
    fn nested_lists_print_recursively() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        // ((1 2) 3)
        let inner_tail = m.alloc_cons(Word::fixnum(2), Word::NIL);
        let inner = m.alloc_cons(Word::fixnum(1), inner_tail);
        let outer_tail = m.alloc_cons(Word::fixnum(3), Word::NIL);
        let outer = m.alloc_cons(inner, outer_tail);
        assert_eq!(format_word(outer), "((1 2) 3)");
    }

    #[test]
    fn singleton_list() {
        let coord = GcCoordinator::new(small_config());
        let mut m = coord.register_mutator();
        let c = m.alloc_cons(Word::fixnum(42), Word::NIL);
        assert_eq!(format_word(c), "(42)");
    }
}
