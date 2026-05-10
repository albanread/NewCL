//! `format` — CL's printf, more or less. Walks a control string and
//! emits the result either to stdout (dest = `t`) or as a freshly
//! allocated string (dest = `nil`).
//!
//! Directives implemented in this slice:
//!   `~A`  aesthetic: `princ`-style — strings without quotes, chars
//!         as the bare codepoint.
//!   `~S`  readable: `prin1`-style — strings quoted, chars as `#\X`.
//!   `~D`  decimal integer (errors on non-fixnum).
//!   `~%`  newline.
//!   `~&`  newline (simplified — full `fresh-line` waits on stream
//!         column tracking).
//!   `~~`  literal `~`.
//!
//! Everything else is deferred. The walker doesn't try to be
//! forgiving with unknown directives — it panics, which gives the
//! condition system something concrete to translate into a Lisp
//! error when it lands.

use crate::word::{Tag, Word};

/// `(format dest control args-list)` — the runtime core. The
/// `args-list` is a proper list; the `format-shim` ABI function
/// builds it from variadic args before calling here.
pub fn run_format(
    m: &mut crate::mutator::MutatorState,
    dest: Word,
    ctrl: Word,
    args_list: Word,
) -> Word {
    if ctrl.tag() != Tag::String {
        panic!("format: control argument must be a string, got {ctrl:?}");
    }

    let mut out = String::new();
    let mut args = args_list;
    let n = crate::gc_string::char_count(ctrl);
    let mut i = 0;
    while i < n {
        let cp = crate::gc_string::codepoint_at(ctrl, i);
        let c = char::from_u32(cp).expect("invalid codepoint in control string");
        if c != '~' {
            out.push(c);
            i += 1;
            continue;
        }
        i += 1;
        if i >= n {
            panic!("format: trailing '~' with no directive");
        }
        let dcp = crate::gc_string::codepoint_at(ctrl, i);
        let dc = char::from_u32(dcp)
            .expect("invalid codepoint in control string")
            .to_ascii_uppercase();
        i += 1;
        match dc {
            'A' => {
                let arg = pop_arg(&mut args);
                out.push_str(&crate::printer::format_word_aesthetic(arg));
            }
            'S' => {
                let arg = pop_arg(&mut args);
                out.push_str(&crate::printer::format_word(arg));
            }
            'D' => {
                let arg = pop_arg(&mut args);
                match arg.as_fixnum() {
                    Some(n) => out.push_str(&n.to_string()),
                    None => panic!("format: ~D argument is not a fixnum: {arg:?}"),
                }
            }
            '%' | '&' => out.push('\n'),
            '~' => out.push('~'),
            other => panic!("format: unknown directive ~{other}"),
        }
    }

    if dest.is_t() {
        // Send to stdout. The trailing flush keeps `~%` ordering
        // sane in interactive contexts.
        use std::io::Write;
        let stdout = std::io::stdout();
        let mut h = stdout.lock();
        let _ = h.write_all(out.as_bytes());
        let _ = h.flush();
        Word::NIL
    } else if dest.is_nil() {
        crate::gc_string::alloc_string_in_young(m, &out)
    } else {
        // Streams aren't a thing yet. We could be permissive but
        // surprising silently is worse than failing loudly here.
        panic!("format: dest must be t (stdout) or nil (return string), got {dest:?}");
    }
}

fn pop_arg(args: &mut Word) -> Word {
    if args.is_nil() {
        panic!("format: not enough args for control string");
    }
    if args.tag() != Tag::Cons {
        panic!("format: args must be a proper list, got {args:?}");
    }
    let p = args.as_ptr::<u64>(Tag::Cons).expect("cons");
    let car = Word::from_raw(unsafe { *p });
    let cdr = Word::from_raw(unsafe { *p.add(1) });
    *args = cdr;
    car
}
