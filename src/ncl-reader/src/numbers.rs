//! Number parsing.
//!
//! `try_parse_number` is called on every unescaped atom before symbol
//! resolution. If it returns `Some`, the atom IS a number — that
//! decision is independent of readtable case (numbers are
//! case-insensitive by spec).
//!
//! Phase 1c supports fixnums (i64) and floats (f64). Bignums and
//! ratios land with the numeric tower. Demos that use `3/4` will
//! currently parse it as a symbol named `3/4`; we'll catch this in
//! the corpus run (1d) and decide whether to bring up ratios early
//! or list it as a known limitation.

use ncl_runtime::Value;

/// Try to read `text` as a number. Returns `None` if it's not a
/// number — caller should fall through to symbol resolution.
pub fn try_parse_number(text: &str) -> Option<Value> {
    if let Some(n) = parse_integer(text, 10) {
        return Some(Value::Fixnum(n));
    }
    if let Some(f) = parse_float(text) {
        return Some(Value::Float(f));
    }
    None
}

/// Parse an integer in the given radix. Allows optional leading `+`
/// or `-`. For radix 10 only, allows an optional trailing `.` (CL
/// quirk: `123.` is the integer 123, not a float).
pub fn parse_integer(text: &str, radix: u32) -> Option<i64> {
    let bytes = text.as_bytes();
    if bytes.is_empty() { return None; }
    let (sign, rest) = match bytes[0] {
        b'+' => (1i64, &text[1..]),
        b'-' => (-1i64, &text[1..]),
        _ => (1i64, text),
    };
    let body = if radix == 10 && rest.ends_with('.') {
        &rest[..rest.len() - 1]
    } else {
        rest
    };
    if body.is_empty() { return None; }
    let mut acc: i64 = 0;
    for c in body.chars() {
        let d = c.to_digit(radix)? as i64;
        acc = acc.checked_mul(radix as i64)?.checked_add(d)?;
    }
    Some(sign.checked_mul(acc)?)
}

/// Parse a float per CL syntax. Must contain either `.` or an
/// exponent marker (`eEdDsSfFlL`) — otherwise it's an integer (or
/// not a number). The exponent marker is normalised to `e` for
/// Rust's `parse::<f64>`.
pub fn parse_float(text: &str) -> Option<f64> {
    let mut iter = text.chars().peekable();
    let mut out = String::with_capacity(text.len());
    let mut has_dot_or_exp = false;

    if let Some(&c) = iter.peek() {
        if c == '+' || c == '-' { out.push(c); iter.next(); }
    }

    let mut digits_before = 0;
    while let Some(&c) = iter.peek() {
        if c.is_ascii_digit() { out.push(c); iter.next(); digits_before += 1; }
        else { break; }
    }

    let mut digits_after = 0;
    if iter.peek() == Some(&'.') {
        out.push('.');
        iter.next();
        has_dot_or_exp = true;
        while let Some(&c) = iter.peek() {
            if c.is_ascii_digit() { out.push(c); iter.next(); digits_after += 1; }
            else { break; }
        }
    }

    if digits_before == 0 && digits_after == 0 {
        return None;
    }

    if let Some(&c) = iter.peek() {
        if matches!(c, 'e' | 'E' | 'd' | 'D' | 's' | 'S' | 'f' | 'F' | 'l' | 'L') {
            // Must have at least one digit before exponent marker.
            if digits_before == 0 && digits_after == 0 { return None; }
            out.push('e');
            iter.next();
            has_dot_or_exp = true;
            if let Some(&sc) = iter.peek() {
                if sc == '+' || sc == '-' { out.push(sc); iter.next(); }
            }
            let mut exp_digits = 0;
            while let Some(&c) = iter.peek() {
                if c.is_ascii_digit() { out.push(c); iter.next(); exp_digits += 1; }
                else { break; }
            }
            if exp_digits == 0 { return None; }
        }
    }

    if !has_dot_or_exp { return None; }
    if iter.next().is_some() { return None; }

    out.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fxn(v: &Value) -> i64 { match v { Value::Fixnum(n) => *n, _ => panic!("not fixnum: {v:?}") } }
    fn flt(v: &Value) -> f64 { match v { Value::Float(f) => *f, _ => panic!("not float: {v:?}") } }

    #[test]
    fn integers() {
        assert_eq!(fxn(&try_parse_number("42").unwrap()), 42);
        assert_eq!(fxn(&try_parse_number("-7").unwrap()), -7);
        assert_eq!(fxn(&try_parse_number("+123").unwrap()), 123);
        assert_eq!(fxn(&try_parse_number("0").unwrap()), 0);
        // CL trailing dot on integer
        assert_eq!(fxn(&try_parse_number("42.").unwrap()), 42);
    }

    #[test]
    fn floats() {
        assert_eq!(flt(&try_parse_number("1.0").unwrap()), 1.0);
        assert_eq!(flt(&try_parse_number(".5").unwrap()), 0.5);
        assert_eq!(flt(&try_parse_number("1e10").unwrap()), 1e10);
        assert_eq!(flt(&try_parse_number("1.5e-3").unwrap()), 1.5e-3);
        assert_eq!(flt(&try_parse_number("-3.14").unwrap()), -3.14);
        // CL exponent markers — d/s/f/l all parse as float; we
        // ignore precision distinctions for now.
        assert_eq!(flt(&try_parse_number("1.0d0").unwrap()), 1.0);
        assert_eq!(flt(&try_parse_number("1.5s2").unwrap()), 150.0);
    }

    #[test]
    fn not_numbers() {
        assert!(try_parse_number("foo").is_none());
        assert!(try_parse_number("").is_none());
        assert!(try_parse_number("3foo").is_none());
        assert!(try_parse_number("1e").is_none());
        assert!(try_parse_number(".").is_none());
        // Ratios — Phase 1c does NOT recognise these. They fall
        // through and become symbols. See module docstring.
        assert!(try_parse_number("3/4").is_none());
    }

    #[test]
    fn radix() {
        assert_eq!(parse_integer("FF", 16), Some(255));
        assert_eq!(parse_integer("ff", 16), Some(255));
        assert_eq!(parse_integer("777", 8), Some(0o777));
        assert_eq!(parse_integer("1010", 2), Some(10));
        assert_eq!(parse_integer("ZZ", 36), Some(35 * 36 + 35));
        assert_eq!(parse_integer("-FF", 16), Some(-255));
        assert_eq!(parse_integer("FG", 16), None);
    }
}
