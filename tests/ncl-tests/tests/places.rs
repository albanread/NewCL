//! Coverage for `Lisp/Library/places.lisp` — the standard
//! `(setf accessor)` declarations and the short-form DEFSETF
//! macro.
//!
//! There are two pieces that have to line up:
//!
//!   1. `(defun (setf NAME) (val args…) …)` is accepted by the
//!      compiler. The function-name slot can now be either a
//!      symbol or a `(setf SYMBOL)` cons; the latter mangles to
//!      `%SETF-SYMBOL` for the generic-setf-fallback to find.
//!
//!   2. Library/places.lisp registers the standard accessors
//!      (FIRST, REST, NTH, the CXR family, LAST) so user code
//!      can write `(setf (first xs) 99)` instead of
//!      `(setf (car xs) 99)`.
//!
//!   3. DEFSETF lets a user register the inverse of any access
//!      function: `(defsetf place setter)` arranges for
//!      `(setf (place args…) val)` to call `(setter args… val)`.

use std::path::PathBuf;

use ncl_compiler::Session;
use ncl_tests::TestSession;

fn library_path(name: &str) -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p.push("Lisp");
    p.push("Library");
    p.push(name);
    p
}

fn fresh_session_with_places() -> TestSession {
    let mut s = Session::with_stdlib().expect("session boots");
    s.activate();
    let path = library_path("places.lisp");
    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    s.eval(&src).expect("Library/places.lisp loads");
    TestSession::with_thread_name(s)
}

// ── (defun (setf NAME) …) — the compiler-side enabler ─────────────────

#[test]
fn defun_setf_name_compiles_and_dispatches() {
    // No Library load — just verify the compiler accepts the
    // `(setf NAME)` shape on its own. User defines an accessor +
    // its setter; setf-fallback finds the setter.
    let mut s = TestSession::with_thread_name(
        Session::with_stdlib().expect("session boots"),
    );
    s.activate();
    let prog = "
        (defun my-foo (obj) (car obj))
        (defun (setf my-foo) (val obj) (setf (car obj) val))
        (defparameter *xs* (list 1 2 3))
        (setf (my-foo *xs*) 99)
        *xs*
    ";
    assert_eq!(s.eval(prog).unwrap(), "(99 2 3)");
}

#[test]
fn defun_setf_supports_multiple_args() {
    // Verify the compiler accepts `(setf NAME)` defuns with more
    // than one place-arg. Stays inside core (no places.lisp
    // dependency) by using only setf-(car) under the hood.
    let mut s = TestSession::with_thread_name(
        Session::with_stdlib().expect("session boots"),
    );
    s.activate();
    let prog = "
        ;; Two-arg accessor: head of a numbered ALIST entry.
        (defun nth-pair (n alist) (nth n alist))
        (defun (setf nth-pair) (val n alist)
          (setf (car (nthcdr n alist)) val))
        (defparameter *a* (list 'a 'b 'c 'd))
        (setf (nth-pair 1 *a*) :HIT)
        *a*
    ";
    assert_eq!(s.eval(prog).unwrap(), "(A :HIT C D)");
}

// ── Named cons accessors ──────────────────────────────────────────────

#[test]
fn setf_first_through_fourth_each_set_the_named_position() {
    let mut s = fresh_session_with_places();
    let prog = "
        (let ((xs (list 1 2 3 4 5)))
          (setf (first  xs) 'a)
          (setf (second xs) 'b)
          (setf (third  xs) 'c)
          (setf (fourth xs) 'd)
          xs)
    ";
    assert_eq!(s.eval(prog).unwrap(), "(A B C D 5)");
}

#[test]
fn setf_rest_replaces_the_tail() {
    let mut s = fresh_session_with_places();
    assert_eq!(
        s.eval(
            "(let ((xs (list 1 2 3)))
               (setf (rest xs) '(20 30 40))
               xs)"
        )
        .unwrap(),
        "(1 20 30 40)",
    );
}

#[test]
fn setf_cxr_family_walks_through_nested_lists() {
    let mut s = fresh_session_with_places();
    // (setf (caddr xs) v) replaces the third element. Same as
    // (setf (third xs) v) but through the cdr-chain notation.
    let prog = "
        (let ((xs (list 1 2 3 4 5)))
          (setf (caddr xs) :third)
          xs)
    ";
    assert_eq!(s.eval(prog).unwrap(), "(1 2 :THIRD 4 5)");
    // (setf (cadar xs) v) — second element of the first sublist.
    let prog = "
        (let ((xs (list (list 'a 'b 'c) (list 'd 'e 'f))))
          (setf (cadar xs) :Z)
          xs)
    ";
    assert_eq!(s.eval(prog).unwrap(), "((A :Z C) (D E F))");
}

// ── (setf nth N list) ────────────────────────────────────────────────

#[test]
fn setf_nth_replaces_indexed_element() {
    let mut s = fresh_session_with_places();
    assert_eq!(
        s.eval(
            "(let ((xs (list 'a 'b 'c 'd 'e)))
               (setf (nth 0 xs) :zero)
               (setf (nth 2 xs) :two)
               (setf (nth 4 xs) :four)
               xs)"
        )
        .unwrap(),
        "(:ZERO B :TWO D :FOUR)",
    );
}

// ── DEFSETF (short form) ─────────────────────────────────────────────

#[test]
fn defsetf_short_form_installs_inverse() {
    let mut s = fresh_session_with_places();
    let prog = "
        ;; A toy place: middle of a 3-list.
        (defun mid (xs) (second xs))
        (defun set-mid (xs val) (setf (second xs) val))
        (defsetf mid set-mid)

        (defparameter *xs* (list 'a 'b 'c))
        (setf (mid *xs*) :HIT)
        *xs*
    ";
    assert_eq!(s.eval(prog).unwrap(), "(A :HIT C)");
}

#[test]
fn defsetf_passes_multiple_place_args() {
    let mut s = fresh_session_with_places();
    let prog = "
        ;; Two-arg place. (cell row grid) reads the row-th element;
        ;; setter takes both args + val and stores.
        (defun cell (row grid) (nth row grid))
        (defun set-cell (row grid val) (setf (nth row grid) val))
        (defsetf cell set-cell)

        (defparameter *g* (list 'a 'b 'c 'd))
        (setf (cell 2 *g*) :TWO)
        *g*
    ";
    assert_eq!(s.eval(prog).unwrap(), "(A B :TWO D)");
}
