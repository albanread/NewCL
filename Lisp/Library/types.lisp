;;;; Lisp/Library/types.lisp — basic CL type system helpers.
;;;;
;;;; SUBTYPEP — the missing complement to TYPEP. CL's full
;;;; subtype calculus is undecidable in general (SATISFIES alone
;;;; makes it so), but the practical 80% boils down to a finite
;;;; hierarchy of named primitive types — INTEGER ⊂ RATIONAL ⊂
;;;; REAL ⊂ NUMBER, NULL ⊂ LIST ⊂ SEQUENCE, etc.
;;;;
;;;; This file implements the symbol-vs-symbol case via an
;;;; explicit ancestor table. Compound type specs (`(integer 1 3)`,
;;;; `(or A B)`, `(satisfies P)`, …) fall through as
;;;; (values nil nil) — "I don't know" — which is the spec-permitted
;;;; conservative answer. CL code that checks the SECOND value
;;;; (certainty) handles that gracefully; code that only checks the
;;;; first treats unknown as a negative.
;;;;
;;;; Returns are (values result certainty):
;;;;   (T T)     — TYPE1 is definitely a subtype of TYPE2
;;;;   (NIL T)   — TYPE1 is definitely NOT a subtype of TYPE2
;;;;   (NIL NIL) — undecided (compound spec, satisfies, etc.)
;;;;
;;;; Lives in Library/ (loaded by init.lisp) rather than core.lisp
;;;; because the ancestor table is large and only needed once the
;;;; CL surface is being exercised.

;; ── Ancestor table ─────────────────────────────────────────────────
;;
;; Each entry: (TYPE . LIST-OF-DIRECT-SUPERTYPES). T is the implicit
;; supertype of every type (handled separately). NIL is the implicit
;; subtype of every type (handled separately).
;;
;; Mirrors the type-recognition table inside `typep_shim`
;; (src/ncl-runtime/src/abi.rs ~ "match name.as_ref()"). Keep them
;; in sync: a type SUBTYPEP recognises should also TYPEP-check.

(defparameter *subtype-supertypes*
  '(;; numeric tower
    (fixnum integer)
    (bignum integer)
    (integer rational)
    (ratio rational)
    (rational real)
    (short-float float)
    (single-float float)
    (double-float float)
    (long-float float)
    (float real)
    (real number)
    (complex number)
    ;; symbols + booleans
    (null symbol list)
    (boolean symbol)
    (keyword symbol)
    ;; sequences
    (cons list)
    (list sequence)
    (string vector)
    (simple-string string)
    (simple-vector vector)
    (vector array)
    (vector sequence)
    (array sequence) ;; not strictly CL — arrays aren't always seqs;
                     ;; safe for the corpus.
    ;; functions
    (compiled-function function)
    (generic-function function)
    (standard-generic-function generic-function)
    (standard-object t)
    (built-in-class class)
    (standard-class class)
    (structure-class class)
    (class standard-object)
    (method standard-object)
    (standard-method method)
    ;; chars + strings
    (standard-char character)
    (base-char character)
    (extended-char character)
    ;; atoms
    (atom t)
    ;; hash table
    (hash-table t)
    ;; package
    (package t)))

(defun %type-ancestors (type)
  "Return TYPE plus all transitive supertypes per *subtype-supertypes*.
   Always includes T at the tail (every type is a subtype of T)."
  (cond
    ((eq type 't) '(t))
    ((eq type 'nil) '())  ; NIL is the bottom; its ancestor set is
                          ; "everything" but we represent that
                          ; through the (eq type1 'nil) short-circuit
                          ; in subtypep itself.
    (t
     (let ((result (list type))
           (frontier (list type)))
       (loop
         (cond
           ((null frontier) (return (append (reverse result) '(t))))
           (t
            (let* ((current (car frontier))
                   (parents (cdr (assoc current *subtype-supertypes*))))
              (setq frontier (cdr frontier))
              (dolist (p parents)
                (unless (member p result)
                  (setq result (cons p result))
                  (setq frontier (cons p frontier))))))))))))

(defun subtypep (type1 type2)
  "Return (VALUES RESULT CERTAINTY). See file header for semantics."
  (cond
    ;; Compound type specs — leave the analysis to a future
    ;; canonicaliser. Returning unknown is spec-conformant.
    ((or (consp type1) (consp type2))
     (values nil nil))
    ;; Same name → trivially a subtype.
    ((eq type1 type2) (values t t))
    ;; NIL is a subtype of every type.
    ((eq type1 'nil) (values t t))
    ;; T is a supertype of every type.
    ((eq type2 't) (values t t))
    ;; T is a subtype only of T (the (eq type1 type2) clause covered
    ;; that), so (subtypep 't OTHER) is NIL for any OTHER ≠ T.
    ((eq type1 't) (values nil t))
    ;; Walk ancestors of type1.
    (t
     (cond
       ((member type2 (%type-ancestors type1)) (values t t))
       ;; Both are named primitives but unrelated → definitely not
       ;; subtype. We could refine with sibling-relation analysis;
       ;; for now any named-non-ancestor is treated as a clean
       ;; "definitely not."
       (t (values nil t))))))

;; ── coerce-friendly type-of refinement ───────────────────────────
;;
;; The runtime TYPE-OF returns broad names (FIXNUM / BIGNUM / FLOAT /
;; SIMPLE-VECTOR / …). CL spec sometimes wants more specific names —
;; chapter 4 tests `(type-of (expt 2 40))` => BIGNUM and we already
;; return that, so no further refinement is needed here. Future
;; tightening can layer on top.

(provide 'types)
nil
