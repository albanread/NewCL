# NCL PROGRAMMER'S MANUAL

*A Guide for the User of the NCL System.*
*In the manner of the LISP 1.5 Programmer's Manual.*

---

## FOREWORD

NewCL is a dialect of LISP intended for use on the modern
sixty-four-bit electronic digital computer. 

The notation is that of Common LISP as set forth by the American National Standards Institute;
the implementation traces its lineage through the Corman LISP system
of Roger Corman (1996–2015). Roger created a LISP I could actually afford and
that didnt need a mainframe.

It is the author's hope that the practitioner familiar with any one of the LISPs 
INTERLISP, MACLISP, ZetaLisp, SBCL — shall find himself entirely at home, and that one new
to the language shall find here a small set of ideas of considerable
power.

The reader is presumed to have access to the compiler, `ncl`. To
begin a session he writes, at the command prompt of his operating
system,

```
    ncl
```

whereupon the system replies with its prompt, conventionally written
as `>` in what follows. To leave the system he writes `(quit)` or
strikes the end-of-file character of his terminal.

The author has organised this manual in seven Sections, an Appendix
of evaluable forms, and a glossary of the system's principal
functions. Each Section may be read independently of the others; the
practitioner is, however, advised to read Sections 1 and 2 first.

---

## SECTION 1 — The Elementary Functions

### 1.1 Atomic Symbols and Their Evaluation

The atomic symbols of LISP are written as one would write a name in
English: a sequence of letters, digits, and a few permitted
punctuation marks. Examples are

```
    FOO          X1          NIL          *PRINT-CASE*
```

By convention the reader of NCL folds every unquoted
letter to upper case before interning; thus `foo`, `Foo`, and `FOO`
denote the same symbol. This convention may be altered through the
readtable, but the practitioner is advised to leave it alone.

A *number* is also an atom. NCL recognises:

```
    42                  a decimal integer
   -17                  a negative integer
    3.14                a floating-point number
    1.5e3               a number in scientific notation
    #b1011              a binary integer (=11)
    #o755               an octal integer (=493)
    #x1A                a hexadecimal integer (=26)
    #36rZZ              a number in radix 36 (=1295)
    3/4                 a rational number
    100000000000000000000  a "bignum" (arbitrarily large)
```

A *string* is a sequence of characters between quotation marks,
`"like so"`. A *character* is written `#\a`, `#\Space`, `#\Newline`.

To evaluate an atom the system performs the following:

> *If the atom is a number, a string, a character, or the symbol
> `NIL` or `T`, the atom evaluates to itself. If it is a keyword
> (an atom whose name begins with the colon, as `:FOO`), it
> evaluates to itself. Otherwise the atom is taken as the name of
> a variable, and the value of that variable is returned.*

The two pre-defined variables `T` and `NIL` represent truth and
falsity respectively; they also represent the empty list and the
"non-list," as we shall see. They are sacred symbols of the system;
they cannot be re-assigned.

### 1.2 S-Expressions: The Form of All Things

Every program written in LISP, and every datum upon which a LISP
program operates, is an *S-expression*. An S-expression is either
an atom or a *list*. A list is written as a sequence of
S-expressions enclosed in parentheses:

```
    (A B C)
    (1 2 3 4)
    ((A B) (C D) E)
    ()                                  this is NIL — the empty list
```

To evaluate a non-empty list `(F A1 A2 ... AN)`, the system
first evaluates each of the arguments `A1, ..., AN` in turn, then
applies the function named by `F` to the resulting values. Hence
`(* 3 4)` evaluates to 12, `(+ (- 5 3) (* 2 7))` evaluates to 16,
and so on.

Some forms — the *special forms* — alter this rule. We shall meet
them in due course.

### 1.3 The Five Elementary Functions

Following McCarthy, every operation upon S-expressions is built up
from a very small number of primitives. NCL implements
these as follows:

```
    (car   X)         the first element of the list X
    (cdr   X)         the list X with its first element removed
    (cons  X Y)       prepend X to the list Y
    (atom  X)         T if X is an atom; NIL if X is a cons
    (eq    X Y)       T if X and Y are the same object; NIL otherwise
```

The names `car` and `cdr` are historical, dating from the IBM 704
register-pair instruction the original implementation used. They
are pronounced "car" and "could-er." The combinations `caar`,
`cadr`, `cddr`, `caddr`, ... up to four levels are pre-defined;
they expand in the obvious manner — `cadr` is `(car (cdr ...))`,
and so on.

The following session illustrates:

```
    > (car '(A B C))
    A
    > (cdr '(A B C))
    (B C)
    > (cons 'A '(B C))
    (A B C)
    > (atom 'A)
    T
    > (atom '(A B))
    NIL
    > (eq 'A 'A)
    T
```

The single quote, `'X`, is read as `(quote X)` and prevents the
evaluator from looking at the form inside. Without it, `(car (A B
C))` would attempt to call a function named `A`.

### 1.4 The Conditional

The fundamental conditional of LISP 1.5 is the function `COND`;
ncl also provides `IF`, `WHEN`, and `UNLESS`. The general
shape of `COND` is

```
    (COND (P1 E1)
          (P2 E2)
          ...
          (PN EN))
```

The predicates `P1, P2, ...` are evaluated in turn until one yields
a non-`NIL` value; the corresponding `E` is then evaluated and its
value returned. Thus

```
    > (cond ((= 1 2)  'no)
            ((= 1 1)  'yes)
            (T        'fallback))
    YES
```

The `IF` form is simpler: `(IF P E1 E2)` returns the value of `E1`
when `P` is true, of `E2` otherwise. When the false branch is
omitted, `NIL` is returned. The forms `(WHEN P E ...)` and
`(UNLESS P E ...)` evaluate any number of body forms when the
predicate is, respectively, true and false; they return the value
of the last body form, or `NIL`.

---

## SECTION 2 — Definition of Functions

### 2.1 The DEFUN Form

A function is defined by the special form `DEFUN`:

```
    (DEFUN name (parameter-list) body...)
```

The name must be an atom. The parameter-list is a (possibly empty)
list of atoms naming the formal arguments. The body is one or more
S-expressions, the value of the last being the value of the function.

Hence the factorial:

```
    (DEFUN FACT (N)
      (IF (= N 0)
          1
          (* N (FACT (- N 1)))))
```

and the length of a list:

```
    (DEFUN LEN (L)
      (IF (NULL L)
          0
          (+ 1 (LEN (CDR L)))))
```

Recursive definitions, as in the foregoing, are entirely natural.
The system is compiled (not interpreted); `DEFUN` compiles its body
to machine instructions at the moment of definition.

### 2.2 LAMBDA — Functions without Names

A function may be written without being given a name, by means of
the LAMBDA form:

```
    (LAMBDA (X) (* X X))
```

denotes the squaring function. To apply a lambda to arguments, use
`FUNCALL`:

```
    > (FUNCALL (LAMBDA (X) (* X X)) 7)
    49
```

A more concise form is provided by the function-quote, written
`#'NAME` (it reads as `(FUNCTION NAME)`). Thus

```
    > (FUNCALL #'CAR '(A B C))
    A
    > (MAPCAR #'(LAMBDA (X) (* X X)) '(1 2 3 4))
    (1 4 9 16)
```

Function values are first-class. They may be passed as arguments,
stored in variables, and returned from other functions.

### 2.3 Argument-Lists with Optionals, Rest, and Keywords

The lambda-list has the same shape as that of full Common LISP.
After the required parameters the practitioner may write

```
    &OPTIONAL          followed by names whose absence is replaced by NIL
    &REST              followed by one name to receive the remaining args
                       as a list
    &KEY               followed by names whose values are taken from
                       keyword-marked arguments
```

A default value may be given in the form `(NAME DEFAULT)`. For
example:

```
    (DEFUN GREET (NAME &OPTIONAL (GREETING "Hello"))
      (FORMAT NIL "~A, ~A!" GREETING NAME))

    > (GREET "world")
    "Hello, world!"
    > (GREET "world" "Greetings")
    "Greetings, world!"
```

The `&KEY` parameters are matched by keyword, not position:

```
    (DEFUN MEMBER* (ITEM LST &KEY (TEST #'EQL))
      (COND
        ((NULL LST) NIL)
        ((FUNCALL TEST ITEM (CAR LST)) LST)
        (T (MEMBER* ITEM (CDR LST) :TEST TEST))))

    > (MEMBER* 3 '(1 2 3 4) :TEST #'=)
    (3 4)
```

### 2.4 LET, LET*, and Local Bindings

Temporary bindings are introduced by `LET`:

```
    (LET ((A 1) (B 2))
      (+ A B))
```

The variables of a `LET` are bound in parallel. To introduce them
sequentially — so that each may refer to the preceding — use
`LET*`:

```
    (LET* ((A 5)
           (B (+ A 1)))
      (* A B))                      ⇒ 30
```

For local functions the form is `FLET` (parallel) or `LABELS`
(sequential and mutually recursive). The shape is the same as
`LET` save that each binding is a function definition:

```
    (LABELS ((EVEN? (N) (IF (= N 0) T (ODD?  (- N 1))))
             (ODD?  (N) (IF (= N 0) NIL (EVEN? (- N 1)))))
      (EVEN? 10))
```

### 2.5 Sequential Evaluation, Assignment

`PROGN` evaluates a sequence of forms and returns the value of the
last. Most special forms with a "body" — `LET`, `WHEN`, `DEFUN`,
and so on — perform an implicit `PROGN`.

Assignment to a variable is by `SETQ`:

```
    (SETQ X 10)                     X is now 10
```

A more general assignment, capable of modifying any *place* in a
structure, is `SETF`:

```
    (SETF (CAR X) 'NEW)             modify the car of X
    (SETF (CDR X) '(2 3))           modify the cdr of X
    (SETF (AREF V 0) 'FIRST)        modify element 0 of vector V
    (SETF (GETHASH 'K H) "value")   modify a hash-table entry
```

Top-level variables are introduced by `DEFPARAMETER` (always
assigned) or `DEFVAR` (assigned only if previously unbound).

### 2.6 Iteration

The two principal iterative forms are `DOTIMES`, for counted loops,
and `DOLIST`, for traversal:

```
    (DOTIMES (I 5)
      (FORMAT T "~D " I))           prints  0 1 2 3 4

    (DOLIST (X '(A B C))
      (PRINT X))                    prints  A B C
```

A more general loop is supplied as `LOOP`, with `RETURN` to exit:

```
    (LET ((I 0))
      (LOOP
        (WHEN (= I 5) (RETURN I))
        (SETQ I (1+ I))))           ⇒ 5
```

Recursion remains the more natural form for many programs. The
practitioner is encouraged to use whichever leads to the clearer
program; the compiler optimises tail calls, so a tail-recursive
function does not consume stack indefinitely.

---

## SECTION 3 — The S-Expression as Data

We have noted that every program is an S-expression. The converse
is also true: every S-expression can be operated upon as data, by
the same functions that operate upon any other list.

### 3.1 QUOTE and the Self-Evaluating Atoms

The reader-macro `'X` is shorthand for `(QUOTE X)`. The form
`(QUOTE X)` returns `X` unevaluated. Without the quote, `(CAR (A B
C))` would try to call a function named `A`; with it, `(CAR '(A B
C))` yields `A`.

A keyword (`:FOO`) is its own value, as are numbers, strings, and
characters. They do not need to be quoted.

### 3.2 Backquote, Comma, and Splice

For building list-shaped output it is convenient to write a list
*almost* literally, but with selected positions filled in by the
values of expressions. This is the *backquote*, `` ` ``:

```
    (LET ((X 10))
      `(THE VALUE IS ,X))           ⇒ (THE VALUE IS 10)
```

Inside a `` ` `` form, a `,EXPR` substitutes the value of `EXPR`,
and `,@EXPR` splices a list-valued expression in:

```
    (LET ((MIDDLE '(B C D)))
      `(A ,@MIDDLE E))              ⇒ (A B C D E)
```

The backquote is the principal tool for writing macros.

### 3.3 Cons-Cells, Lists, and Dotted Pairs

A list is built from cons-cells; each cell has a *car* and a *cdr*.
The list `(A B C)` is, in cons notation,

```
    (cons 'A (cons 'B (cons 'C nil)))
```

A cons whose cdr is not a list is a *dotted pair*, written `(A . B)`.
A list whose last cdr is non-`NIL` is *improper*; most list
functions do not expect them.

The list reader also accepts the `#(...)` notation for *vectors*,
which are fixed-length one-dimensional arrays:

```
    > #(10 20 30)
    #(10 20 30)
    > (AREF #(10 20 30) 1)
    20
```

### 3.4 Strings and Characters

A string is a sequence of characters. The reader recognises the
following escapes inside strings:

```
    \"      a literal quote
    \\      a literal backslash
```

The functions `LENGTH`, `CHAR`, `AREF`, and `STRING=` operate on
strings as they do on vectors. `(FORMAT NIL ...)` builds a string
without writing to any stream:

```
    > (FORMAT NIL "~A + ~A = ~A" 2 3 (+ 2 3))
    "2 + 3 = 5"
```

The format directives most commonly used are

```
    ~A      print, as for *PRINT-ESCAPE* NIL (human)
    ~S      print, as for *PRINT-ESCAPE* T   (re-readable)
    ~D      decimal integer
    ~X      hexadecimal
    ~%      newline
    ~~      literal tilde
```

---

## SECTION 4 — Symbols and Packages

### 4.1 The Symbol

A *symbol* is an atom carrying four cells:

```
    NAME            the printable name, a string
    PACKAGE         the home package (q.v.)
    VALUE CELL      the symbol's variable binding, accessed by SYMBOL-VALUE
    FUNCTION CELL   the symbol's function binding, accessed by SYMBOL-FUNCTION
```

Ncl is, as Common LISP is, a "Lisp-2": variable bindings
and function bindings live in two distinct namespaces. The variable
`X` and the function `X` may coexist; references through a function
position consult the function cell, references through a variable
position consult the value cell. To pass the function value of `X`
as data, write `#'X`.

### 4.2 Defining a Function Atomically

`DEFUN` performs one atomic write to the named symbol's
function cell. Calls to that symbol from compiled code load the
cell and indirect through it. Redefining a function is therefore
*one pointer store*; every live call site sees the new definition
on its next invocation.

This makes interactive development direct: edit, hit `DEFUN`, call
the function. No images to save, no library to relink. The old
machine-code body remains in the live image until no reference can
reach it, at which time the garbage collector reclaims it.

### 4.3 Packages

A *package* is a namespace for symbols. Three packages are
established at start-up:

```
    COMMON-LISP             the language; abbreviated CL
    COMMON-LISP-USER        the default user package; abbreviated CL-USER
    KEYWORD                 the package of self-evaluating symbols
```

A symbol in another package may be named with a colon-qualifier:

```
    COMMON-LISP:CAR          the external symbol CAR of CL
    CCL::QUIT                the internal symbol QUIT of the CCL package
```

The single colon refers to an *external* symbol (one explicitly
made visible by `EXPORT`); the double colon refers to any symbol
in the package.

The current package — that which the reader interns into when no
qualifier is given — is named by the special variable `*PACKAGE*`.
A session begins in `COMMON-LISP-USER`.

---

## SECTION 5 — Macros

A *macro* is a function from program-text to program-text.
NCL expands macros at compile-time; the resulting
S-expression is what the compiler actually sees.

A macro is defined by `DEFMACRO`:

```
    (DEFMACRO WHILE (TEST &REST BODY)
      `(LOOP
         (UNLESS ,TEST (RETURN))
         ,@BODY))
```

Thereafter the programmer may write `(WHILE (> N 0) (PRINT N)
(SETQ N (1- N)))`, and the compiler will see

```
    (LOOP
      (UNLESS (> N 0) (RETURN))
      (PRINT N)
      (SETQ N (1- N)))
```

The body of a `DEFMACRO` runs at compile time; its arguments are
the *unevaluated* sub-expressions of the call. The value returned
is the replacement S-expression.

A handful of the most useful macros (`WHEN`, `UNLESS`, `LET`,
`LET*`, `COND`, `DOLIST`, `DOTIMES`, `LOOP`, `WHILE`, `CASE`,
`HANDLER-CASE`, `WITH-OPEN-FILE`, `WITH-OUTPUT-TO-STRING`,
`MULTIPLE-VALUE-BIND`) are pre-defined in the user-Lisp portion of
the standard library. Their source is in `Lisp/core.lisp` and is
intended to be read.

### 5.1 Gensym

When a macro introduces a name, that name must not collide with one
the user has bound. The function `GENSYM` returns a fresh symbol
distinct from every interned one:

```
    (DEFMACRO MY-WHEN (TEST &REST BODY)
      (LET ((G (GENSYM)))
        `(LET ((,G ,TEST))
           (IF ,G (PROGN ,@BODY) NIL))))
```

The local `G` will be a name such as `#:G273` that the user cannot
have written by accident.

---

## SECTION 6 — Objects

NCL implements the Common LISP Object System (CLOS) in
the manner of Closette — the metaobject-protocol implementation
described by Kiczales, des Rivières and Bobrow. 
The version here is based on the Corman Lisp version.
The four central operators are `DEFCLASS`, `MAKE-INSTANCE`, `DEFGENERIC`, and
`DEFMETHOD`.

### 6.1 Classes and Instances

```
    (DEFCLASS ANIMAL ()
      ((NAME  :INITARG :NAME  :ACCESSOR NAME)
       (SOUND :INITFORM "..." :ACCESSOR SOUND)))

    (DEFCLASS DOG (ANIMAL)
      ((SOUND :INITFORM "Woof")))
```

A class is constructed with `MAKE-INSTANCE`:

```
    > (DEFPARAMETER REX (MAKE-INSTANCE 'DOG :NAME "Rex"))
    REX
    > (NAME REX)
    "Rex"
    > (SOUND REX)
    "Woof"
```

The slot options `:INITARG`, `:INITFORM`, `:READER`, `:WRITER`, and
`:ACCESSOR` have the meanings the practitioner expects from any CL
implementation.

### 6.2 Generic Functions and Methods

A *generic function* is a function whose behaviour depends upon
the classes of its arguments. It is declared with `DEFGENERIC`,
and individual *methods* are added with `DEFMETHOD`:

```
    (DEFGENERIC SPEAK (A))

    (DEFMETHOD SPEAK ((A ANIMAL))
      (FORMAT NIL "~A says ~A" (NAME A) (SOUND A)))

    (DEFMETHOD SPEAK ((A DOG))
      (FORMAT NIL "[bark] ~A" (CALL-NEXT-METHOD)))

    > (SPEAK REX)
    "[bark] Rex says Woof"
```

The call `(CALL-NEXT-METHOD)` invokes the next most-specific method
in the class-precedence list — here, the `ANIMAL` method.

Method qualifiers `:BEFORE`, `:AFTER`, and `:AROUND` produce side
effects before, after, and surrounding the primary method:

```
    (DEFMETHOD SPEAK :BEFORE ((A ANIMAL))
      (FORMAT T "*pause*~%"))
```

`EQL`-specializers may be used to specialise upon a particular
value:

```
    (DEFMETHOD GREET ((LANG (EQL :FRENCH)))
      "Bonjour")
```

Multiple inheritance is supported; class precedence follows the
standard linearisation. The functions `FIND-CLASS`, `CLASS-OF`,
`CLASS-PRECEDENCE-LIST`, `SUBCLASSP`, and `CLOS-TYPEP` are
available for introspection.

---

## SECTION 7 — Conditions, I/O, and the World

### 7.1 Conditions and Their Handling

An exceptional circumstance is signalled by the function `ERROR`,
which raises a condition. The form `HANDLER-CASE` is the principal
way to catch and recover:

```
    (HANDLER-CASE
        (/ X Y)
      (DIVISION-BY-ZERO ()
        :INFINITE)
      (ERROR (C)
        (FORMAT NIL "unhandled: ~A" C)))
```

The first matching clause is run; its value becomes the value of
the `HANDLER-CASE`. A clause of `(ERROR (C) ...)` catches every
condition and binds `C` to it.

Condition types are themselves classes (see Section 6); the
practitioner defines new ones with `DEFINE-CONDITION` from the
`Library/conditions.lisp` module, which is loaded automatically
unless `--lean` has been requested.

### 7.2 Reading and Writing

```
    (PRINT X)                       writes X, with escapes, plus a newline
    (PRINC X)                       writes X without escapes
    (FORMAT T  ...)                 writes formatted text to standard out
    (FORMAT NIL ...)                returns the formatted text as a string

    (WITH-OPEN-FILE (S "foo.txt" :DIRECTION :INPUT)
      (READ-LINE S))                returns the first line of foo.txt
```

File-system primitives `OPEN-INPUT-FILE`, `OPEN-OUTPUT-FILE`,
`READ-LINE`, `READ-CHAR-FROM`, `WRITE-STRING-TO`, `FILE-POSITION`,
`FILE-LENGTH`, `FILE-EXISTS`, and `DELETE-FILE` lie underneath; the
practitioner is encouraged to use `WITH-OPEN-FILE` and `FORMAT`
in the first instance.

### 7.3 The Graphical Display

NCL ships with a graphical substrate — *iGui* — derived
from its sister project. To draw upon the screen, open a child
window, batch a sequence of drawing calls, and submit them:

```
    (DEFUN HELLO ()
      (IGUI-START)
      (LET ((ID (OPEN-CHILD "hello")))
        (WITH-BATCH ID
          (CLEAR +SLATE+)
          (FILL-RECT 60 80 100 60 +RED+)
          (DRAW-TEXT 76 142 "red"   13 +WHITE+)
          (FILL-RECT 200 80 100 60 +GREEN+)
          (DRAW-TEXT 212 142 "green" 13 +WHITE+))))
```

The pre-defined colours `+BLACK+`, `+WHITE+`, `+RED+`, `+GREEN+`,
`+BLUE+`, `+YELLOW+`, `+SLATE+`, and `+PANEL+` are convenient;
`(RGB R G B)` and `(RGBA R G B A)` build packed colour values.
Events from the window system are obtained from `NEXT-EVENT` as a
property list with keys `:KIND`, `:CHILD-ID`, `:WIDTH`, `:HEIGHT`,
`:X`, `:Y`, and so on.

A working tour of the graphics surface, the CLOS port, and the
event loop is in `Lisp/demos/`.

### 7.4 The Foreign Function Interface

Common LISP programs may call out to native code through the
`#! ... !#` reader macro, which captures a header plist and a C
body verbatim:

```
    #!(:library "user32" :pascal "WINAPI")
    int MessageBoxA(void* hwnd, char* text, char* caption, unsigned type);
    !#
```

The Corman LISP demos use this surface heavily. The
NCL implementation honours their convention; the FFI
machinery is, however, separate from the language and is reserved
for the user. NCL's own standard library is implemented
in Rust and does not pass through the FFI.

---

## SECTION 8 — The Driver

The command-line driver is `ncl`. Its principal options are

```
    ncl                       enter the REPL with the full stdlib loaded
    ncl --eval "(...)"        evaluate one form, print the result, exit
    ncl --load file.lisp      read and evaluate every form in the file
    ncl --eval ... --eval ... evaluate forms in order
    ncl --eval ... --repl     evaluate forms, then enter the REPL
    ncl --lean                load only the bare compiler — no CLOS,
                              no Library/init.lisp
    ncl --version             print the version and exit
    ncl --help                print the usage summary
```

The driver looks for a `Library/` directory beside its executable
(or as named by the environment variable `NCL_LIBRARY`) and, if
found, prepends it to `*LOAD-PATH*` and loads `Library/init.lisp`
if present. The contents of `Library/` are an excellent place to
keep one's own utility modules; each may be loaded once per
session via `(REQUIRE 'NAME)`.

The interactive REPL prints results in the customary CL style.
Multiple values are printed one per line:

```
    > (VALUES 1 2 3)
    1
    2
    3
    > (TRUNCATE 17 5)
    3
    2
```

Striking control-D (Unix) or control-Z (Windows) ends the session,
as does `(QUIT)`.

---

## APPENDIX I — Selected Standard Functions

The following are available in every session unless `--lean` is
requested. The list is not exhaustive; see `Lisp/core.lisp` for
the full set, which is itself a readable example of the language.

### Arithmetic and Numbers

```
    +  -  *  /  1+  1-                    arithmetic, variadic
    TRUNCATE  REM  MOD                    integer division
    FLOOR  CEILING  ROUND                 division with rounding rule
    ABS  SIGNUM  MIN  MAX
    EXPT  SQRT  ISQRT  GCD  LCM
    ZEROP  PLUSP  MINUSP  ODDP  EVENP
    =  <  >  <=  >=                       numeric comparison
    SIN  COS  TAN  ASIN  ACOS  ATAN
    SINH  COSH  TANH  EXP  LOG
    NUMERATOR  DENOMINATOR  RATIONAL
    ASH  LOGAND  LOGIOR  LOGXOR  LOGNOT
    INTEGER-LENGTH  LOGBITP
```

### Cons and List

```
    CAR  CDR  CONS  LIST  LIST*           construction
    NULL  CONSP  ATOM  LISTP              predicates
    FIRST  SECOND  THIRD  ... FOURTH
    CAAR  CADR  CDAR  CDDR  CADDR ...     classical compositions
    NTH  NTHCDR  LAST  BUTLAST
    APPEND  REVERSE  NREVERSE  NCONC
    COPY-LIST  LIST-LENGTH  LENGTH
    MAPCAR  MAPC  EVERY  SOME
    MEMBER  FIND  POSITION  ASSOC
    FIND-IF  REMOVE-IF  REMOVE-IF-NOT  REMOVE
    SORT  REMOVE-DUPLICATES
    SET-DIFFERENCE  INTERSECTION  UNION
    SUBSEQ
    PUSH  POP                             (macros: cons / pop the place)
```

### Equality and Type

```
    EQ                                    identity
    EQL                                   eq, plus value-eq for numbers/chars
    EQUAL                                 structural equality
    TYPEP                                 (TYPEP X 'TYPE)
    SYMBOLP STRINGP VECTORP LISTP CONSP
    NUMBERP INTEGERP FIXNUMP BIGNUMP
    CHARACTERP FUNCTIONP
```

### Strings, Vectors, Hash-Tables

```
    LENGTH  CHAR  STRING-CHAR  STRING=    strings
    AREF  SVREF  MAKE-ARRAY  VECTOR       vectors
    MAKE-HASH-TABLE  GETHASH  REMHASH     hash tables
    CLRHASH  HASH-TABLE-COUNT  MAPHASH
```

### Symbols and Functions

```
    INTERN  MAKE-SYMBOL  GENSYM
    SYMBOL-FUNCTION  FBOUNDP  FMAKUNBOUND
    FDEFINITION  COMPLEMENT
    FUNCALL  APPLY
```

### Control and Sequencing

```
    IF  WHEN  UNLESS  COND  CASE  TYPECASE
    AND  OR  NOT
    PROGN  PROG1  PROG2
    LET  LET*  FLET  LABELS
    BLOCK  RETURN-FROM  RETURN
    LOOP  DOTIMES  DOLIST
    VALUES  MULTIPLE-VALUE-BIND  MULTIPLE-VALUE-LIST
    HANDLER-CASE  ERROR
```

### Macros, Definition, Top-Level

```
    DEFUN  DEFMACRO  LAMBDA
    DEFPARAMETER  DEFVAR
    DEFCLASS  DEFGENERIC  DEFMETHOD
    MAKE-INSTANCE  SLOT-VALUE  SLOT-BOUNDP
    FIND-CLASS  CLASS-OF  CLASS-PRECEDENCE-LIST
    CALL-NEXT-METHOD  NEXT-METHOD-P
    DEFSTRUCT
    QUOTE  FUNCTION  SETQ  SETF
```

### Input, Output, Files

```
    PRINT  PRINC  TERPRI  FORMAT
    OPEN-INPUT-FILE  OPEN-OUTPUT-FILE  OPEN-APPEND-FILE
    CLOSE-STREAM  READ-LINE  READ-CHAR-FROM  WRITE-STRING-TO
    FILE-POSITION  FILE-LENGTH  FILE-EXISTS  DELETE-FILE
    WITH-OPEN-FILE
```

### Graphics (iGui)

```
    IGUI-START   OPEN-CHILD   CLOSE-CHILD
    WITH-BATCH   CLEAR        FILL-RECT      STROKE-RECT
    FILL-OVAL    STROKE-OVAL  FILL-CIRCLE    STROKE-CIRCLE
    DRAW-LINE    DRAW-TEXT    DRAW-TEXT-STYLED  DRAW-ARC
    MEASURE-TEXT  NEXT-EVENT  RGB  RGBA
    +BLACK+ +WHITE+ +RED+ +GREEN+ +BLUE+ +YELLOW+ +SLATE+ +PANEL+
    LOG-FORMAT   LOG-WRITE
```

---

## APPENDIX II — A Worked Example

The following defines a tiny *symbolic differentiator* in the
manner of McCarthy's original LISP examples. It accepts an
algebraic expression and a variable, and returns the symbolic
derivative.

```
(DEFUN DERIV (E X)
  (COND
    ((NUMBERP E) 0)
    ((SYMBOLP E) (IF (EQ E X) 1 0))
    ((EQ (CAR E) '+)
     `(+ ,(DERIV (CADR E) X)
         ,(DERIV (CADDR E) X)))
    ((EQ (CAR E) '*)
     `(+ (* ,(CADR E)  ,(DERIV (CADDR E) X))
         (* ,(CADDR E) ,(DERIV (CADR  E) X))))
    ((EQ (CAR E) 'EXPT)
     `(* ,(CADDR E)
         (* (EXPT ,(CADR E) ,(- (CADDR E) 1))
            ,(DERIV (CADR E) X))))
    (T (ERROR "DERIV: don't know how to differentiate ~S" E))))
```

A session with the differentiator:

```
    > (DERIV 'X 'X)
    1
    > (DERIV '(+ X 1) 'X)
    (+ 1 0)
    > (DERIV '(* X X) 'X)
    (+ (* X 1) (* X 1))
    > (DERIV '(EXPT X 3) 'X)
    (* 3 (* (EXPT X 2) 1))
```

A second pass, a *simplifier* (left as an exercise to the reader),
would render these as `1`, `1`, `(* 2 X)`, and `(* 3 (EXPT X 2))`.
The author hopes the practitioner will discover for himself that
LISP is the language in which problems of this sort are most
naturally expressed.

---

## APPENDIX III — Notes on the Implementation

NCL is implemented in the Rust language. The compiler is
built on top of LLVM and emits machine code for the host
architecture; every form, including those typed at the REPL, is
compiled before it is executed.

The garbage collector is precise, generational (young / old, plus
a pinned static area for compiled code), and stop-the-world. Each
mutator thread allocates from a thread-local buffer and polls a
flag at safe-points. Pointer tags occupy three bits; fixnums have
the tag `000`, conses the tag `001`, forwarding pointers the tag
`111`.

The image is not persistent. A session is constructed from source
on every launch; an artifact cache, keyed by source hash, may be
re-used or deleted at will without loss of correctness. Source
files are the only persistence; the image is what the running
process *is*.

These details should not concern the practitioner during ordinary
use. The author records them only that the curious may know what
is going on beneath the surface.

---

*"LISP is worth learning for the profound enlightenment experience
you will have when you finally get it. That experience will make
you a better programmer for the rest of your days, even if you
never actually use LISP itself a lot."*  — Eric S. Raymond

*The author would have it that the practitioner does, in fact, use
LISP itself a lot.*
