# NCL User Guide

*A Lisp for working hackers, by way of Corman, McCarthy, and a JIT
that thinks it's an interpreter.*

---

## FOREWORD

NCL — NewCormanLisp — is a Common Lisp. It compiles every form
through LLVM, runs on a precise generational GC, talks fluently to
Win32, and answers a `>` prompt within a few hundred milliseconds of
the icon being clicked. Its language is the Corman Lisp dialect of
ANSI Common Lisp; its implementation is a from-scratch Rust core.
The compatibility lives at the source level. Recompile from source
and your old `.lisp` files run.

If you know SBCL, CCL, Clozure, or Allegro, nothing in here will
surprise you for long. If you came up on Corman and miss it, the
demos still run. If LISP for you means McCarthy 1960, you'll find
everything from `(car (cdr ...))` up to `defmethod :around` waiting
for you, and the same `(quote foo)` trick still works.

You start a session by typing

```
    ncl
```

at the command prompt of your operating system. The system answers
with `ncl>`. You leave by typing `(quit)`, `(exit)`, or by striking
end-of-file (Ctrl+D on Unix, Ctrl+Z+Enter on Windows). Everything
else is detail.

This manual has eight sections, three appendices, and a glossary's
worth of functions in Appendix I. Section 1 and 2 are the floor;
everything else assumes them. Beyond that you can read in any
order — Section 6 (CLOS) and Section 7 (conditions, I/O, FFI,
graphics) are written to stand alone.

---

## SECTION 1 — Atoms, Lists, and the Five Primitives

### 1.1 Atoms

An *atom* is a symbol, a number, a string, or a character. Symbols
are written as plain words:

```
    FOO          X1          NIL          *PRINT-CASE*
```

The reader folds unquoted letters to upper case before interning, so
`foo`, `Foo`, and `FOO` denote the same symbol. You can change the
readtable to keep case, but: don't. The whole standard library
expects upper-case interned names and you will spend an afternoon
debugging if you fight it.

NCL recognises numbers in the full Common Lisp menagerie:

```
    42                  fixnum (signed 61-bit on x86-64)
   -17                  ditto, negative
    100000000000000000000  bignum (arbitrary precision; integer math
                          promotes through fixnum → bignum on overflow,
                          transparently)
    3/4                 ratio (exact rational; reduces automatically)
    3.14                double float
    1.5e3               float in scientific notation
    #b1011              binary integer = 11
    #o755               octal integer = 493
    #x1A                hexadecimal integer = 26
    #36rZZ              radix-36 integer = 1295
    #c(1 2)             complex number, real 1 imag 2
```

A *string* is `"like so"`; a *character* is `#\a`, `#\Space`,
`#\Newline`. Backslash inside a string escapes the next character;
the recognised escapes are `\"` and `\\`.

Atoms evaluate as follows:

> Numbers, strings, and characters evaluate to themselves. So do `T`
> and `NIL`. Keywords (symbols whose name begins with `:`) evaluate
> to themselves. Everything else is a variable name; the value of
> the variable is returned.

`T` and `NIL` are sacred. They cannot be reassigned. `NIL` is both
boolean falsity and the empty list — those are the same object.
Every other value is true.

### 1.2 Lists and S-Expressions

Every datum in Lisp is an S-expression. Every program in Lisp is an
S-expression. This is the whole trick.

```
    (A B C)
    (1 2 3 4)
    ((A B) (C D) E)
    ()                                  this is NIL — the empty list
```

To evaluate a non-empty list `(F A1 A2 ... AN)`, the system
evaluates each `Ai` in turn, then applies the function named by `F`
to the resulting values:

```
    > (* 3 4)
    12
    > (+ (- 5 3) (* 2 7))
    16
```

A handful of forms — the *special forms* — break this rule and look
at their arguments unevaluated. You'll meet them as we go.

### 1.3 The Five Primitives

Following McCarthy, you can build every list operation out of five
primitives. NCL provides them:

```
    (car   X)         the first element of list X
    (cdr   X)         X with its first element removed
    (cons  X Y)       prepend X to list Y
    (atom  X)         T if X is an atom; NIL if a cons
    (eq    X Y)       T if X and Y are the same object; NIL otherwise
```

`car` and `cdr` are pronounced "car" and "could-er", because in 1958
the IBM 704 had instructions named `CAR` and `CDR` that took the
address part and decrement part of a register pair. We've kept the
names for the same reason Latin lives on in anatomy. The
compositions `caar`, `cadr`, `cddr`, `caddr`, ..., up to four levels
deep, are predefined and expand as you'd expect.

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

The leading `'X` is `(quote X)` — a special form that hands its
argument back unevaluated. Without it, `(car (A B C))` would try to
call a function named `A`.

### 1.4 The Conditional

`COND` is the conditional McCarthy gave us; NCL also provides `IF`,
`WHEN`, `UNLESS`, `CASE`, and `TYPECASE`. The shape of `COND`:

```
    (COND (P1 E1)
          (P2 E2)
          ...
          (PN EN))
```

The predicates `P1, P2, ...` are evaluated in turn until one yields
a non-`NIL` value; the corresponding `Ei` is then evaluated and its
value returned.

```
    > (cond ((= 1 2)  'no)
            ((= 1 1)  'yes)
            (T        'fallback))
    YES
```

`IF` is the binary form: `(IF P THEN ELSE)`. The `ELSE` branch may
be omitted, in which case the false case returns `NIL`. `WHEN` and
`UNLESS` allow multiple body forms and return the last one's value.

If you find your `COND` tower wanting more visual structure, do
`(require 'ifstar)` and use Foderaro's `if*`:

```
    (if* (eq op :add) then
           (incf counter)
           (push val results)
         elseif (eq op :reset) then
           (setq counter 0)
           (setq results nil)
         else
           (error "unknown op ~A" op))
```

---

## SECTION 2 — Functions

### 2.1 DEFUN

```
    (DEFUN name (parameter-list) body...)
```

Every form in the body is evaluated; the last one's value is the
function's value. Recursion is the natural style; the compiler
optimises tail calls, so a properly written tail-recursive function
doesn't grow the stack.

```
    (DEFUN FACT (N)
      (IF (= N 0)
          1
          (* N (FACT (- N 1)))))

    (DEFUN LEN (L)
      (IF (NULL L)
          0
          (+ 1 (LEN (CDR L)))))
```

NCL is a JIT-first system: `DEFUN` compiles the body to LLVM IR and
then to machine instructions *at the moment of definition*, even at
the REPL. There is no interpreter. The form `(disassemble 'fact)`
prints the machine code, and yes, you can read it.

### 2.2 LAMBDA

Functions without names are written with `LAMBDA`:

```
    (LAMBDA (X) (* X X))
```

Apply one with `FUNCALL`, or quote it with `#'` (which reads as
`(FUNCTION ...)`):

```
    > (FUNCALL (LAMBDA (X) (* X X)) 7)
    49
    > (FUNCALL #'CAR '(A B C))
    A
    > (MAPCAR #'(LAMBDA (X) (* X X)) '(1 2 3 4))
    (1 4 9 16)
```

Functions are first-class values: pass them, store them, return them.

### 2.3 Lambda Lists: Optionals, Rest, and Keywords

The lambda list has the full CL shape:

```
    &OPTIONAL          names whose absence is NIL (or a default)
    &REST              one name; bound to the remaining args as a list
    &KEY               keyword-marked arguments, matched by name
```

A default value is given as `(NAME DEFAULT)`:

```
    (DEFUN GREET (NAME &OPTIONAL (GREETING "Hello"))
      (FORMAT NIL "~A, ~A!" GREETING NAME))

    > (GREET "world")
    "Hello, world!"
    > (GREET "world" "Greetings")
    "Greetings, world!"

    (DEFUN MEMBER* (ITEM LST &KEY (TEST #'EQL))
      (COND
        ((NULL LST) NIL)
        ((FUNCALL TEST ITEM (CAR LST)) LST)
        (T (MEMBER* ITEM (CDR LST) :TEST TEST))))

    > (MEMBER* 3 '(1 2 3 4) :TEST #'=)
    (3 4)
```

### 2.4 Local Bindings

`LET` is parallel binding; `LET*` is sequential.

```
    (LET ((A 1) (B 2))             A and B both bound; no order
      (+ A B))

    (LET* ((A 5)                   each binding sees the previous
           (B (+ A 1)))
      (* A B))                     ⇒ 30
```

Local functions get `FLET` (parallel) or `LABELS` (sequential and
mutually recursive):

```
    (LABELS ((EVEN? (N) (IF (= N 0) T   (ODD?  (- N 1))))
             (ODD?  (N) (IF (= N 0) NIL (EVEN? (- N 1)))))
      (EVEN? 10))                  ⇒ T
```

### 2.5 PROGN, SETQ, SETF

`PROGN` evaluates its body in order and returns the last value.
Most body-bearing forms (`DEFUN`, `LET`, `WHEN`, `UNLESS`, …) are
implicit `PROGN`s.

`SETQ` assigns to a variable. `SETF` assigns to a *place* — any
location accessible by an inverted-function syntax:

```
    (SETQ X 10)
    (SETF (CAR X) 'NEW)             modify the car of cons X
    (SETF (CDR X) '(2 3))
    (SETF (AREF V 0) 'FIRST)        modify element 0 of vector V
    (SETF (GETHASH 'K H) "value")   modify a hash-table entry
    (SETF (SYMBOL-FUNCTION 'F) #'G) replace F's function definition
```

`DEFPARAMETER` introduces a top-level variable, always assigned;
`DEFVAR` introduces one, assigned only if previously unbound;
`DEFCONSTANT` is the immutable form.

### 2.6 Iteration

For straightforward counted and traversal loops:

```
    (DOTIMES (I 5) (FORMAT T "~D " I))     prints  0 1 2 3 4
    (DOLIST (X '(A B C)) (PRINT X))        prints  A B C
```

The full extended `LOOP` macro is loaded by `(require 'loop)` (and
already loaded by `init.lisp` in the default setup):

```
    (LOOP FOR I FROM 1 TO 10
          WHEN (ODDP I) COLLECT I)        ⇒ (1 3 5 7 9)

    (LOOP FOR X IN '(A B C D E)
          AS I FROM 0
          DO (FORMAT T "~D: ~A~%" I X))

    (LOOP FOR LINE = (READ-LINE STREAM NIL NIL)
          WHILE LINE
          COLLECT LINE)
```

The plain unannotated `(LOOP body ...)` is also available — it
repeats body until `(RETURN value)` fires.

When recursion is clearer, use it. The compiler handles tail
calls; a tail-recursive `LEN` does not blow the stack.

---

## SECTION 3 — Data Is Code

Every program is an S-expression. The converse is the punchline:
every S-expression can be operated on as data, by the same `CAR`,
`CDR`, and `CONS` that operate on every other list. This is what
makes macros worth writing.

### 3.1 QUOTE

`'X` is shorthand for `(QUOTE X)`, which returns its argument
unevaluated. Numbers, strings, characters, and keywords don't need
the quote — they evaluate to themselves anyway.

### 3.2 Backquote, Comma, Splice

For building list-shaped output that's *almost* literal, use the
backquote `` ` ``:

```
    (LET ((X 10))
      `(THE VALUE IS ,X))                ⇒ (THE VALUE IS 10)

    (LET ((MIDDLE '(B C D)))
      `(A ,@MIDDLE E))                   ⇒ (A B C D E)
```

`,EXPR` substitutes the value; `,@EXPR` splices a list. This is the
machine you build macros on. Get familiar with it.

### 3.3 Cons-Cells and Dotted Pairs

A list is a chain of cons-cells, each with a `car` and a `cdr`. The
list `(A B C)` is, expanded:

```
    (CONS 'A (CONS 'B (CONS 'C NIL)))
```

A cons whose cdr is not a list is a *dotted pair*, written
`(A . B)`. A list whose last cdr is non-`NIL` is *improper*; most
list functions do not expect them, and the few that do (`MAPL`,
`LDIFF`) say so.

### 3.4 Vectors, Strings, Hash Tables

Strings are vectors of characters. Vectors are one-dimensional
arrays:

```
    > #(10 20 30)
    #(10 20 30)
    > (AREF #(10 20 30) 1)
    20
    > (LENGTH "hello")
    5
    > (CHAR "hello" 1)
    #\e
```

Hash tables are made with `MAKE-HASH-TABLE`; access by `GETHASH`,
mutation by `(SETF (GETHASH ...) ...)`, iteration by `MAPHASH`.

### 3.5 FORMAT

`FORMAT` is the printf you've always wanted. The first argument
chooses the destination:

```
    (FORMAT T   ...)    write to standard output
    (FORMAT NIL ...)    return the formatted string
    (FORMAT s   ...)    write to stream s
```

The directives you'll use 80% of the time:

```
    ~A      print, no escapes — humans
    ~S      print, with escapes — re-readable
    ~D      decimal integer
    ~X      hexadecimal
    ~B      binary
    ~F      fixed-point float
    ~E      scientific float
    ~R      english cardinal (~R: "one hundred twenty-three")
    ~%      newline
    ~&      newline if not already at column 0
    ~~      a literal tilde
    ~{ ~}   loop over a list — body uses ~A, ~D, etc.
```

Loading `(require 'xp)` adds Waters' XP pretty-printer machinery
(`pprint`, `pprint-logical-block`, conditional newlines, fill mode).

---

## SECTION 4 — Symbols, Cells, and Packages

### 4.1 The Symbol

A symbol is an atom carrying four cells:

```
    NAME            its printable name (a string)
    PACKAGE         its home package
    VALUE CELL      its variable binding   — SYMBOL-VALUE
    FUNCTION CELL   its function binding   — SYMBOL-FUNCTION
```

NCL, like all of Common Lisp, is a "Lisp-2": variable and function
bindings live in distinct cells. The variable `X` and the function
`X` are independent. Function-position references consult the
function cell; variable-position references consult the value cell.
To pass a function value as data, you wrap it in `#'`.

### 4.2 DEFUN Is One Atomic Pointer Store

This is the trick that makes interactive development real.
`(DEFUN FOO ...)` compiles a fresh function object and *atomically
stores it* into `FOO`'s function cell. Compiled calls to `FOO` load
the cell and indirect through it. Redefining `FOO` is one pointer
store; every live call site picks up the new definition on its next
invocation.

No images to save. No library to relink. The old machine code stays
in the static heap until no closure can still reach it, then the GC
reclaims it. This is the SBCL/CCL shape, and it is why a Lisper at
a REPL feels like they're conjuring functions out of the air. Edit,
hit DEFUN, call. That is the whole loop.

### 4.3 Packages

A package is a namespace for symbols. Three are created at startup:

```
    COMMON-LISP             the language          — nickname CL
    COMMON-LISP-USER        the default workspace — nickname CL-USER
    KEYWORD                 the home of :KEYWORDS
```

A symbol from another package is named with a colon-qualifier:

```
    COMMON-LISP:CAR          external symbol CAR of CL
    CCL::QUIT                internal symbol QUIT of CCL
```

Single colon ⇒ external (exported); double colon ⇒ any symbol in
the package. The current package — what the reader interns into
when no prefix is given — is the value of `*PACKAGE*`. A fresh
session starts in `COMMON-LISP-USER`.

---

## SECTION 5 — Macros

A macro is a function from program text to program text. NCL
expands macros at compile time. The expansion is what the compiler
actually sees.

```
    (DEFMACRO WHILE (TEST &REST BODY)
      `(LOOP
         (UNLESS ,TEST (RETURN))
         ,@BODY))
```

Now `(WHILE (> N 0) (PRINT N) (SETQ N (1- N)))` expands at compile
time to a `LOOP` with the obvious shape. The body of `DEFMACRO`
runs against unevaluated arguments and returns the replacement
S-expression. If the user wrote `(WHILE ...)`, the compiler never
sees that — it sees the `LOOP`.

A useful starter set lives in `core.lisp` and the Library — `WHEN`,
`UNLESS`, `LET`, `LET*`, `COND`, `DOLIST`, `DOTIMES`, `LOOP`,
`CASE`, `TYPECASE`, `HANDLER-CASE`, `RESTART-CASE`,
`WITH-OPEN-FILE`, `WITH-OUTPUT-TO-STRING`, `MULTIPLE-VALUE-BIND`,
`WITH-SYNCHRONIZATION`. Read the source. The standard library is
written in NCL itself, and it's an excellent way to see what good
macros look like in this dialect.

### 5.1 GENSYM

When a macro introduces a local name, that name must not collide
with one the user has bound. `GENSYM` returns a fresh symbol
distinct from every interned one:

```
    (DEFMACRO MY-WHEN (TEST &REST BODY)
      (LET ((G (GENSYM)))
        `(LET ((,G ,TEST))
           (IF ,G (PROGN ,@BODY) NIL))))
```

The local will be a name like `#:G273` that no user wrote. This is
the hygiene story; NCL does not have implicit hygiene like Scheme,
so you GENSYM by hand. It is one line; live with it.

### 5.2 Trace and Time

When debugging a macro you wrote, two utilities will save you:

```
    (REQUIRE 'TRACE)
    (TRACE FOO BAR)        wraps FOO and BAR; prints entry, exit, depth
    (UNTRACE FOO)

    (REQUIRE 'TIME)
    (TIME (FOO 30))        prints real seconds, GC stats; returns the value
```

`(MACROEXPAND-1 '(WHILE ...))` shows what your macro produced.
`(MACROEXPAND ...)` keeps expanding until no macros remain.

---

## SECTION 6 — CLOS

NCL's object system is CLOS, implemented in the manner of Closette
(Kiczales / des Rivières / Bobrow's *The Art of the Metaobject
Protocol*), seeded from Corman Lisp's port. The four operators you
need are `DEFCLASS`, `MAKE-INSTANCE`, `DEFGENERIC`, and `DEFMETHOD`.

### 6.1 Classes and Instances

```
    (DEFCLASS ANIMAL ()
      ((NAME  :INITARG :NAME  :ACCESSOR NAME)
       (SOUND :INITFORM "..." :ACCESSOR SOUND)))

    (DEFCLASS DOG (ANIMAL)
      ((SOUND :INITFORM "Woof")))

    > (DEFPARAMETER REX (MAKE-INSTANCE 'DOG :NAME "Rex"))
    REX
    > (NAME REX)
    "Rex"
    > (SOUND REX)
    "Woof"
```

Slot options `:INITARG`, `:INITFORM`, `:READER`, `:WRITER`,
`:ACCESSOR`, `:ALLOCATION`, `:TYPE`, `:DOCUMENTATION` mean what they
do everywhere else.

### 6.2 Generic Functions and Methods

A generic function dispatches on the classes of its arguments.

```
    (DEFGENERIC SPEAK (A))

    (DEFMETHOD SPEAK ((A ANIMAL))
      (FORMAT NIL "~A says ~A" (NAME A) (SOUND A)))

    (DEFMETHOD SPEAK ((A DOG))
      (FORMAT NIL "[bark] ~A" (CALL-NEXT-METHOD)))

    > (SPEAK REX)
    "[bark] Rex says Woof"
```

`CALL-NEXT-METHOD` invokes the next most-specific method on the
class precedence list. Method qualifiers `:BEFORE`, `:AFTER`, and
`:AROUND` give you the standard method combination's full power:

```
    (DEFMETHOD SPEAK :BEFORE ((A ANIMAL))
      (FORMAT T "*pause*~%"))
```

`EQL`-specializers dispatch on a particular value:

```
    (DEFMETHOD GREET ((LANG (EQL :FRENCH)))
      "Bonjour")
```

Multiple inheritance works; class precedence follows the standard
C3 linearisation. Introspection: `FIND-CLASS`, `CLASS-OF`,
`CLASS-PRECEDENCE-LIST`, `SUBCLASSP`, `CLOS-TYPEP`,
`COMPUTE-APPLICABLE-METHODS`.

Method redefinition lands the same way function redefinition does:
the generic function's dispatch table swaps in the new method body
atomically, the per-class effective-method cache is flushed, and
the next call sees the new version. The demo `clos-tour.lisp` walks
the whole story end to end.

---

## SECTION 7 — The Outside World

### 7.1 Conditions and Restarts

An exceptional circumstance is signalled by `ERROR`, which raises a
condition. The first-pass machinery is `HANDLER-CASE`:

```
    (HANDLER-CASE
        (/ X Y)
      (DIVISION-BY-ZERO () :INFINITE)
      (ERROR (C) (FORMAT NIL "unhandled: ~A" C)))
```

The first clause whose class matches the condition wins; its value
becomes the value of the `HANDLER-CASE`. `(ERROR (C) ...)` catches
everything and binds `C` to the condition.

The full Common Lisp condition system — non-unwinding handlers and
restarts — lives in the `conditions` module (loaded by default):

```
    (DEFINE-CONDITION my-trouble (error)
      ((thing :initarg :thing :reader trouble-thing)))

    (HANDLER-BIND ((my-trouble
                    (LAMBDA (C)
                      (FORMAT T "saw ~A~%" (trouble-thing C))
                      (INVOKE-RESTART 'CONTINUE))))
      (RESTART-CASE
          (ERROR 'my-trouble :thing 42)
        (CONTINUE () :recovered)))
    ⇒ :RECOVERED
```

`HANDLER-CASE` unwinds; `HANDLER-BIND` does not — it lets the
handler decide whether to transfer control via a restart or to
decline and let the condition propagate. This is the distinguishing
feature of the Lisp condition system: the *handler* and the *unwind
target* are different decisions.

Standard restart names — `ABORT`, `CONTINUE`, `USE-VALUE`,
`STORE-VALUE`, `MUFFLE-WARNING` — are recognised by the helpers.

The REPL itself is wrapped in a top-level handler-case; signalled
errors print and return you to the prompt instead of taking down
the session. As a belt-and-braces measure, the REPL also installs a
setjmp/longjmp shield around each form, so Rust-level panics
(usually from undefined-function or wrong-arity slip-ups in your
code) recover cleanly back to a fresh `ncl>`.

### 7.2 Input and Output

```
    (PRINT X)               write X with escapes, then a newline
    (PRINC X)               write X without escapes
    (FORMAT T   ...)        formatted, to *standard-output*
    (FORMAT NIL ...)        formatted, returned as a string

    (WITH-OPEN-FILE (S "foo.txt" :DIRECTION :INPUT)
      (READ-LINE S))                first line of foo.txt
```

The streams module gives you `WITH-OUTPUT-TO-STRING`,
`MAKE-STRING-INPUT-STREAM`, and friends. Under the hood the file
primitives are `OPEN-INPUT-FILE`, `OPEN-OUTPUT-FILE`,
`OPEN-APPEND-FILE`, `CLOSE-STREAM`, `READ-LINE`, `READ-CHAR-FROM`,
`WRITE-STRING-TO`, `FILE-POSITION`, `FILE-LENGTH`, `FILE-EXISTS`,
`DELETE-FILE` — but prefer the `WITH-OPEN-FILE` macro in production
code. It closes the file on any non-local exit.

### 7.3 Hot Reload

The most under-appreciated feature of a Lisp is that your editor
and your running image can stay synchronised. NCL ships a
filesystem-watch hot-reload module:

```
    > (REQUIRE 'HOT-RELOAD)        already loaded by init.lisp
    > (START-HOT-RELOAD)
    ;;; hot-reload: watching .../Lisp/Library
    T
```

Now edit a `.lisp` file in the watched directory, hit save, and the
next REPL prompt picks the file up and `(LOAD)`s it. Because
`DEFUN` is one atomic pointer store (Section 4.2), live closures
that named the old version finish on it; the next call lands on the
new code. `(CHECK-RELOADS)` runs the drain by hand from inside a
long computation.

A bad save — half a paren, a typo — is caught by a parse pre-check;
the file is skipped, the previous definitions stay in place,
nothing in the running session is touched. Re-save and the watcher
retries.

### 7.4 Threads

NCL's threads are OS threads with a precise, generational, multi-
mutator GC behind them. The native primitive surface is small;
Corman's threads package is layered on top:

```
    (REQUIRE 'THREADS)
    (DEFPARAMETER *CS* (MAKE-INSTANCE 'CRITICAL-SECTION))
    (DEFPARAMETER *COUNT* 0)

    (DEFUN BUMP ()
      (DOTIMES (I 1000)
        (WITH-SYNCHRONIZATION (CS *CS*)
          (SETQ *COUNT* (+ *COUNT* 1)))))

    (LET ((A (CREATE-THREAD #'BUMP))
          (B (CREATE-THREAD #'BUMP)))
      (DECLARE (IGNORE A B))
      ...)
```

GCs are cooperative stop-the-world: each thread polls a safe-point
flag and parks voluntarily; the collector runs only after every
mutator is parked. In tight CPU loops, drop the occasional
`(THREAD-SAFEPOINT)` so cooperative-suspend works promptly.

### 7.5 Graphics (iGui)

NCL has a graphical substrate called *iGui*, sitting on Direct2D /
DirectWrite / DXGI on Windows. To draw, open a child window, batch
a sequence of calls, and submit them:

```
    (DEFUN HELLO ()
      (IGUI-START)
      (LET ((ID (OPEN-CHILD "hello")))
        (WITH-BATCH ID
          (CLEAR +SLATE+)
          (FILL-RECT 60  80 100 60 +RED+)
          (DRAW-TEXT  76 142 "red"   13 +WHITE+)
          (FILL-RECT 200 80 100 60 +GREEN+)
          (DRAW-TEXT 212 142 "green" 13 +WHITE+))))
```

Predefined colours: `+BLACK+ +WHITE+ +RED+ +GREEN+ +BLUE+ +YELLOW+
+SLATE+ +PANEL+`. Build packed colours with `(RGB R G B)` or
`(RGBA R G B A)`. Events come from `NEXT-EVENT` as a property list
with keys `:KIND`, `:CHILD-ID`, `:WIDTH`, `:HEIGHT`, `:X`, `:Y`,
`:CHAR`, `:KEY`, and so on. A typed event-loop helper is in the
`events` module: `(WITH-EVENTS-FROM child-id (event ...) body)`.

The demos in `Lisp/demos/` are the tour:

```
    hello-igui.lisp     a rectangle, some text
    shapes.lisp         the primitive shape vocabulary
    text-styles.lisp    fonts, sizes, weights, layout boxes
    draw-square.lisp    the absolute minimum
    buttons.lisp        click handling
    click-counter.lisp  click handling with state
    paint-and-log.lisp  multi-child windows, side-channel logging
    bouncing.lisp       animation loop on a timer
    clos-tour.lisp      CLOS dispatch driving the renderer
    heap-monitor.lisp   live GC stats on a child window
    gui-repl.lisp       an in-process editor + REPL, in NCL itself
```

### 7.6 Foreign Functions and Windows API

CL programs may call out to native code. The Corman FFI surface is
preserved: the `#! ... !#` reader macro captures a C-style header
and body verbatim, parsed at load time to install a callable Lisp
shim:

```
    #!(:library "user32" :pascal "WINAPI")
    int MessageBoxA(void* hwnd, char* text, char* caption, unsigned type);
    !#

    (MESSAGE-BOX-A NIL "Hi from NCL" "NCL" 0)
```

`DEFUN-DLL` is the same machinery under another name; the Corman
demos use both interchangeably.

For Win32 specifically, NCL also ships a pre-built metadata pack
(`packs/windows_api.pack`) holding signatures for the entire Win32
API — every type, every function, every parameter — derived from
Microsoft's official `Windows.Win32.winmd`. With that loaded
(automatic when you start with `--windows`), you can call any
function by name, no FFI declaration:

```
    (WIN32 KERNEL32 GetCurrentProcessId)              ⇒ 12340
    (WIN32 USER32   MessageBoxW NULL "hi" "NCL" 0)
```

For a function you call often, `DEFWIN32` mints a named wrapper:

```
    (DEFWIN32 USER32 MessageBoxW)
    (MessageBoxW NULL "hi" "NCL" 0)
```

`SetLastError`/`GetLastError` plumbing is automatic on the ~13K
functions Microsoft has annotated for it. Calling-convention,
ANSI/Unicode pairing, and out-parameters are all picked up from the
metadata. See `docs/WINDOWS_FFI.md` for the full story.

NCL's *own* standard library does **not** go through the FFI: file
I/O, strings, time, threads, hash tables, and so on are backed by
Rust's `std`. The FFI sits beside the language for user code to
reach; the language itself stays portable.

### 7.7 Inline Assembly

When the FFI to a C library is too coarse and you want a tight
inner loop in handwritten x86_64 — a `popcnt`, a `bsr`, a
SIMD-friendly reduction — NCL gives you the `DEFASM` form:

```
    (DEFASM NAME (PARAMS...) "line1" "line2" ... "ret")
```

Each body line is a string of Intel-syntax x86_64. Parameter names
appear in the body prefixed with `#` — the assembler substitutes
each `#NAME` for the Windows x64 register that holds that
positional integer argument. The first four integer parameters
travel in `rcx`, `rdx`, `r8`, `r9`; parameters five and above sit
on the stack at `[rsp+40]`, `[rsp+48]`, and so on. Integer return
goes in `rax`. This is the Microsoft x64 calling convention, the
same one every Win32 function uses, with no NCL embellishment.

A first example:

```
    (DEFASM FAST-ADD (A B)
      "mov rax, #a"
      "add rax, #b"
      "ret")

    > (FAST-ADD 17 25)
    42
```

That worked without a single tag instruction — and it had to,
because the body never wrote one. The reason is bookkeeping NCL
arranges on your behalf: fixnums are stored as the integer shifted
left by three (the low three bits are the tag, `000` for fixnum).
Addition is *tag-preserving* — adding two left-shifted integers
gives you the left-shifted sum — so `add` between two fixnum
words drops out a fixnum word with no further work.

Multiplication is not so generous. `(a<<3) * (b<<3) = (a*b)<<6`,
which has six low zero bits where the fixnum tag wants three. You
must shift one operand back down by 3 before multiplying:

```
    (DEFASM FAST-MUL (A B)
      "mov rax, #a"
      "sar rax, 3"     ; untag A to its raw integer
      "imul rax, #b"   ; (a) * (b<<3) = (a*b)<<3 — correctly tagged
      "ret")

    > (FAST-MUL 17 25)
    425
```

This is the asm-side discipline you inherit: incoming params arrive
shifted-left-by-three; the value returned in `rax` must also be a
tagged Lisp word. For pointer-returning code that means the low
three bits encode the tag class (cons = `001`, vector = `011`,
function = `100`, string = `101`); for fixnums it means the value
left-shifted by three. NCL does not check this for you. If you
return a raw integer for what the caller treats as a fixnum, the
caller will see it as eight times your intended value. The bug
will look the way it should.

The reach-down pays off when you want a CPU instruction the language
does not expose. `popcnt`, `bsr`, `lzcnt`, `bswap`, `pdep`/`pext`,
the AES-NI family, the AVX gather/scatter ops — every one of them is
a `DEFASM` away. A bit-population count:

```
    (DEFASM POPCOUNT (N)
      "mov rax, #n"
      "sar rax, 3"            ; untag N
      "popcnt rax, rax"       ; hardware bit-count
      "shl rax, 3"            ; retag as fixnum
      "ret")

    > (POPCOUNT 255)          ; 0xFF — eight bits set
    8
    > (POPCOUNT 1024)         ; 0x400 — one bit set
    1
```

The runtime mechanics, briefly: each `DEFASM` form goes to LLVM as a
module-level inline-asm blob (Intel-syntax, with the `#name`
substitutions resolved to registers), and the JIT generates a thin
shim that adapts NCL's call ABI to the Win64 ABI and back. The shim
is what NCL's symbol cell points at; you never see it. Calling
`(POPCOUNT 255)` from compiled Lisp jumps through the shim, into
your handwritten asm, and back out — three call instructions to
get from a `defun` call site down to a `popcnt`.

The same warnings that apply to handwritten asm anywhere apply
here. The asm body sees the world as Win64 sees it: `rcx`, `rdx`,
`r8`, `r9` are the first four parameter registers; `rax`, `rcx`,
`rdx`, `r8`–`r11`, `xmm0`–`xmm5` are volatile (you may clobber them
freely); `rbx`, `rbp`, `rsi`, `rdi`, `r12`–`r15`, `xmm6`–`xmm15` are
non-volatile (save and restore if you touch them). A function
longer than a handful of lines should `push rbp / mov rbp, rsp` at
entry and pop on exit; the unwinder thanks you.

`DEFASM` is the reach-down for the cycle-pressing hot inner loop and
for getting at instructions the CPU offers but CL does not name. The
manifesto reserves "no handwritten assembly" for *our* implementation,
not for *yours*; this is where yours lives.

---

## SECTION 8 — The Driver

The command-line driver is `ncl`. Its principal options:

```
    ncl                        REPL with the full stdlib loaded
    ncl --repl                 same; explicit
    ncl -e "(...)"             evaluate one form, print, exit
    ncl --eval "(...)"         same; long form
    ncl -l file.lisp           load and evaluate every form in the file
    ncl --load file.lisp       same
    ncl -c file.lisp           dry-run: parse + macroexpand + lower
    ncl --check file.lisp      every top-level form, but execute only
                               definitions (defun, defmacro, defparameter,
                               defconstant, require). Non-definition
                               forms pass through the JIT pipeline (so
                               syntax / macro / lowering errors surface)
                               but never run. Use it as a fast lint.
    ncl --lean                 load only the bare compiler — no CLOS,
                               no Library/init.lisp. Useful for sandboxes
                               and minimal scripts.
    ncl --windows              enable the Windows surface: thread 0 runs
                               the Win32 message pump on a worker thread,
                               (windows-enabled-p) returns T, the Win32
                               metadata pack is loaded, and the win32-*
                               modules are auto-required.
    ncl --eval ... --load ... --repl
                               multiple --eval / --load / --check chain;
                               --repl drops you into the prompt after.
    ncl --version
    ncl --help
```

Short forms exist for the common flags: `-e -l -c -r -L -W -V -h`.

### 8.1 Environment Variables

```
    NCL_HEAP_BACKEND    pick the GC implementation:
                          semispace   (default — production)
                          page-heap   (under construction; see docs/GC_DESIGN.md)
    NCL_LIBRARY         override the Library/ directory location
    NCL_PACK_DIR        override the packs/ directory (Win32 metadata pack)
```

### 8.2 Library Bootstrap

The driver looks for `Library/` next to its executable (or
`NCL_LIBRARY` if you set it). If found, it prepends the directory
to `*LOAD-PATH*` and runs `Library/init.lisp` if present. The
shipping `init.lisp` `(require)`s the standard layer: `streams`,
`conditions`, `loop`, `sequences`, `trees`, `characters`, `lists`,
`places`, `numbers`, `xp`, `describe`, `events`, `hot-reload`, and
— when `--windows` is on — the win32-* modules.

Drop your own `.lisp` files into `Library/` and add `(require
'name)` to `init.lisp` to load them every session. Each module is
loaded exactly once per session; `REQUIRE` consults `*MODULES*`.

### 8.3 The REPL

Prompt is `ncl>`. Continued lines get `...>` until the input is a
complete S-expression. Hit `(exit)`, `(quit)`, Ctrl+D, or Ctrl+Z to
leave.

Multiple values print one per line:

```
    > (VALUES 1 2 3)
    1
    2
    3
    > (TRUNCATE 17 5)
    3
    2
```

Hot reload (Section 7.3) runs between prompts once you've called
`(START-HOT-RELOAD)`. Each user input is wrapped in a top-level
handler-case so errors print and the prompt comes back instead of
taking the process down.

### 8.4 The Cache

NCL JIT-compiles your image on every launch. To keep that quick,
the loader maintains a non-canonical artifact cache keyed by
`(source hash, compiler version, codegen flags)`. On launch, the
loader stats your source files and reuses cached artifacts where
the key matches. Anything stale is recompiled.

The cache is **never canonical**. Delete it whenever you want; it
will rebuild. It never round-trips through git. Source files are
the only persistence in this system — the image is what the running
process *is*.

---

## APPENDIX I — Selected Standard Functions

The list is not exhaustive; `Lisp/core.lisp` and the modules under
`Lisp/Library/` are themselves readable examples of the language.

### Arithmetic and Numbers

```
    +  -  *  /  1+  1-                    arithmetic, variadic
    TRUNCATE  REM  MOD                    integer division (3 forms)
    FLOOR  CEILING  ROUND                 division with rounding rule
    ABS  SIGNUM  MIN  MAX
    EXPT  SQRT  ISQRT  GCD  LCM
    ZEROP  PLUSP  MINUSP  ODDP  EVENP
    =  <  >  <=  >=                       numeric comparison, variadic
    SIN  COS  TAN  ASIN  ACOS  ATAN
    SINH  COSH  TANH  EXP  LOG
    NUMERATOR  DENOMINATOR  RATIONAL  RATIONALIZE
    REALPART  IMAGPART  CONJUGATE  PHASE
    ASH  LOGAND  LOGIOR  LOGXOR  LOGNOT  LOGEQV
    INTEGER-LENGTH  LOGBITP  LOGCOUNT
    RANDOM  MAKE-RANDOM-STATE
```

Bignum promotion is transparent on `+`, `-`, `*`, `EXPT`, `ASH`,
and friends — overflow a fixnum and you get a bignum without
asking.

### Cons and List

```
    CAR  CDR  CONS  LIST  LIST*           construction
    NULL  CONSP  ATOM  LISTP              predicates
    FIRST .. FOURTH                       positional accessors
    CAAR  CADR  CDAR  CDDR  CADDR ...     classical compositions
    NTH  NTHCDR  LAST  BUTLAST
    APPEND  REVERSE  NREVERSE  NCONC  REVAPPEND
    COPY-LIST  COPY-TREE  LIST-LENGTH  LENGTH
    MAPCAR  MAPC  MAPL  MAPLIST  MAPCAN  MAPCON
    EVERY  SOME  NOTEVERY  NOTANY
    MEMBER  MEMBER-IF  FIND  FIND-IF  POSITION  POSITION-IF
    ASSOC  ASSOC-IF  RASSOC  PAIRLIS  ACONS
    REMOVE  REMOVE-IF  REMOVE-IF-NOT  REMOVE-DUPLICATES
    SORT  STABLE-SORT  MERGE
    SET-DIFFERENCE  INTERSECTION  UNION  ADJOIN
    SUBSEQ  TAILP  LDIFF  TREE-EQUAL  SUBST  SUBLIS
    PUSH  POP  PUSHNEW                    place macros
```

### Equality and Type

```
    EQ                                    identity
    EQL                                   eq + value-eq for numbers/chars
    EQUAL                                 structural equality
    EQUALP                                EQUAL + case/numeric coercion
    TYPEP   TYPE-OF
    SYMBOLP STRINGP VECTORP LISTP CONSP
    NUMBERP INTEGERP FIXNUMP BIGNUMP RATIONALP FLOATP COMPLEXP
    CHARACTERP FUNCTIONP HASH-TABLE-P
```

### Strings, Vectors, Hash-Tables

```
    LENGTH  CHAR  STRING=  STRING<  STRING>           strings
    STRING-UPCASE  STRING-DOWNCASE  STRING-CAPITALIZE
    AREF  SVREF  MAKE-ARRAY  VECTOR  COPY-SEQ          vectors
    MAKE-HASH-TABLE  GETHASH  REMHASH  CLRHASH         hash tables
    HASH-TABLE-COUNT  HASH-TABLE-SIZE  MAPHASH
```

### Symbols and Functions

```
    INTERN  MAKE-SYMBOL  GENSYM  COPY-SYMBOL
    SYMBOL-FUNCTION  FBOUNDP  FMAKUNBOUND
    SYMBOL-VALUE  BOUNDP  MAKUNBOUND
    SYMBOL-NAME  SYMBOL-PACKAGE
    FDEFINITION  COMPLEMENT
    FUNCALL  APPLY  MULTIPLE-VALUE-CALL
```

### Control and Sequencing

```
    IF  WHEN  UNLESS  COND  CASE  TYPECASE  IF*
    AND  OR  NOT
    PROGN  PROG1  PROG2
    LET  LET*  FLET  LABELS  SYMBOL-MACROLET
    BLOCK  RETURN-FROM  RETURN
    LOOP  DOTIMES  DOLIST  DO  DO*  WHILE
    VALUES  MULTIPLE-VALUE-BIND  MULTIPLE-VALUE-LIST  NTH-VALUE
    HANDLER-CASE  HANDLER-BIND  RESTART-CASE  RESTART-BIND
    INVOKE-RESTART  FIND-RESTART  COMPUTE-RESTARTS
    ERROR  WARN  SIGNAL  CERROR  DEFINE-CONDITION
    UNWIND-PROTECT
```

### Macros, Definition, Top-Level

```
    DEFUN  DEFMACRO  LAMBDA  MACROLET
    DEFPARAMETER  DEFVAR  DEFCONSTANT
    DEFCLASS  DEFGENERIC  DEFMETHOD  DEFINE-METHOD-COMBINATION
    MAKE-INSTANCE  SLOT-VALUE  SLOT-BOUNDP  SLOT-MAKUNBOUND
    FIND-CLASS  CLASS-OF  CLASS-PRECEDENCE-LIST  CLOS-TYPEP
    CALL-NEXT-METHOD  NEXT-METHOD-P
    DEFSTRUCT  DEFPACKAGE  IN-PACKAGE  EXPORT  USE-PACKAGE
    QUOTE  FUNCTION  SETQ  SETF  DEFSETF  PSETF
    DECLARE  THE
    MACROEXPAND  MACROEXPAND-1
```

### Input, Output, Files

```
    READ  READ-FROM-STRING  PRINT  PRINC  TERPRI  WRITE
    FORMAT  PPRINT
    OPEN-INPUT-FILE  OPEN-OUTPUT-FILE  OPEN-APPEND-FILE
    CLOSE-STREAM  READ-LINE  READ-CHAR-FROM  WRITE-STRING-TO
    FILE-POSITION  FILE-LENGTH  FILE-EXISTS  DELETE-FILE
    WITH-OPEN-FILE  WITH-OUTPUT-TO-STRING
    MAKE-STRING-INPUT-STREAM  MAKE-STRING-OUTPUT-STREAM
```

### Threads

```
    CREATE-THREAD  CURRENT-THREAD-ID  THREAD-HANDLE
    SUSPEND-THREAD  RESUME-THREAD  TERMINATE-THREAD
    THREAD-SAFEPOINT  EXIT-THREAD
    ALLOCATE-CRITICAL-SECTION  DEALLOCATE-CRITICAL-SECTION
    ENTER-CRITICAL-SECTION  LEAVE-CRITICAL-SECTION
    WITH-SYNCHRONIZATION
```

### Graphics (iGui)

```
    IGUI-START   OPEN-CHILD   CLOSE-CHILD
    WITH-BATCH   CLEAR        FILL-RECT      STROKE-RECT
    FILL-OVAL    STROKE-OVAL  FILL-CIRCLE    STROKE-CIRCLE
    DRAW-LINE    DRAW-TEXT    DRAW-TEXT-STYLED  DRAW-ARC
    MEASURE-TEXT  NEXT-EVENT  WITH-EVENTS-FROM
    RGB  RGBA
    +BLACK+ +WHITE+ +RED+ +GREEN+ +BLUE+ +YELLOW+ +SLATE+ +PANEL+
    LOG-FORMAT   LOG-WRITE
```

### Windows FFI

```
    WIN32        DEFWIN32              dynamic + named entries
    DEFUN-DLL                          Corman-style FFI declaration
    WINDOWS-ENABLED-P                  is the surface live?
    ON-UI-THREAD   POST-TO-UI-THREAD   marshal work to the message pump
    DEFSTRUCT-WIN32                    struct layouts with packing
    DEFINE-WIN32-CALLBACK              expose a Lisp fn to native code
```

### Developer Conveniences

```
    TRACE   UNTRACE                    function-call tracing
    TIME    BENCH                      timing macros
    DESCRIBE                           interactive inspection
    MEMOIZE-FUNCTION   UNMEMOIZE-FUNCTION   automatic memoization
    LAZY-CONS   LAZY-CDR   LAZY-TAKE   SICP-style streams
    DISASSEMBLE                        show the JIT'd machine code
    GC                                 force a collection
    GC-STATS  ROOM                     heap statistics
```

---

## APPENDIX II — A Worked Example

A symbolic differentiator, in the spirit of McCarthy's original
LISP examples. Reads an algebraic expression and a variable;
returns the symbolic derivative.

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

A simplifier — left as an exercise — turns these into `1`, `1`,
`(* 2 X)`, and `(* 3 (EXPT X 2))`. This is the kind of program LISP
was made for, and it's the kind of program that is still hard to
write fluently in anything else. Try it in Python.

---

## APPENDIX III — What's Under the Hood

You don't need any of this to use NCL. The author records it
because the curious always ask.

**Compiler.** Reader → small typed IR → LLVM IR → machine code, JIT
first. Every form, including the one you just typed at the prompt,
is compiled. There is no interpreter. The compiler is written in
Rust, hosted on LLVM, and emits for the host architecture.
`disassemble` will show you what it produced.

**GC.** Precise, generational, stop-the-world. Two GC-managed
generations (young semispace + old two-semispace) plus a pinned
static area for compiled code and the loaded image. Each mutator
thread allocates from a thread-local buffer (TLAB) so the fast path
takes no locks; each polls a safe-point flag and parks
cooperatively. Pointer tags occupy three bits in a 64-bit word —
fixnums tag `000`, conses tag `001`, forwarding pointers tag `111`.
Roots come from LLVM `gc.statepoint`-emitted stack maps.

**Image.** Not persistent. A session is constructed from source on
every launch; the artifact cache keyed by `(source hash, compiler
version, codegen flags)` may be reused or deleted without loss of
correctness. Source files are the only persistence. The image is
what the running process *is*.

**Bignums.** Sign-magnitude, base-2³², stored as inline 8-byte
header + variable-length limb tail in the static heap. Overflow
promotion from fixnum is automatic at `+`, `-`, `*`, `EXPT`. The
algorithms are textbook (schoolbook multiply, Knuth-D divide); a
Karatsuba/Toom-3 ladder lives behind a tuning threshold.

**Threads and the Windows surface.** When you start with `--windows`,
thread 0 is the Win32 message pump; the Lisp evaluator runs on a
worker thread; cross-thread Win32 calls marshal back to thread 0 via
a private `WM_NCL_EXECUTE` message. Without `--windows`, Lisp runs
on thread 0 like every other CLI program. The whole UI/runtime
split lives behind a thin shim; the rest of `ncl-runtime` has no
platform awareness.

The simplicity rule applies to the rest of the system. The GC is
intentionally complex — modern generational, multi-threaded,
precisely-rooted collectors *are* complex, and that complexity buys
real throughput and correctness. Everything else is meant to fit in
a long evening's reading.

---

*"LISP is worth learning for the profound enlightenment experience
you will have when you finally get it. That experience will make
you a better programmer for the rest of your days, even if you
never actually use LISP itself a lot."*  — Eric S. Raymond

*The author would prefer you used LISP itself a lot.*
