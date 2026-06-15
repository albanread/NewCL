# Completing the NCL Compiler: a Lisp-aware type/representation inference pass

*Design document. Status: design + Slice 1 in progress.*

## 0. Thesis

NCL lowers `Value` (the read+macroexpanded s-expression) to an `Expr` tree, then
emits LLVM IR more-or-less directly and leans on LLVM `-O2` for the rest. That is
the right division of labour for *generic* optimization — LLVM owns SSA, GVN, DCE,
register allocation, instruction selection. But there is a class of optimization
LLVM **cannot** perform, because the information it needs is gone by the time we
hand it `i64` Words: knowledge of NCL's **3-bit tag scheme** and its **boxed-value
representations** (heap floats, bignums, ratios). Removing a tag-check, choosing an
unboxed register representation, or deleting a `box`/`unbox` round-trip all require
proving a *source-language type fact* — and LLVM has no model of the tag invariant.

This document specifies the one missing pass that supplies those facts: a **forward,
flow-sensitive abstract-interpretation type/representation inference over the `Expr`
tree**, inspired by Cleavir's type inferencer (SICL), coarsened to NCL's tag classes.

The key observation that makes this cheap and natural: **NCL's `Repr { Word | F64 }`
is already the value-level shadow of the type lattice.** Today `emit_expr_repr`
rediscovers the Word-vs-unboxed distinction *locally and conservatively, per node*
— `repr_as_f64_static` literally refuses an unknown runtime `Word`, with the comment
*"Sprint 1 does no type inference; that arrives in Sprints 2–4."* This pass is that
missing piece: compute the distinction **once, ahead of emit, across nodes and
control flow**, instead of emit rediscovering it pessimistically everywhere.

## 1. What we take from Cleavir (and what we deliberately do not)

We are not porting Lisp. We take two ideas:

1. **The lattice design rule** — *keep only the leaves that map to distinct machine
   representations / tags; collapse everything else to `T`.* This is the discipline
   that makes CL type inference both cheap (finite, shallow lattice ⇒ fast fixpoint)
   and sound. A value "known to be some number but maybe a bignum" buys zero fast
   path, so it collapses to `T`.
2. **Kildall + the `typeq` narrowing rule** — the generic monotone dataflow skeleton
   (lattice + transfer functions + meet + work-list), and the branch narrowing on a
   type test: on the *true* branch of `(if (floatp x) …)`, `x`'s type is `meet`ed
   with `DoubleFloat`; on the *false* branch it is `difference`d. That narrowing is
   the single highest-value rule and it is free on a tree.

What we deliberately **do not** port: reaching-definitions, full SSA liveness,
register allocation, value numbering. **LLVM already does all of that** once we emit
IR. Duplicating it in a hand-rolled SSA mid-IR would be a large build for a win LLVM
already banks. We port only the part LLVM cannot do.

## 2. Why this pass, ranked against the alternatives

It deletes code NCL emits **unconditionally today** and that LLVM **provably cannot
remove**:

- **The `coerce_to_f64` tag-check diamond** (`ncl-llvm/src/lib.rs` ~2460–2516): a
  4-block CFG — tag-test `(w&7)==3`, header-type-test `(*(w&!7)&31)==7`, inlined
  fast load at cell 2, and a `slow_bb` call to `ncl_unbox_float_checked`, joined by a
  phi. LLVM can *hoist* this out of a loop but never *delete* it, because "this Word
  is always a float" is a source-type fact. Proving `x : DoubleFloat` lets emit drop
  the diamond entirely and emit the single inlined load.
- **`repr_as_f64_static` refusing runtime Words** (~2531–2544): explicitly returns
  `None` for a runtime `Word` of unknown type. This is a *placeholder for this pass*.
  Generalizing it from "literal" to "provably float" is declaration-free unboxing.
- **`coerce_to_word` boxing runtime F64s** (~2424–2429): every unboxed double crossing
  into a Word context calls `ncl_box_float` — a young-heap allocation **and a future
  GC root**. The `box`→`unbox` round-trip (a value boxed then immediately re-unboxed
  by its sole consumer) is pure waste.
- **The fixnum `both_fixnum` diamonds** (~1415/1566/1759/2281): `Add/Sub/Mul/Lt/…`
  emit a fast/slow split. Proving both operands `Fixnum` lets emit drop the
  `*_promote` **type-dispatch** slow path (float contagion / ratio / bignum operands),
  keeping only the overflow check. (Dropping the *overflow* check needs a range proof
  we do not have — see §6.)

**Why not escape/dynamic-extent analysis first?** Real, and valuable for GC pressure,
but #3 in priority: NCL's per-op pain is arithmetic tag-checks + float boxing on
straight-line code, not closures. Escape analysis needs the capture machinery and an
interprocedural `enter↔enclose` threading for a narrower (allocation-only) win. Do it
second, on the same fixpoint substrate.

**Why not dispatch devirtualization?** Needs a sealed-world / global call-graph
assumption that a redefinable-function CL does not have. Type inference is sound
**intraprocedurally** — no whole-program assumption.

**Why not path-replication?** It is the right *second* arithmetic pass, but it
presupposes the redundant predicate is a visible IR node; in NCL the redundant tag
tests live *inside* codegen's diamonds. Inference removes the bulk by proving the type
outright; path-replication only earns its keep on the residue.

## 3. Where it fits

A **tree pass, not a new SSA/CFG mid-IR.** The `Expr` IR is a recursive tree with
*structured* control flow (`If`/`Progn`/`Let`/`FastLoop`/`InlineLoop`/`TailLoop`) and
slot-resolved reads (`Param(i)`/`Local(i)`/f64-slot reads). Abstract interpretation
over structured control flow is a textbook fit — you get branch-sensitivity *for free*
from the tree shape; you do not need basic blocks or phis to be flow-sensitive when the
control flow is already a tree.

Pipeline:

```
reader → macroexpand → lower (Value→Expr) → [NEW: optimize/infer] → ncl-llvm emit → LLVM IR → MCJIT
```

- **New module:** `src/ncl-compiler/src/optimize.rs`, an `Expr → Expr` pass (it rewrites
  the tree, wrapping proven-typed subexpressions in marker nodes the emitter reads).
- **Insertion point:** `src/ncl-compiler/src/lib.rs`, `compile_function_raw` (~line 594),
  immediately before `rewrite_self_tail_calls` — mirroring the two existing Expr→Expr
  passes (`lower::instrument_tail_for_mv`, `rewrite_self_tail_calls`). Secondary call
  site: `eval_value` (~236) for top-level forms.

The annotation carrier is a new transparent `Expr` variant (see §5) so the inferred
fact survives the crate boundary into `ncl-llvm` without a side-table (the `Expr` tree
has no stable node IDs to key a side-table on).

## 4. The lattice

Keep only leaves that codegen can distinguish; collapse everything else to `T`.

```
                       T   (any Word; a runtime test is still required)
      ┌────────┬───────┬────────┬─────────────┬────────┬──────────┐
   Fixnum    Cons   String   DoubleFloat    Symbol  Function   Null   …
                                  │
                          Unboxed(DoubleFloat)   (in an f64 register; never a GC root)
      └────────┴───────┴────────┴─────────────┴────────┴──────────┘
                       ⊥   (bottom — unreachable / no value on this path)
```

- **Leaves are exactly the codegen-distinguishable representations.** `Unboxed(DoubleFloat)`
  is a *representation refinement* of `DoubleFloat` meaning "currently an f64 register,
  not boxed" — this is precisely what `Repr::F64` already *is* at the codegen level.
  Inference lifts `Repr` into the type system.
- **Collapse to `T`:** `Integer`/`Number`/`Real` (still need a representation test),
  any mixed union like `(or Fixnum Cons)`, bignum/ratio (no fast inline path → tracking
  them buys nothing yet), `standard-object` subclasses, anything where a runtime test
  would still be needed.
- **Finiteness:** a fixed shallow set of leaves + `T` + `⊥` + the one `Unboxed`
  refinement ⇒ the fixpoint terminates.

Lattice ops:
- `join(x,x)=x`; `join(x,y)=T` for distinct leaves; `join(_,T)=T`; `join(x,⊥)=x`.
- `meet(x,x)=x`; `meet(x,y)=⊥` for distinct leaves; `meet(x,T)=x`; `meet(Unboxed(F),DoubleFloat)=Unboxed(F)`.
- `difference(x,tested)`: `x` if disjoint from `tested`, `⊥` if `x ⊑ tested` — the false
  branch of a type test.

## 5. Transfer functions + the annotation node

The analysis computes a per-program-point map `slot → AbstractType` (forward,
flow-sensitive). Reads are already slot-indexed (`Param(i)`/`Local(i)`), giving
SSA-like read clarity without building SSA.

| `Expr` node | Transfer (forward) |
|---|---|
| `Const(n)` | `Fixnum` (eql-precise) |
| float literal / f64-slot read | `Unboxed(DoubleFloat)` |
| `Param(i)` | seed from `decl_specs`: `(double-float x)`→`DoubleFloat`, `(fixnum x)`→`Fixnum`, else `T` |
| `Local(i)` read | current abstract env value for the slot |
| `Let{bindings,body}` | bind each slot to its init's inferred type; analyze body under the extended env |
| `If(c,t,e)` | analyze `c`; **sharpen** on a type test: then-env `meet`s the tested slot with the type, else-env `difference`s it; `join` the two branch-exit envs at the merge |
| `Add/Sub/Mul(a,b)` | float **contagion**: a `DoubleFloat` operand ⇒ result `Unboxed(DoubleFloat)`; `ta=tb=Fixnum` ⇒ `Fixnum` *with overflow preserved* (§6); else `T` |
| `Div(a,b)` | both `DoubleFloat` ⇒ `Unboxed(DoubleFloat)`; otherwise `T` (fixnum/fixnum may be a ratio) |
| `Lt/Gt/Le/Ge/NumEq` | result `T` (boolean Word); **propagate operand types** to emit so it picks native `fcmp` / fixnum-compare without the diamond |
| `Car/Cdr` | result `T` |
| `Cons` | `Cons` |
| `Call`/`Funcall`/`Apply` | result `T` (no interprocedural type yet — sound); clobber the type of any *boxed/closure-captured* slot to `T`; plain unboxed `Local`/`Param` reads survive |
| `SetCar`/f64-store/`StoreGlobal` | update the abstract env for the written slot |
| loop nodes | **fixpoint** over loop-carried slots: init to pre-loop type, analyze body, join back-edge types, re-iterate until the env stabilizes (≤ slots × height iterations) |

**Sound float sources (Slice 1).** Float is closed under `+,-,*,/` *of floats* and
under `(float x)`; float literals and declared `double-float` params/locals are float.
`sqrt`/`log`/`expt`/`asin`/`acos` are **not** sound float sources — they return complex
for some real inputs (`(sqrt -4)` ⇒ `#C(0 2)`) — so they collapse to `T` until value-range
reasoning exists.

**Annotation node.** A new transparent variant `Expr::TheFloat(Box<Expr>)` (and later
`TheFixnum`) carries the proven fact across the crate boundary:
- the pass wraps a subexpression in `TheFloat` iff it has proven `⊑ DoubleFloat`;
- emit treats `TheFloat(e)` as "value is definitely a heap float / unboxed double":
  `repr_as_f64_static` returns `Some`, and `coerce_to_f64` emits the single inlined
  cell-2 load with **no** diamond, no phi, no slow path.

`TheFloat` is *semantically transparent* (it evaluates `e`); it only changes the
representation/check decision. An untrusted source can therefore never make it appear
(only the sound transfer rules do), which is what keeps it safe.

## 6. How inferred types feed emit (the wins)

1. **Skip the `coerce_to_f64` diamond** for `TheFloat`/`Unboxed` operands — single
   inlined load, delete the `cf_hdr`/`cf_fast`/`cf_slow`/`cf_cont` blocks for that operand.
2. **Pick `Repr::F64` without a declaration** — generalize `repr_as_f64_static` from
   "`Repr::F64` or integer literal" to "… **or `TheFloat`**". `has_float_decl` /
   f64-param machinery becomes one *source* of the `DoubleFloat` fact among several.
3. **Drop box/unbox round-trips** — when a node is `Unboxed(DoubleFloat)` and its sole
   consumer immediately unboxes, thread `Repr::F64` straight through; delete the
   `ncl_box_float` call `coerce_to_word` would emit (an alloc + a GC root).
4. **Trim the fixnum diamond** — both operands proven `Fixnum`: drop the `*_promote`
   contagion slow path; **keep** the `sadd.with.overflow` → bignum branch.

## 7. Incremental path

- **Slice 1** — float unboxing from declarations + literals + float arithmetic, straight-line
  + `If` only (no loops, no fixnum, no overflow). Wires changes #1 and #2. Smallest slice
  that measurably deletes diamonds with zero loop/overflow reasoning to get wrong.
- **Slice 2** — loop fixpoint (loop-carried float accumulators) + the box/unbox-roundtrip
  peephole (change #3).
- **Slice 3** — `Fixnum` propagation + the trimmed fixnum diamond (change #4), gated on the
  overflow-soundness discipline.
- **Slice 4 (optional)** — `Cons`/`Null` sharpening through `IsCons`/`IsNull` tests; the hook
  for a later path-replication pass.

## 8. Validation + soundness traps

**Validation.** Gauntlet + ANSI on every slice (this pass changes numeric/type behaviour —
any divergence is a *soundness* bug, not a perf regression). A/B the emitted LLVM IR on
float/fixnum kernels to confirm the diamond blocks and `ncl_box_float` calls are gone for
proven operands (module optimization is off, so the IR is a faithful image of emit). A
**paranoid mode** flag that runs inference but still emits the checked path, asserting at
runtime that the proven type held — run the gauntlet under it before trusting a new transfer
rule. Differential fuzzing of well-typed forms with the pass on/off.

**Traps (over-promotion miscompiles — coarsening is always safe; sharpening on a wrong rule
is not):**

1. **Fixnum overflow → bignum.** `x,y : Fixnum` does not prove `(+ x y) : Fixnum`. Drop the
   *contagion* slow path on type info; **never** drop the overflow→bignum branch without a
   range proof. Be conservative: keep overflow.
2. **Numeric tower / contagion.** Encode contagion in `Add/Mul`; collapse to `T` for
   `truncate`/`rem`/`/` where the result representation depends on values. When in doubt, `T`.
3. **NaN / comparisons.** Propagate float-ness *to* the compare emit so it stays on the
   NaN-correct ordered `fcmp` path — never let a float operand take an integer-compare path.
4. **`⊥` is reachability, not a value.** Never lower a `⊥`-typed operand to a concrete
   representation; treat it as "skip specialization."
5. **Mutation / aliasing.** A cell-promoted (mutated, `SetCar`-written) local is not
   single-assignment; update the env on `SetCar`, and conservatively clobber boxed/
   closure-captured slots to `T` across a `Call`. Plain unboxed `Local`/`Param` reads survive.
6. **Declaration trust policy — verify-once-then-propagate.** Do not *trust*
   `(declare (double-float x))` blind: a caller passing a non-float must not segfault (the
   Gap-1 bug). Emit **one** checked `coerce_to_f64` (or a type assertion) at the prologue,
   then inference treats `x` as proven and every *subsequent* use skips the diamond. Keeps the
   safety, deletes the *repeated* checks — which is where the cost is.

## 9. File map

- New: `src/ncl-compiler/src/optimize.rs` — lattice, abstract env, forward inference, the
  `TheFloat` rewrite. Mirror the `pub(crate)` export of `lower::instrument_tail_for_mv`.
- `src/ncl-ir/src/lib.rs` — add `Expr::TheFloat(Box<Expr>)` (transparent marker).
- `src/ncl-compiler/src/lib.rs` — call the pass in `compile_function_raw` (~594) before
  `rewrite_self_tail_calls`; and in `eval_value` (~236).
- `src/ncl-compiler/src/lower.rs` — `LocalEnv`, `decl_specs`, `has_float_decl` as fact sources.
- `src/ncl-llvm/src/lib.rs` — emit consumers: `coerce_to_f64` (drop diamond on `TheFloat`),
  `repr_as_f64_static` (accept `TheFloat`), `coerce_to_word` (skip box on round-trip),
  `emit_arith_repr` / `emit_cmp_repr`, and the fixnum `both_fixnum` diamonds (trim contagion
  path only — keep overflow).
- Reference (BSD, study-only, do not copy): `E:/CL/SICL/Code/Cleavir/Type-inference/` —
  `type-descriptor.lisp` (the coarsen-to-representation-leaves manifesto) and `transfer.lisp`
  (the `meet`/`difference` narrowing).
