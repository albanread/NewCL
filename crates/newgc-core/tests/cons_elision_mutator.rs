//! Faithful **Mutator / `set_stack_range`** repro for the NCL team's
//! **cons-elision** bug — the coordinator path (NOT plain `PageHeap`).
//!
//! ## Why a separate test from `cons_elision.rs`
//! `cons_elision.rs` drives the *plain* `PageHeap::collect_minor` and pins
//! via `pin_pointers_in_ranges` directly. NCL does neither: it runs through
//! `GcCoordinator`/`Mutator`, publishes a conservative stack window with
//! `Mutator::set_stack_range`, and drives minors via `Mutator::collect_minor`
//! (→ `drive_collect` → conservative pin + `PageHeap::collect_minor`). This
//! test mirrors that path exactly so the repro exercises the engine code the
//! frontend actually hits.
//!
//! ## Shape (matches the report)
//! - Small *effective* young (`GcCoordinator::new(young, old)` sets a real
//!   `young_page_cap`), so minors fire mid-build of even short lists and the
//!   partially-pinned chain is evacuated G0→G0 (and promoted G0→G1 on every
//!   3rd minor). Large old → no false OOM.
//! - Build a DESCENDING list by prepending: `acc = cons(v, acc)`.
//! - Hold `acc` in a real stack array AND keep a small ring of recent `acc`
//!   snapshots in that array. The ring entries are pointer-shaped Words that
//!   point at *interior* nodes of the in-progress chain — exactly the stale
//!   register/frame spills NCL's conservative scan sees. The published window
//!   `[lo,hi)` covers the array, so `drive_collect`'s conservative pin pins
//!   those interior nodes in place while their neighbours stay movable.
//! - After each completed list, walk it: it must read `len-1 … 1 0`. On the
//!   first mismatch, DUMP predecessor + bad node (addr/gen/cells), then panic.
//!
//! ## Bounded + deterministic + fail-fast
//! Fixed params; hard cap of `n_lists * len` conses; each minor evacuates only
//! the tiny live G0 working set (lists are short and dropped before they grow),
//! so per-minor cost is bounded and the run can't go quadratic.

#![cfg(feature = "conservative-pin")]

use newgc_core::{GcCoordinator, Generation, LispLayout, Tag, Word, PAYLOAD_MASK};

type Coord = GcCoordinator<LispLayout>;

/// Oracle: walk `head` via `cdr`, require cars `len-1, len-2, …, 0`. On the
/// first mismatch dump the predecessor + bad node (addr / generation / raw
/// cells) so one failing run pins the exact mechanism, then panic.
fn check_descending(coord: &Coord, head: Word, len: i64, list_idx: usize, conses: u64) {
    coord.with_heap(|h| {
        let gen_of = |w: Word| -> String {
            if w.raw() & 1 == 0 || w.raw() == Word::NIL.raw() {
                return "(immediate/nil)".into();
            }
            let a = (w.raw() & PAYLOAD_MASK) as *const u8;
            match h.page_of(a) {
                Some(p) => format!("{:?}", h.desc(p).generation),
                None => "(not in reservation)".into(),
            }
        };
        let mut node = head;
        let mut prev = Word::NIL;
        let mut expected = len - 1;
        let mut pos = 0i64;
        while node.raw() != Word::NIL.raw() {
            let base = (node.raw() & PAYLOAD_MASK) as *const u64;
            let car = unsafe { *base };
            let cdr = Word::from_raw(unsafe { *base.add(1) });
            let want = Word::fixnum(expected).raw();
            if car != want {
                eprintln!("=== CONS-ELISION at list {list_idx}, @ {conses} conses, position {pos} ===");
                eprintln!("  expected car fixnum {expected} (raw {want:#x}), got {car:#x}");
                eprintln!(
                    "  bad node : addr {:#x}  gen {}  cells [{:#x}, {:#x}]",
                    node.raw() & PAYLOAD_MASK,
                    gen_of(node),
                    car,
                    cdr.raw()
                );
                if prev.raw() != Word::NIL.raw() {
                    let pbase = (prev.raw() & PAYLOAD_MASK) as *const u64;
                    let (pcar, pcdr) = unsafe { (*pbase, *pbase.add(1)) };
                    eprintln!(
                        "  predecessor: addr {:#x}  gen {}  car {:#x} (fixnum {})  cdr {:#x} -> {} (this is the unrewritten pointer)",
                        prev.raw() & PAYLOAD_MASK,
                        gen_of(prev),
                        pcar,
                        pcar >> 3,
                        pcdr,
                        gen_of(Word::from_raw(pcdr)),
                    );
                }
                panic!(
                    "cons-elision: list {list_idx} position {pos}: car {car:#x} != {want:#x} \
                     (interior node spliced during partially-pinned evacuation)"
                );
            }
            expected -= 1;
            pos += 1;
            prev = node;
            node = cdr;
        }
        assert_eq!(
            pos, len,
            "cons-elision: list {list_idx}: walked {pos} nodes, expected {len} \
             (a node was dropped or the tail truncated)",
        );
    });
}

/// Walk the partial list `acc` (cars `top, top-1, …, 0`); return Some(pos) of
/// the first mismatch/break, else None. Read-only.
fn first_break(coord: &Coord, acc: Word, top: i64) -> Option<i64> {
    coord.with_heap(|h| {
        let mut node = acc;
        let mut expected = top;
        let mut pos = 0i64;
        while node.raw() != Word::NIL.raw() {
            if node.raw() & 1 == 0 { return Some(pos); }
            let a = (node.raw() & PAYLOAD_MASK) as *const u8;
            if h.page_of(a).is_none() { return Some(pos); }
            let b = a as *const u64;
            if unsafe { *b } != Word::fixnum(expected).raw() { return Some(pos); }
            expected -= 1;
            pos += 1;
            node = Word::from_raw(unsafe { *b.add(1) });
        }
        if expected != -1 { Some(pos) } else { None }
    })
}

/// One deterministic build+collect run over `n_lists` descending lists,
/// returning the total conses allocated. Parameterized so a fast smoke
/// variant and the long stress variant share one body.
fn run(young_pages: usize, old_pages: usize, n_lists: usize, len: i64, every: i64) -> u64 {
    let probe = std::env::var_os("NEWGC_PROBE_EACH").is_some();
    let coord = Coord::new(young_pages * 64 * 1024, old_pages * 64 * 1024);
    let mut m = coord.register_mutator();

    // Conservative stack window: a single slot holding the live `acc` (the
    // list HEAD), updated after every collect — exactly the shape of the
    // proven plain-`PageHeap` repro in `cons_elision.rs`, but driven through
    // the coordinator/`Mutator` path. The driver's conservative scan pins the
    // head in place each minor; `roots` ALSO carries the head as a precise
    // root so its forwarding is followed. The interior+tail are kept alive
    // (and, on the threshold cycle, promoted across the G1→Tenured cascade)
    // by the head's retained reference — which is exactly where the splice
    // dropped an interior node. Pinning only the head keeps Tenured pressure
    // identical to a non-pinned run, so a small old never false-OOMs.
    let mut window = [0u64; 1];
    let lo = window.as_ptr() as usize;
    let hi = lo + std::mem::size_of_val(&window);
    m.set_stack_range(lo, hi);

    let mut total: u64 = 0;
    for list_idx in 0..n_lists {
        let mut acc = Word::NIL;
        window[0] = 0;
        let mut since = 0i64;
        for v in 0..len {
            let p = loop {
                if let Some(p) = m.try_alloc_cons_in(Generation::G0) {
                    break p;
                }
                // G0 momentarily full: drive a minor (pins the head window),
                // then retry. `roots` carries the precise head so it's
                // rewritten to its post-collection location.
                let mut roots = [acc];
                m.collect_minor(&mut roots, |_| {});
                acc = roots[0];
                window[0] = acc.raw();
            };
            unsafe {
                *p.as_ptr() = Word::fixnum(v).raw();
                *p.as_ptr().add(1) = acc.raw();
            }
            acc = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
            window[0] = acc.raw();
            total += 1;

            since += 1;
            if since >= every {
                // Mid-build minor through the coordinator/Mutator path: pins
                // the published head window, then collects (G0→G0, or G0→G1
                // + G1→Tenured cascade on the threshold cycles).
                let mut roots = [acc];
                let res = m.collect_minor(&mut roots, |_| {});
                acc = roots[0];
                window[0] = acc.raw();
                since = 0;
                if probe {
                    if let Some(pos) = first_break(&coord, acc, v) {
                        eprintln!(
                            "[probe] FIRST BREAK list {list_idx} after {} conses at pos {pos}; \
                             promoted_g0={} promoted_g1={} cascade={}",
                            v + 1, res.promoted_g0, res.promoted_g1, res.cascade.is_some(),
                        );
                        coord.with_heap(|h| {
                            let gen_of = |raw: u64| -> String {
                                if raw & 1 == 0 { return "imm".into(); }
                                let a = (raw & PAYLOAD_MASK) as *const u8;
                                match h.page_of(a) {
                                    Some(pg) => format!("{:?}", h.desc(pg).generation),
                                    None => "out".into(),
                                }
                            };
                            // Walk to the predecessor of the break.
                            let mut node = acc;
                            let mut prev = Word::NIL;
                            let mut p = 0i64;
                            while p < pos && node.raw() != Word::NIL.raw() {
                                let b = (node.raw() & PAYLOAD_MASK) as *const u64;
                                prev = node;
                                node = Word::from_raw(unsafe { *b.add(1) });
                                p += 1;
                            }
                            if prev.raw() != Word::NIL.raw() {
                                let pb = (prev.raw() & PAYLOAD_MASK) as *const u64;
                                let (pcar, pcdr) = unsafe { (*pb, *pb.add(1)) };
                                eprintln!(
                                    "[probe] last-good node car fixnum {} gen {} cdr {:#x} -> gen {} (car-there {:#x})",
                                    (pcar as i64) >> 3, gen_of(prev.raw()), pcdr, gen_of(pcdr),
                                    if (pcdr & 1) != 0 {
                                        let a = (pcdr & PAYLOAD_MASK) as *const u64;
                                        if h.page_of(a as *const u8).is_some() { unsafe { *a } } else { 0 }
                                    } else { 0 },
                                );
                            }
                        });
                        panic!("[probe] first break at list {list_idx} pos {pos}");
                    }
                }
            }
        }
        check_descending(&coord, acc, len, list_idx, total);
        // Drop the list before the next one so old stays near-empty and
        // minors stay cheap.
        acc = Word::NIL;
        let _ = acc;
        window[0] = 0;
        if list_idx % 500 == 0 {
            eprintln!("cons_elision_mutator: list {list_idx}/{n_lists}, {total} conses");
        }
    }
    // Keep the window alive until the very end so the pinner always had a
    // valid range to scan.
    std::hint::black_box(&window);
    eprintln!(
        "cons_elision_mutator: {n_lists} lists x {len} = {total} conses, all intact"
    );
    total
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

// Long stress repro (~minutes). The splice surfaces ~1 per ~2M conses, so a
// faithful run needs a few million conses. `#[ignore]` so CI doesn't run it;
// run manually with a hard timeout:
//   cargo test -p newgc-core --test cons_elision_mutator -- --ignored --nocapture
#[test]
#[ignore = "minutes-long cons-elision stress repro on the Mutator/coordinator path"]
fn cons_list_survives_partially_pinned_evacuation_via_mutator() {
    newgc_core::crash::install();
    let young = env_usize("CONS_ELISION_YOUNG", 2);
    let old = env_usize("CONS_ELISION_OLD", 64);
    let n_lists = env_usize("CONS_ELISION_LISTS", 60_000);
    let len = env_usize("CONS_ELISION_LEN", 200) as i64;
    let every = env_usize("CONS_ELISION_EVERY", 40) as i64;
    run(young, old, n_lists, len, every);
}

// Fast smoke variant — small fixed bound so it runs in CI in well under a
// second. Does not necessarily trigger the splice; it guards the harness
// (the build/collect/walk loop) against regressions and documents the shape.
#[test]
fn cons_list_partially_pinned_smoke() {
    newgc_core::crash::install();
    run(2, 64, 200, 120, 25);
}

// Bounded stress with HARDCODED params (no env needed) so it runs under the
// permitted `cargo test <name> -- --nocapture` invocation. ~4M conses; fails
// fast on the first splice with the rich dump. `#[ignore]` (minutes-long).
#[test]
#[ignore = "minutes-long bounded cons-elision stress repro (Mutator path); ~4M conses"]
fn cons_list_partially_pinned_bounded_stress() {
    newgc_core::crash::install();
    // 20_000 lists x 200 = 4_000_000 conses; minor every 40 conses.
    run(2, 64, 20_000, 200, 40);
}

// TEMP diagnostic: dangling-cdr scan after each evac chunk. The last
// [dangle-cdr] line before the splice panic pinpoints the phase/chunk that
// leaves a cdr pointing into a freed page. Remove before commit.
#[test]
#[ignore = "temp diagnostic: per-minor first-break + cross-gen mark-gap"]
fn cons_list_partially_pinned_dangle_cdr_probe() {
    unsafe {
        std::env::set_var("NEWGC_PROBE_EACH", "1");
        std::env::set_var("NEWGC_DEBUG_REWRITE", "1");
    }
    newgc_core::crash::install();
    run(2, 64, 20_000, 200, 40);
}
