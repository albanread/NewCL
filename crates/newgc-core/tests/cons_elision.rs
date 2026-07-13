//! Regression test for the NCL team's **cons-elision** bug.
//!
//! ## Hypothesis under test
//! When a **partially-pinned** cons chain is evacuated — the list HEAD is
//! pinned in place by a conservative stack window while its tail is
//! movable — the forwarding-pointer rewrite occasionally splices out a
//! single **interior** node: the predecessor's `cdr` ends up pointing at
//! the *grandchild* (`26.cdr → 24`, dropping `25`); head and tail survive.
//! Reported as ~1 list per few-million conses. No byte-strings, so it is
//! NOT the GAP-010 / mark-card (opaque-payload) class.
//!
//! ## Test design
//! - **Faithful shape**: small *effective* young (frequent mid-build
//!   minors) + large old, exactly the NCL setup. Evacuation is therefore
//!   single-chunk (Free is plentiful) — the suspected path is the
//!   forwarding rewrite of a partially-pinned chain, not a chunk boundary.
//! - **No OOM**: the reservation is large, so a minor always has Free to
//!   evacuate into (the earlier 8-page version hit a *false* GcStall — an
//!   all-live list nearly filling a tiny heap — which masked the real bug).
//! - **Deterministic**: fixed parameters ⇒ identical allocation and
//!   evacuation every run; a failure at `list_idx N` reproduces on rerun.
//! - **Bounded**: `n_lists * len` conses, hard-capped; each minor evacuates
//!   only the small live G0 working set, so per-minor cost is bounded.
//! - **Precise oracle**: every completed list must read exactly
//!   `len-1, len-2, …, 1, 0`; any splice/dup/wrong value fails with the
//!   list index, position, and expected-vs-actual car.
//!
//! Pure `PageHeap` + `collect_minor` + `pin_pointers_in_ranges`, no NCL.

#![cfg(feature = "conservative-pin")]

use newgc_core::{Generation, LispLayout, PageHeap, Tag, Word, PAYLOAD_MASK};

type Heap = PageHeap<LispLayout>;

/// Oracle: walk `head` via `cdr`, require cars `len-1, len-2, …, 0`. On the
/// first mismatch, dump the predecessor's and the bad node's address +
/// generation + raw cells, so one failing run pins the exact mechanism.
fn check_descending(h: &Heap, head: Word, len: i64, list_idx: usize, conses: u64) {
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
}

/// One minor that mirrors the NCL contract: pin the conservative stack
/// window (keeps the list head in place) AND visit `acc` as a precise root
/// (marks + evacuates the movable tail). Follow the rewritten root back.
fn minor(h: &mut Heap, acc: &mut Word, window: &mut [u64; 1], lo: usize, hi: usize) {
    window[0] = acc.raw();
    h.pin_pointers_in_ranges(Generation::G0, &[(lo, hi)]);
    let mut root = *acc;
    h.collect_minor(|e| e.visit(&mut root));
    *acc = root;
    window[0] = root.raw();
}

// `#[ignore]`: this is a minutes-long stochastic stress repro (the splice
// surfaces ~1 per ~2M conses), not a CI unit test. It currently STILL
// reproduces on this *plain* `collect_minor` path (which rebuilds the card
// table each minor) even after the Phase 3 flip-carding fix — the plain-path
// facet is under investigation. Run manually:
//   cargo test -p newgc-core --test cons_elision -- --ignored --nocapture
#[test]
#[ignore = "minutes-long cons-elision stress repro; still reproduces on the plain collect_minor path (under investigation)"]
fn cons_list_survives_partially_pinned_evacuation() {
    newgc_core::crash::install();

    // Regime that matches the report: a *tiny young* (so minors fire
    // mid-build of even short lists, evacuating the partially-pinned chain
    // G0->G0) + a large old (no false OOM). Many short lists, dropped
    // before they promote, so old stays empty and minors stay cheap/fast.
    let young_pages: usize = env_usize("CONS_ELISION_YOUNG", 2);
    let old_pages: usize = env_usize("CONS_ELISION_OLD", 64);
    let n_lists: usize = env_usize("CONS_ELISION_LISTS", 50_000);
    let len: i64 = env_usize("CONS_ELISION_LEN", 200) as i64;
    // Force a minor mid-build of every list, so every list undergoes a
    // partially-pinned evacuation.
    let every: i64 = env_usize("CONS_ELISION_EVERY", 100) as i64;

    let mut h = Heap::new(young_pages * 64 * 1024, old_pages * 64 * 1024);
    let mut window = [0u64; 1];
    let lo = window.as_ptr() as usize;
    let hi = lo + std::mem::size_of::<u64>();

    let mut total: u64 = 0;
    for list_idx in 0..n_lists {
        let mut acc = Word::NIL;
        window[0] = 0;
        let mut since = 0i64;
        for v in 0..len {
            let p = h
                .try_alloc_cons_in(Generation::G0)
                .or_else(|| {
                    // G0 momentarily full: collect (pinning head), then retry.
                    minor(&mut h, &mut acc, &mut window, lo, hi);
                    h.try_alloc_cons_in(Generation::G0)
                })
                .expect("cons alloc after minor");
            unsafe {
                *p.as_ptr() = Word::fixnum(v).raw();
                *p.as_ptr().add(1) = acc.raw();
            }
            acc = Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons);
            window[0] = acc.raw();
            total += 1;

            since += 1;
            if since >= every {
                // Mid-build minor: evacuate the partially-pinned chain.
                minor(&mut h, &mut acc, &mut window, lo, hi);
                since = 0;
            }
        }
        check_descending(&h, acc, len, list_idx, total);
        acc = Word::NIL;
        window[0] = 0;
        if list_idx % 100 == 0 {
            eprintln!("cons_elision: list {list_idx}/{n_lists}, {total} conses");
        }
    }
    eprintln!("cons_elision: {n_lists} lists x {len} = {total} conses, all intact");
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
