//! GAP-010 isolation repro (from the NewOpenDylan GC-team write-up).
//!
//! Hypothesis (a): `collect_full` fails to rewrite an *externally-supplied
//! transient root slot* (a raw `&mut Word` handed in via the `visit_roots`
//! closure) when it relocates an **old-generation** object — specifically
//! a `<byte-string>` in Tenured that Pass 3 (Tenured→Tenured compaction)
//! copies to a fresh page.
//!
//! This reproduces the exact integration contract in `newgc-core`'s own
//! terms: hold a rooted byte-string, promote it to Tenured, fragment
//! Tenured with dead objects, then run a major collection visiting only
//! that one root. If the GC honors the contract, `keep` is rewritten to
//! its new address and its content is intact. If hypothesis (a) holds,
//! `keep` is left pointing at its old (now-forwarded/dead) address and
//! the content check fails — the GC bug, reproduced in isolation.

use newgc_core::{
    Generation, HeapHeader, HeapType, LispLayout, PageHeap, G0_PROMOTION_THRESHOLD,
    G1_PROMOTION_THRESHOLD, PAYLOAD_MASK, Tag, Word,
};

type Heap = PageHeap<LispLayout>;

/// Allocate a String-shaped boxed object with `n` opaque payload cells,
/// every cell filled with `byte`. For churn (varied, larger sizes than
/// `keep`, to mimic the Dylan size-512 byte-strings).
fn alloc_string_n(h: &mut Heap, byte: u8, n: usize) -> Word {
    let p = h.try_alloc_boxed_in(Generation::G0, 1 + n).expect("string alloc");
    let fill = u64::from_le_bytes([byte; 8]);
    unsafe {
        *p.as_ptr() = HeapHeader::new(HeapType::String, n as u32).raw();
        for i in 1..=n {
            *p.as_ptr().add(i) = fill;
        }
    }
    Word::from_ptr(p.as_ptr() as *const u8, Tag::String)
}

const PAYLOAD_CELLS: usize = 3; // header + 3 payload = a small <byte-string>

/// Allocate a String-shaped boxed object whose payload cells are all
/// `0xPP..` for the given byte, in G0. Returns its tagged Word.
fn alloc_string(h: &mut Heap, byte: u8) -> Word {
    let total = 1 + PAYLOAD_CELLS;
    let p = h.try_alloc_boxed_in(Generation::G0, total).expect("string alloc");
    let fill = u64::from_le_bytes([byte; 8]);
    unsafe {
        *p.as_ptr() = HeapHeader::new(HeapType::String, PAYLOAD_CELLS as u32).raw();
        for i in 1..=PAYLOAD_CELLS {
            *p.as_ptr().add(i) = fill;
        }
    }
    Word::from_ptr(p.as_ptr() as *const u8, Tag::String)
}

fn addr_of(w: Word) -> usize {
    (w.raw() & PAYLOAD_MASK) as usize
}

fn page_gen(h: &Heap, w: Word) -> Generation {
    let page = h.page_of(addr_of(w) as *const u8).expect("addr in heap");
    h.desc(page).generation
}

/// True iff `w` points at a live String with our expected header + payload.
fn string_intact(w: Word, byte: u8) -> bool {
    let base = (w.raw() & PAYLOAD_MASK) as *const u64;
    let header = HeapHeader::from_raw(unsafe { *base });
    if header.length_cells() as usize != PAYLOAD_CELLS {
        return false;
    }
    let fill = u64::from_le_bytes([byte; 8]);
    for i in 1..=PAYLOAD_CELLS {
        if unsafe { *base.add(i) } != fill {
            return false;
        }
    }
    true
}

#[test]
fn gap010_external_root_rewritten_after_major_compaction() {
    // SEH net: if `keep` goes stale and we deref it, report the faulting
    // address + backtrace instead of an opaque crash.
    newgc_core::crash::install();

    let mut h = Heap::with_reservation(64 * 64 * 1024);

    // `keep` is bracketed by garbage so Tenured is fragmented around it.
    let mut below: Vec<Word> = (0..150).map(|i| alloc_string(&mut h, (i & 0x7f) as u8)).collect();
    let mut keep = alloc_string(&mut h, 0xAB);
    let mut above: Vec<Word> = (0..150).map(|i| alloc_string(&mut h, 0x40 | (i & 0x3f) as u8)).collect();

    // Promote everything to Tenured (collect_full forces G0→G1→Tenured in
    // passes 1+2). Visit ALL roots so nothing is collected yet.
    {
        let mut roots: Vec<Word> = Vec::with_capacity(1 + below.len() + above.len());
        roots.extend_from_slice(&below);
        roots.push(keep);
        roots.extend_from_slice(&above);
        h.collect_full(|evac| {
            for r in roots.iter_mut() {
                evac.visit(r);
            }
        });
        // Write the (forwarded) roots back.
        let (b, rest) = roots.split_at(below.len());
        below.copy_from_slice(b);
        keep = rest[0];
        above.copy_from_slice(&rest[1..]);
    }

    assert_eq!(
        page_gen(&h, keep),
        Generation::Tenured,
        "keep should be in Tenured after a full collection"
    );
    assert!(string_intact(keep, 0xAB), "keep corrupted by the promoting collection");
    let addr_before = addr_of(keep);

    // Drop every other root: the bracketing garbage is now unreachable and
    // becomes dead Tenured fragmentation.
    drop(below);
    drop(above);

    // The major collection under test. Pass 3 (Tenured→Tenured) is a
    // *copying* compaction: it reclaims the dead garbage and relocates the
    // live `keep` to a fresh Tenured page. Only `keep` is rooted, supplied
    // as a raw &mut Word slot — exactly the Dylan contract.
    h.collect_full(|evac| {
        evac.visit(&mut keep);
    });

    let addr_after = addr_of(keep);
    eprintln!(
        "GAP-010: keep {addr_before:#x} -> {addr_after:#x} (moved = {})",
        addr_before != addr_after
    );

    // The decisive assertions: the slot must resolve to a live, intact
    // String. If hypothesis (a) holds, `keep` still equals `addr_before`
    // (a now-dead/forwarded location) and this fails or faults.
    assert_ne!(
        page_gen(&h, keep),
        Generation::Free,
        "keep's page is Free — root left pointing at reclaimed memory (root-rewrite gap)"
    );
    assert!(
        string_intact(keep, 0xAB),
        "keep stale/corrupted after major compaction: {addr_before:#x} -> {addr_after:#x} \
         (external root slot not rewritten by collect_full Pass 3)"
    );
}

/// GAP-010 §8 — the *actual* shape, per the Dylan team's probe: one
/// long-lived rooted multi-cell **byte-string** promoted through a MINOR
/// cascade (G0→G1→Tenured) under heavy churn, with a deref-and-validate
/// after **every** minor (the pointer-validation probe they couldn't run
/// without a debugger). The first cycle that corrupts `keep` pins the bad
/// promotion / forward. collect_full never runs here.
#[test]
fn gap010_minor_promotion_of_surviving_string() {
    newgc_core::crash::install();

    // Small heap so minors fire often; promotion is cycle-count driven.
    let mut h = Heap::with_reservation(32 * 64 * 1024);

    // The long-lived rooted string (multi-cell, opaque payload), in G0.
    let mut keep = alloc_string(&mut h, 0xAB);
    assert_eq!(page_gen(&h, keep), Generation::G0, "keep starts in G0");

    // Enough minors to promote G0→G1 (every G0_PROMOTION_THRESHOLD) and
    // cascade G1→Tenured (every G0_PROMOTION_THRESHOLD × G1_PROMOTION_THRESHOLD).
    let n_minors =
        (G0_PROMOTION_THRESHOLD as usize) * (G1_PROMOTION_THRESHOLD as usize) + 4;

    let mut last_addr = addr_of(keep);
    let mut last_gen = page_gen(&h, keep);
    for cycle in 0..n_minors {
        // Heavy churn of larger, immediately-dead byte-strings (mimics the
        // size-512 `churn` allocations in the Dylan repro).
        for _ in 0..32 {
            let _ = alloc_string_n(&mut h, 0x5A, 64);
        }
        // Minor, visiting only the long-lived root.
        h.collect_minor(|evac| {
            evac.visit(&mut keep);
        });

        // The decisive probe: deref the rewritten slot and confirm it
        // still points at a live, intact byte-string of the right size.
        let g = page_gen(&h, keep);
        let addr = addr_of(keep);
        assert_ne!(
            g,
            Generation::Free,
            "cycle {cycle}: keep on a Free page — root points at reclaimed memory \
             (gen {last_gen:?}→Free, {last_addr:#x}→{addr:#x})"
        );
        assert!(
            string_intact(keep, 0xAB),
            "cycle {cycle}: keep corrupted after minor promotion \
             (gen {last_gen:?}→{g:?}, {last_addr:#x}→{addr:#x})"
        );
        last_addr = addr;
        last_gen = g;
    }

    let g = page_gen(&h, keep);
    eprintln!(
        "GAP-010 minor promotion: keep survived {n_minors} minors, ended in {g:?} at {:#x}",
        addr_of(keep)
    );
    assert!(
        matches!(g, Generation::Tenured | Generation::G1),
        "keep should have promoted out of G0; ended in {g:?}"
    );
}

/// GAP-010 root cause: the dirty-card scan visits EVERY cell in a dirty
/// card via `visit_cell`, including a byte-string's **opaque byte
/// payload**. Vectors are safe (slots are tagged Words — fixnums classify
/// as immediates, pointers are real). But a byte-string's raw bytes can
/// alias a heap-pointer-shaped value aimed at a real object start; the
/// card scan then marks that (possibly dead) object as live AND rewrites
/// the opaque payload cell to the moved address — corrupting the string.
///
/// This is the coverage gap: no byte-payload object was ever on a scanned
/// dirty card in our tests, and synthetic vector payloads can't alias.
#[test]
fn gap010_card_scan_must_not_treat_byte_payload_as_pointer() {
    newgc_core::crash::install();
    let mut h = Heap::with_reservation(32 * 64 * 1024);

    // Promote a byte-string into G1 so the minor's card scan (which scans
    // non-G0 pages for cross-gen pointers) will consider it.
    let mut keep = alloc_string(&mut h, 0xAB);
    for _ in 0..(G0_PROMOTION_THRESHOLD as usize) {
        h.collect_minor(|evac| {
            evac.visit(&mut keep);
        });
    }
    assert_ne!(page_gen(&h, keep), Generation::G0, "keep should be promoted out of G0");

    // A fresh G0 cons that is NOT rooted — i.e. dead, should be reclaimed.
    let target = {
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *p.as_ptr() = Word::fixnum(0x1234).raw();
            *p.as_ptr().add(1) = Word::NIL.raw();
        }
        Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)
    };

    // Simulate byte-string CONTENT that happens to be a cons-shaped,
    // in-reservation pointer (raw bytes aliasing a heap pointer), and dirty
    // keep's card as if a neighbouring write barrier had fired on it.
    let keep_base = (keep.raw() & PAYLOAD_MASK) as *mut u64;
    unsafe {
        *keep_base.add(1) = target.raw(); // payload[0] := bogus "pointer"
        h.mark_card_at(keep_base.add(1) as *const u8);
    }

    // A minor. `keep` is in G1 so it doesn't move; only the card scan
    // touches its payload. The GC must treat the byte payload as opaque.
    h.collect_minor(|evac| {
        evac.visit(&mut keep);
    });

    let payload0_after =
        unsafe { *((keep.raw() & PAYLOAD_MASK) as *const u64).add(1) };
    eprintln!(
        "byte-string payload[0]: wrote {:#x}, after minor = {payload0_after:#x}",
        target.raw()
    );
    assert_eq!(
        payload0_after,
        target.raw(),
        "card scan CORRUPTED a byte-string's opaque payload — it followed the \
         raw bytes as a heap pointer and rewrote them to the relocated address"
    );
}
