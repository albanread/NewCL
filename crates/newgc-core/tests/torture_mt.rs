//! MM hardening: seeded, randomized multi-mutator torture.
//!
//! The single-mutator core now has thousands of adversarial seeds on it
//! (`stochastic_workload.rs`); the *multi-mutator* paths have had far less
//! mileage. This points randomized stress at them. `N` worker threads
//! each run a seeded op stream — alloc (cons/boxed), poll, native
//! excursions (`enter_native`/`leave_native`), pin/unpin across a
//! safepoint, and **driving minor and full collections themselves** (so
//! several mutators contend to be the STW coordinator, serialized by
//! `coord_mutex`). Every worker holds a fixed set of rooted objects with
//! integrity sentinels and re-verifies them after every iteration, so a
//! lost / mis-forwarded / torn object surfaces immediately.
//!
//! Unlike the single-mutator sweep, runs are **not** bit-reproducible:
//! the seed fixes each worker's *op mix*, but the OS schedules the actual
//! interleaving, so coverage comes from many runs × seeds × schedules.
//! `newgc_core::crash::install()` is wired in, so a segfault from a bad
//! interleaving is localized (faulting address + backtrace) on the spot.
//!
//! Tunable: `NEWGC_TORTURE_SEEDS` (base seeds, default 1),
//! `NEWGC_TORTURE_ITERS` (ops per worker, default 120). The default is a
//! deliberately tiny liveness+correctness smoke for `cargo test` (one
//! multi-threaded run); real coverage comes from a deep release sweep
//! (each worker drives STW, so per-iter cost is high and long runs
//! accumulate Tenured garbage — many short seeds beat few long ones):
//!   NEWGC_TORTURE_SEEDS=300 NEWGC_TORTURE_ITERS=800 \
//!     cargo test --release -p newgc-core --test torture_mt -- --nocapture

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;

use newgc_core::{
    GcCoordinator, Generation, HeapHeader, HeapType, LispLayout, PAYLOAD_MASK, Tag, Word,
};

// Per-worker phase breadcrumbs — set before each potentially-blocking
// action so a watchdog can report what every worker is stuck doing if the
// run wedges (the hang-equivalent of the SEH crash handler).
const PH_INIT: u8 = 0;
const PH_ALLOC: u8 = 1;
const PH_POLL: u8 = 2;
const PH_NATIVE: u8 = 3;
const PH_PIN: u8 = 4;
const PH_DRIVE_MINOR: u8 = 5;
const PH_DRIVE_FULL: u8 = 6;
const PH_CHECK: u8 = 7;
const PH_DONE: u8 = 8;
const PH_MUTATE_LINK: u8 = 9;

fn phase_name(p: u8) -> &'static str {
    match p {
        PH_INIT => "init",
        PH_ALLOC => "alloc",
        PH_POLL => "poll_safepoint",
        PH_NATIVE => "enter/leave_native",
        PH_PIN => "pin+poll+unpin",
        PH_DRIVE_MINOR => "collect_minor (driving)",
        PH_DRIVE_FULL => "collect_full (driving)",
        PH_CHECK => "integrity check",
        PH_DONE => "done",
        PH_MUTATE_LINK => "rewrite link + mark_card",
        _ => "?",
    }
}

type Coord = GcCoordinator<LispLayout>;

const N_WORKERS: usize = 5;
const ROOTS_PER_WORKER: usize = 4;

// Deterministic per-worker RNG (LCG) — fixes the op mix; scheduling does
// the rest.
struct Rng {
    state: u64,
}
impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed | 1 }
    }
    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn range(&mut self, lim: usize) -> usize {
        (self.next() as usize) % lim.max(1)
    }
    fn pct(&mut self, n: u64) -> bool {
        self.next() % 100 < n
    }
}

#[derive(Copy, Clone, PartialEq)]
enum Shape {
    Cons,
    Boxed,
    /// A boxed container whose payload[1] points at a child cons. The
    /// worker periodically rewrites that pointer + marks the card — the
    /// concurrent cross-gen write-barrier path.
    Linked,
}

fn alloc_cons(m: &mut newgc_core::Mutator<LispLayout>, car: i64) -> Option<Word> {
    let p = m.try_alloc_cons_in(Generation::G0)?;
    unsafe {
        *p.as_ptr() = Word::fixnum(car).raw();
        *p.as_ptr().add(1) = Word::NIL.raw();
    }
    Some(Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons))
}

fn alloc_boxed(m: &mut newgc_core::Mutator<LispLayout>, sent: i64) -> Option<Word> {
    // header + 2 payload cells.
    let p = m.try_alloc_boxed_in(Generation::G0, 3)?;
    unsafe {
        *p.as_ptr() = HeapHeader::new(HeapType::Vector, 2).raw();
        *p.as_ptr().add(1) = Word::fixnum(sent).raw();
        *p.as_ptr().add(2) = Word::fixnum(sent ^ 0x55).raw();
    }
    Some(Word::from_ptr(p.as_ptr() as *const u8, Tag::Vector))
}

/// Allocate a linked container: a boxed object whose payload[0] is
/// `container_sent` and whose payload[1] points at a fresh child cons
/// carrying `child_sent`. Marks the card on the pointer slot (the
/// cross-gen edge), exactly as a frontend's write barrier would.
fn alloc_linked(
    m: &mut newgc_core::Mutator<LispLayout>,
    container_sent: i64,
    child_sent: i64,
) -> Option<Word> {
    let child = alloc_cons(m, child_sent)?;
    let p = m.try_alloc_boxed_in(Generation::G0, 3)?;
    unsafe {
        *p.as_ptr() = HeapHeader::new(HeapType::Vector, 2).raw();
        *p.as_ptr().add(1) = Word::fixnum(container_sent).raw();
        *p.as_ptr().add(2) = child.raw();
        m.mark_card_at(p.as_ptr().add(2) as *const u8);
    }
    Some(Word::from_ptr(p.as_ptr() as *const u8, Tag::Vector))
}

/// Read the integrity payload of a rooted object at its (possibly
/// forwarded) address: cons -> car; boxed/linked container -> payload[0].
fn payload(root: Word, shape: Shape) -> Option<i64> {
    let base = (root.raw() & PAYLOAD_MASK) as *const u64;
    let cell = match shape {
        Shape::Cons => base,                              // car at cell 0
        Shape::Boxed | Shape::Linked => unsafe { base.add(1) }, // payload[0]
    };
    unsafe { Word::from_raw(*cell).as_fixnum() }
}

/// Follow a linked container's payload[1] to its child cons and read the
/// child's car. The collector must have updated payload[1] in place when
/// it moved the child, so this resolves to the live child after any GC.
fn linked_child_car(container: Word) -> Option<i64> {
    let base = (container.raw() & PAYLOAD_MASK) as *const u64;
    let child = Word::from_raw(unsafe { *base.add(2) });
    let cbase = (child.raw() & PAYLOAD_MASK) as *const u64;
    unsafe { Word::from_raw(*cbase).as_fixnum() }
}

fn run_torture(base_seed: u64, iters: usize) {
    let coord = Coord::with_reservation(512 * 64 * 1024);
    // Cooperating workers park in microseconds; the 10 s default backstop
    // would let a (bug-induced) stall masquerade as a multi-minute hang.
    // Tighten it so the timeout is irrelevant to a correct run and a real
    // deadlock surfaces as a clean watchdog abort, not an opaque wedge.
    coord.set_safepoint_timeout(Duration::from_millis(200));
    let ready = Arc::new(Barrier::new(N_WORKERS));
    let progress = Arc::new(AtomicU64::new(0));
    let phases: Arc<Vec<AtomicU8>> =
        Arc::new((0..N_WORKERS).map(|_| AtomicU8::new(PH_INIT)).collect());
    let finished = Arc::new(AtomicBool::new(false));

    // Watchdog: if the global progress counter freezes for ~6 s the run is
    // wedged — dump each worker's phase and abort with a diagnosis (the
    // hang-equivalent of the SEH crash report). With the 200 ms safepoint
    // timeout, a mere stall keeps progress ticking, so this fires only on
    // a true deadlock.
    let watchdog = {
        let progress = Arc::clone(&progress);
        let phases = Arc::clone(&phases);
        let finished = Arc::clone(&finished);
        thread::spawn(move || {
            let mut last = 0u64;
            let mut stalls = 0u32;
            while !finished.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(2000));
                if finished.load(Ordering::Acquire) {
                    break;
                }
                let now = progress.load(Ordering::Acquire);
                if now == last {
                    stalls += 1;
                    if stalls >= 3 {
                        eprintln!(
                            "\n=== torture_mt WATCHDOG: no progress for ~6 s (seed base {base_seed:#x}) ==="
                        );
                        eprintln!("  global progress stuck at {now}");
                        for w in 0..N_WORKERS {
                            eprintln!(
                                "  worker {w}: stuck in {}",
                                phase_name(phases[w].load(Ordering::Acquire))
                            );
                        }
                        eprintln!("  => likely a multi-mutator deadlock; phases show where each is wedged");
                        use std::io::Write as _;
                        let _ = std::io::stderr().flush();
                        std::process::abort();
                    }
                } else {
                    last = now;
                    stalls = 0;
                }
            }
        })
    };

    let workers: Vec<_> = (0..N_WORKERS)
        .map(|w| {
            let c = coord.clone();
            let ready = Arc::clone(&ready);
            let progress = Arc::clone(&progress);
            let phases = Arc::clone(&phases);
            thread::spawn(move || {
                let mut m = c.register_mutator();
                let mut rng = Rng::new(base_seed ^ (w as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

                let mut roots = [Word::NIL; ROOTS_PER_WORKER];
                let mut shapes = [Shape::Cons; ROOTS_PER_WORKER];
                let mut expect = [0i64; ROOTS_PER_WORKER];
                let mut expect_child = [0i64; ROOTS_PER_WORKER];
                // Monotonic, per-worker-unique sentinel source (high bits =
                // worker id), so no two live values ever collide and a swap
                // can't be masked.
                let mut next_sent: i64 = (w as i64) << 44;
                for s in 0..ROOTS_PER_WORKER {
                    next_sent += 1;
                    roots[s] = alloc_cons(&mut m, next_sent).expect("startup alloc");
                    shapes[s] = Shape::Cons;
                    expect[s] = next_sent;
                }
                let check = |roots: &[Word; ROOTS_PER_WORKER],
                             shapes: &[Shape; ROOTS_PER_WORKER],
                             expect: &[i64; ROOTS_PER_WORKER],
                             expect_child: &[i64; ROOTS_PER_WORKER]| {
                    for s in 0..ROOTS_PER_WORKER {
                        assert_eq!(
                            payload(roots[s], shapes[s]),
                            Some(expect[s]),
                            "worker {w} slot {s} (primary) corrupted across GC"
                        );
                        if shapes[s] == Shape::Linked {
                            assert_eq!(
                                linked_child_car(roots[s]),
                                Some(expect_child[s]),
                                "worker {w} slot {s} linked child lost/corrupted across GC"
                            );
                        }
                    }
                };

                ready.wait();

                for _ in 0..iters {
                    let op = rng.range(100);
                    phases[w].store(
                        match op {
                            0..=29 => PH_ALLOC,
                            30..=42 => PH_MUTATE_LINK,
                            43..=55 => PH_NATIVE,
                            56..=66 => PH_PIN,
                            67..=81 => PH_DRIVE_MINOR,
                            82..=86 => PH_DRIVE_FULL,
                            _ => PH_POLL,
                        },
                        Ordering::Relaxed,
                    );
                    match op {
                        // alloc-replace: drop the old object (garbage), root
                        // a fresh cons / boxed / linked structure.
                        0..=29 => {
                            let s = rng.range(ROOTS_PER_WORKER);
                            match rng.range(3) {
                                0 => {
                                    next_sent += 1;
                                    if let Some(word) = alloc_cons(&mut m, next_sent) {
                                        roots[s] = word;
                                        shapes[s] = Shape::Cons;
                                        expect[s] = next_sent;
                                    }
                                }
                                1 => {
                                    next_sent += 1;
                                    if let Some(word) = alloc_boxed(&mut m, next_sent) {
                                        roots[s] = word;
                                        shapes[s] = Shape::Boxed;
                                        expect[s] = next_sent;
                                    }
                                }
                                _ => {
                                    next_sent += 1;
                                    let cont = next_sent;
                                    next_sent += 1;
                                    let child = next_sent;
                                    if let Some(word) = alloc_linked(&mut m, cont, child) {
                                        roots[s] = word;
                                        shapes[s] = Shape::Linked;
                                        expect[s] = cont;
                                        expect_child[s] = child;
                                    }
                                }
                            }
                        }
                        // rewrite a linked slot's child pointer + mark its
                        // card: the concurrent cross-gen write-barrier path
                        // (one mutator dirties a card while another may be
                        // driving a cascade that scans it).
                        30..=42 => {
                            let s = rng.range(ROOTS_PER_WORKER);
                            if shapes[s] == Shape::Linked {
                                next_sent += 1;
                                let child = next_sent;
                                if let Some(cw) = alloc_cons(&mut m, child) {
                                    // roots[s] is the container's current
                                    // address (refreshed at the last poll;
                                    // no GC has moved it since — a peer
                                    // driver waits for us to park).
                                    let base = (roots[s].raw() & PAYLOAD_MASK) as *mut u64;
                                    unsafe {
                                        *base.add(2) = cw.raw();
                                        m.mark_card_at(base.add(2) as *const u8);
                                    }
                                    expect_child[s] = child;
                                }
                            }
                        }
                        // native excursion: publish roots, "block", return.
                        43..=55 => {
                            m.enter_native(&roots);
                            std::hint::spin_loop();
                            m.leave_native(&mut roots);
                        }
                        // pin a root across a safepoint, then release it.
                        56..=66 => {
                            let s = rng.range(ROOTS_PER_WORKER);
                            let h = m.pin(roots[s]);
                            m.poll_safepoint(&mut roots);
                            m.unpin(h);
                        }
                        // drive a minor collection ourselves.
                        67..=81 => {
                            m.collect_minor(&mut roots, |_| {});
                        }
                        // drive a full collection ourselves.
                        82..=86 => {
                            m.collect_full(&mut roots, |_| {});
                        }
                        // poll (the common case).
                        _ => {
                            m.poll_safepoint(&mut roots);
                        }
                    }
                    // Always reach a safepoint + verify each iteration.
                    phases[w].store(PH_POLL, Ordering::Relaxed);
                    m.poll_safepoint(&mut roots);
                    phases[w].store(PH_CHECK, Ordering::Relaxed);
                    check(&roots, &shapes, &expect, &expect_child);
                    progress.fetch_add(1, Ordering::Relaxed);
                }
                phases[w].store(PH_DONE, Ordering::Relaxed);
                // Done: dropping `m` deregisters this mutator. A peer still
                // driving a cycle drops us from its wait set via the
                // STW-aware Drop (is_active = false + notify under
                // park_mutex), so a worker finishing early never stalls the
                // others.
            })
        })
        .collect();

    for h in workers {
        h.join().expect("worker panicked");
    }
    finished.store(true, Ordering::Release);
    watchdog.join().expect("watchdog panicked");
}

#[test]
fn torture_mt_seeded_sweep() {
    newgc_core::crash::install();
    let n_seeds: u64 = std::env::var("NEWGC_TORTURE_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let iters: usize = std::env::var("NEWGC_TORTURE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);

    for s in 0..n_seeds {
        let base = 0x7012_3400u64.wrapping_add(s);
        run_torture(base, iters);
    }
    eprintln!("torture_mt_seeded_sweep: {n_seeds} seeds x {N_WORKERS} workers x {iters} iters OK");
}
