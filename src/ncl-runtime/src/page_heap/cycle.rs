//! GC cycle policy — minor / major drivers with age-based promotion.
//!
//! Sub-phase 8 of the Phase 3 plan in `docs/GC_DESIGN.md`. Wraps
//! sub-phase 7's `evacuate_with_roots` with a per-heap minor-cycle
//! counter so G0 objects survive `G0_PROMOTION_THRESHOLD` minor
//! cycles before being promoted to G1. Fixes the
//! "promote-on-first-survival" failure mode the design doc opens
//! with.
//!
//! ## Promotion model
//!
//! - Each `collect_minor` call increments
//!   `PageHeap::minors_since_g0_promote`.
//! - If the counter reaches `G0_PROMOTION_THRESHOLD`, that cycle's
//!   destination is G1 (instead of G0). The counter resets to 0.
//! - Otherwise the cycle's destination is G0 (mark-evacuate within
//!   the nursery).
//!
//! This is **cohort promotion**: every G0 object alive at the
//! moment the threshold cycle fires gets promoted, regardless of
//! its individual age. SBCL's per-page age (where each page has
//! its own age field driving its dest gen) is a further refinement
//! that requires per-source-page dest dispatch inside the
//! evacuator — explicitly out of scope for sub-phase 8. The
//! `PageDesc::age` field (from sub-phase 3) is reserved for that
//! refinement; the cycle policy here leaves it untouched.
//!
//! ## Why the counter starts at 0 and increments BEFORE the
//! decision
//!
//! Reading the contract literally: "G0→G1 after 3 minor cycles."
//! Cycle 1 → counter = 1, dest = G0. Cycle 2 → counter = 2, dest =
//! G0. Cycle 3 → counter = 3 ≥ threshold, dest = G1, reset to 0.
//! So objects survive at most 3 minor cycles in G0. Matches the
//! design doc spec.
//!
//! ## Major GC
//!
//! `collect_major` runs back-to-back evacuations: G1 → Tenured
//! followed by G0 → G0. Order matters: collecting G1 first means
//! the G0 evacuation's roots see the already-promoted G1
//! references and don't have to chase across an in-flight
//! collection. (Sub-phase 9 will add card barriers, at which point
//! cross-generation pointers from Tenured/G1 → G0 become a
//! first-class root source. For sub-phase 8, the major path is
//! conservative: do both generations, take the pause.)

use super::evac::{EvacResult, PageEvacuator};
use super::page_desc::Generation;
use super::space::{PageHeap, PAGE_SIZE_BYTES};

/// Minor cycles a G0 cohort survives before promotion to G1.
/// Default 3 — matches the design doc / SBCL conservative value.
pub const G0_PROMOTION_THRESHOLD: u32 = 3;

/// G0 promotion events a G1 cohort survives before promotion to
/// Tenured. Counted in promotion *events*, not minor cycles —
/// only cycles that ALREADY promoted G0 advance this counter.
/// Default 5 (so G1 graduates after 5 G0 promotions = 15 minor
/// cycles by default).
pub const G1_PROMOTION_THRESHOLD: u32 = 5;

/// Summary of one cycle. Returned from `collect_minor` /
/// `collect_major`; consumed by `(gc-stats)` and (later) the
/// trigger policy in sub-phase 10.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CollectResult {
    /// What the evacuation pass did (objects/cells copied, pages
    /// freed/flipped).
    pub evac: EvacResult,
    /// Optional second evac result, populated when the cycle
    /// cascaded (e.g., G0 promotion that also triggered G1
    /// promotion in `collect_major`, or G0→G0 followed by G1→
    /// Tenured if G1's threshold also fired).
    pub cascade: Option<EvacResult>,
    /// True if this cycle promoted G0 to G1.
    pub promoted_g0: bool,
    /// True if this cycle promoted G1 to Tenured.
    pub promoted_g1: bool,
    /// Minor-cycle counter value AFTER this cycle. Diagnostic.
    pub minors_since_g0_promote_after: u32,
}

impl PageHeap {
    /// Run a minor GC cycle: collect everything in G0.
    ///
    /// The destination is `G0` (within-generation evacuate) for
    /// the first `G0_PROMOTION_THRESHOLD - 1` cycles after the
    /// last promotion; the `G0_PROMOTION_THRESHOLD`-th cycle
    /// promotes to G1 and resets the counter.
    ///
    /// `visit_roots` is `FnMut` because cascading G0→G1 + G1→
    /// Tenured replays the same root closure on the second pass.
    /// The caller is responsible for feeding in every mutator-
    /// side root slot. See [`PageEvacuator::visit`].
    ///
    /// **Pre-condition** (inherited from sub-phase 7): the heap
    /// must have enough free pages for evacuation to complete.
    /// In particular, on a within-gen pass (`from == dest`), the
    /// heap must hold at least one Free page even though every
    /// from-page would eventually be reclaimed — page reclaim
    /// happens after the BFS, so the BFS can't borrow from the
    /// post-reclaim state. A bounded trigger lands in sub-phase
    /// 10; until then, calling this on a full heap panics with
    /// `"page heap exhausted mid-evacuation"`.
    pub fn collect_minor<F>(&mut self, mut visit_roots: F) -> CollectResult
    where
        F: FnMut(&mut PageEvacuator<'_>),
    {
        // Bump the counter first, decide afterward. Matches the
        // "cycle 3 promotes" semantics in the module docs.
        self.minors_since_g0_promote =
            self.minors_since_g0_promote.saturating_add(1);

        let promote_g0 =
            self.minors_since_g0_promote >= G0_PROMOTION_THRESHOLD;
        let dest = if promote_g0 {
            Generation::G1
        } else {
            Generation::G0
        };
        let evac_result = self
            .evacuate_with_roots(Generation::G0, dest, |e| visit_roots(e));

        let mut promoted_g1 = false;
        let mut cascade = None;
        if promote_g0 {
            // The G0 cohort just graduated. Reset its counter and
            // tick G1's. If G1 also hits its threshold, cascade
            // into G1 → Tenured. Otherwise leave G1 alone — major
            // cycles handle deeper sweeps.
            self.minors_since_g0_promote = 0;
            self.g0_promotes_since_g1_promote =
                self.g0_promotes_since_g1_promote.saturating_add(1);
            if self.g0_promotes_since_g1_promote >= G1_PROMOTION_THRESHOLD {
                // Cascading G1 → Tenured promotion. Replay the
                // caller's root closure — slots that were just
                // updated to G1 pointers (by the G0 evac above)
                // need a second visit so the evacuator targeting
                // G1 can move them to Tenured. Roots that were
                // never in G0 (and so weren't touched in pass 1)
                // also pass through; if they point into G1 now,
                // they get evacuated.
                let cas = self.evacuate_with_roots(
                    Generation::G1,
                    Generation::Tenured,
                    |e| visit_roots(e),
                );
                self.g0_promotes_since_g1_promote = 0;
                promoted_g1 = true;
                cascade = Some(cas);
            }
        }

        CollectResult {
            evac: evac_result,
            cascade,
            promoted_g0: promote_g0,
            promoted_g1,
            minors_since_g0_promote_after: self.minors_since_g0_promote,
        }
    }

    /// Run a major GC cycle: collect everything that's collectable
    /// in one stop-the-world pause. Order:
    ///
    ///   1. G1 → Tenured (promote all G1).
    ///   2. G0 → G0 (mark-evacuate within nursery).
    ///
    /// This isn't a full mark-sweep — Tenured itself isn't
    /// recollected. Tenured-only collection is sub-phase 10's
    /// trigger problem. For sub-phase 8, major is a manual hammer:
    /// "promote anything that wants to promote and clean G0 in one
    /// pass."
    ///
    /// Resets both promotion counters since the cohort accounting
    /// no longer reflects the actual layout after this pass.
    pub fn collect_major<F>(&mut self, mut visit_roots: F) -> CollectResult
    where
        F: FnMut(&mut PageEvacuator<'_>),
    {
        // G1 → Tenured first, replaying the caller's roots. Slots
        // that point into G1 get evacuated; slots that point into
        // G0 or Tenured pass through (gen-gate rejects).
        let g1_result = self.evacuate_with_roots(
            Generation::G1,
            Generation::Tenured,
            |e| visit_roots(e),
        );

        // Now G0 → G0 with the same closure. The G0 evac sees the
        // freshly-updated slots (any old G0→G1 references now
        // point at Tenured locations; the G0 evac's gen-gate
        // rejects them, so they're left alone — that's correct).
        // Caller's roots that point into G0 get evacuated within
        // G0 (mark-evacuate).
        //
        // Cross-gen references from Tenured/G1 into G0 are NOT
        // tracked yet — sub-phase 9 adds the card barrier. For
        // sub-phase 8 the caller must include such references in
        // the root closure, or accept that they'll be lost.
        let g0_result = self.evacuate_with_roots(
            Generation::G0,
            Generation::G0,
            |e| visit_roots(e),
        );

        // Reset both counters — major absorbed all pending
        // promotion debt. (Counters do not carry over from a
        // major to subsequent minors: the next minor starts a
        // fresh G0 cohort, and the next cascade fires after a
        // fresh `G1_PROMOTION_THRESHOLD` G0 promotions.)
        self.minors_since_g0_promote = 0;
        self.g0_promotes_since_g1_promote = 0;

        // Report `promoted_g1` only when the G1 pass actually
        // moved something. An empty G1 → Tenured pass is a
        // no-op and a caller using this flag for "did anything
        // graduate?" shouldn't see a phantom yes.
        let promoted_g1 = g1_result.objects_copied > 0
            || g1_result.pages_flipped > 0;

        CollectResult {
            evac: g0_result,
            cascade: Some(g1_result),
            promoted_g0: false,
            promoted_g1,
            minors_since_g0_promote_after: 0,
        }
    }

    /// Current minor-cycle counter. Diagnostic — exposed for
    /// `(gc-stats)` and tests.
    pub fn minors_since_g0_promote(&self) -> u32 {
        self.minors_since_g0_promote
    }

    /// Current G1-promotion counter. Diagnostic.
    pub fn g0_promotes_since_g1_promote(&self) -> u32 {
        self.g0_promotes_since_g1_promote
    }

    /// Run a full collection: evacuate all three generations in one
    /// stop-the-world pass, reclaiming everything in Tenured that is
    /// not reachable from the supplied root closure.
    ///
    /// ## Three-pass algorithm
    ///
    /// 1. **G0 → G1 (forced)**: every live G0 object graduates to G1
    ///    regardless of age.  Roots are supplied by the caller.
    /// 2. **G1 → Tenured (forced)**: every live G1 object (including
    ///    survivors of pass 1) graduates to Tenured.  Same root
    ///    closure is replayed.
    /// 3. **Tenured → Tenured**: the entire Tenured generation is
    ///    mark-evacuated.  After passes 1 and 2, the complete live
    ///    heap is in Tenured, so **only explicit roots are needed**
    ///    here — no card scan required.  Unreachable Tenured objects
    ///    (garbage accumulated since the last full collect) are freed.
    ///
    /// Both promotion counters are reset after the three passes so the
    /// next minor cycle starts with a clean cohort slate.
    pub fn collect_full<F>(&mut self, mut visit_roots: F) -> FullCollectResult
    where
        F: FnMut(&mut PageEvacuator<'_>),
    {
        // Pass 1: G0 → G1 (forced).
        let g0_evac = self.evacuate_with_roots(
            Generation::G0,
            Generation::G1,
            |e| visit_roots(e),
        );

        // Pass 2: G1 → Tenured (forced).
        let g1_evac = self.evacuate_with_roots(
            Generation::G1,
            Generation::Tenured,
            |e| visit_roots(e),
        );

        // Pass 3: Tenured → Tenured, explicit roots only.
        // G0 and G1 are now empty so no card scan is needed.
        let tenured_evac = self.evacuate_with_roots(
            Generation::Tenured,
            Generation::Tenured,
            |e| visit_roots(e),
        );

        self.minors_since_g0_promote = 0;
        self.g0_promotes_since_g1_promote = 0;

        FullCollectResult {
            g0_evac,
            g1_evac,
            tenured_evac,
            tenured_freed_bytes: tenured_evac.pages_freed * PAGE_SIZE_BYTES,
        }
    }
}

/// Summary of a full three-pass collection (`collect_full`).
/// Returned to the caller for diagnostics and trigger-policy
/// accounting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FullCollectResult {
    /// Evacuation result for the G0 → G1 forced pass.
    pub g0_evac: EvacResult,
    /// Evacuation result for the G1 → Tenured forced pass.
    pub g1_evac: EvacResult,
    /// Evacuation result for the Tenured → Tenured reclaim pass.
    pub tenured_evac: EvacResult,
    /// Bytes freed from Tenured by the reclaim pass.
    /// `tenured_evac.pages_freed × PAGE_SIZE_BYTES`.
    pub tenured_freed_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::word::{Tag, Word};

    fn small_heap() -> PageHeap {
        // 8 pages = 512 KB. Plenty for a few G0/G1 cohorts.
        PageHeap::with_reservation(8 * 64 * 1024)
    }

    /// Helper: allocate one cons in G0 and return its tagged Word.
    fn one_cons(h: &mut PageHeap, value: i64) -> Word {
        let p = h.try_alloc_cons_in(Generation::G0).unwrap();
        unsafe {
            *p.as_ptr() = Word::fixnum(value).raw();
            *p.as_ptr().add(1) = Word::NIL.raw();
        }
        Word::from_ptr(p.as_ptr() as *const u8, Tag::Cons)
    }

    #[test]
    fn fresh_heap_counters_are_zero() {
        let h = small_heap();
        assert_eq!(h.minors_since_g0_promote(), 0);
        assert_eq!(h.g0_promotes_since_g1_promote(), 0);
    }

    #[test]
    fn nursery_transients_die_in_g0_within_threshold_cycles() {
        // Acceptance test from the design doc. Allocate 50
        // unreachable cons cells; over `G0_PROMOTION_THRESHOLD - 1`
        // minor cycles, the heap should reclaim them entirely and
        // NOTHING should reach G1.
        let mut h = small_heap();
        for i in 0..50 {
            let _ = one_cons(&mut h, i);
        }
        assert!(h.count_pages_in_gen(Generation::G0) >= 1);

        for cycle in 1..G0_PROMOTION_THRESHOLD {
            let result = h.collect_minor(|_| { /* no roots */ });
            assert!(!result.promoted_g0, "cycle {cycle} should not promote");
            assert_eq!(
                result.minors_since_g0_promote_after, cycle,
                "counter must tick once per cycle"
            );
            // Garbage from earlier cycles is collected immediately;
            // by the second cycle G0 has nothing in it.
            assert_eq!(
                h.count_pages_in_gen(Generation::G0),
                0,
                "cycle {cycle}: unrooted conses reclaimed"
            );
            assert_eq!(
                h.count_pages_in_gen(Generation::G1),
                0,
                "cycle {cycle}: no G1 pages — nothing was rooted, nothing was promoted"
            );
        }
    }

    #[test]
    fn rooted_survivor_promotes_after_threshold_cycles() {
        // Companion to the above: a rooted cons must stay in G0
        // for `G0_PROMOTION_THRESHOLD - 1` cycles and promote
        // on the threshold cycle.
        let mut h = small_heap();
        let mut root = [one_cons(&mut h, 42)];

        // Pre-threshold cycles: stays in G0.
        for cycle in 1..G0_PROMOTION_THRESHOLD {
            let result =
                h.collect_minor(|evac| evac.visit(&mut root[0]));
            assert!(!result.promoted_g0, "cycle {cycle}: not yet");
            let addr =
                (root[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
            let page = h.page_of(addr).unwrap();
            assert_eq!(
                h.desc(page).generation,
                Generation::G0,
                "cycle {cycle}: rooted cons still in G0"
            );
            assert_eq!(
                Word::from_raw(unsafe { *(addr as *const u64) })
                    .as_fixnum(),
                Some(42),
                "value preserved"
            );
        }

        // Threshold cycle: promotes to G1.
        let result =
            h.collect_minor(|evac| evac.visit(&mut root[0]));
        assert!(result.promoted_g0, "threshold cycle promotes");
        assert_eq!(
            result.minors_since_g0_promote_after, 0,
            "counter resets after promotion"
        );
        let addr =
            (root[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        let page = h.page_of(addr).unwrap();
        assert_eq!(
            h.desc(page).generation,
            Generation::G1,
            "rooted cons promoted to G1"
        );
        // Value still intact.
        assert_eq!(
            Word::from_raw(unsafe { *(addr as *const u64) }).as_fixnum(),
            Some(42)
        );
    }

    #[test]
    fn promotion_resets_counter_and_starts_new_cohort() {
        // After a promotion cycle, the counter goes to 0 and the
        // next 3 cycles tick 1, 2, 3 with another promotion at 3.
        let mut h = small_heap();
        let mut root = [one_cons(&mut h, 1)];

        // Run THRESHOLD cycles to promote.
        for _ in 0..G0_PROMOTION_THRESHOLD {
            h.collect_minor(|evac| evac.visit(&mut root[0]));
        }
        assert_eq!(h.minors_since_g0_promote(), 0);

        // Allocate a fresh G0 cons, root both.
        let mut root2 = [one_cons(&mut h, 2)];
        for cycle in 1..G0_PROMOTION_THRESHOLD {
            let result = h.collect_minor(|evac| {
                evac.visit(&mut root[0]);
                evac.visit(&mut root2[0]);
            });
            assert!(!result.promoted_g0, "cycle {cycle} of cohort 2");
        }
        // root2 should still be in G0; root in G1 (didn't go further
        // because no G1 cycles in this short test).
        let addr2 =
            (root2[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        let page2 = h.page_of(addr2).unwrap();
        assert_eq!(
            h.desc(page2).generation,
            Generation::G0,
            "second-cohort cons still in G0 before threshold"
        );

        // Threshold cycle for cohort 2: promotes.
        let result = h.collect_minor(|evac| {
            evac.visit(&mut root[0]);
            evac.visit(&mut root2[0]);
        });
        assert!(result.promoted_g0);
        let addr2 =
            (root2[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        assert_eq!(
            h.desc(h.page_of(addr2).unwrap()).generation,
            Generation::G1
        );
    }

    #[test]
    fn g1_promotes_to_tenured_after_threshold_g0_promotions() {
        // Run G0_PROMOTION_THRESHOLD × G1_PROMOTION_THRESHOLD
        // minor cycles. After the last cycle, the rooted object
        // should be in Tenured.
        let mut h = small_heap();
        let mut root = [one_cons(&mut h, 99)];

        let total_minors = G0_PROMOTION_THRESHOLD * G1_PROMOTION_THRESHOLD;
        let mut last_result = None;
        for _ in 0..total_minors {
            last_result =
                Some(h.collect_minor(|evac| evac.visit(&mut root[0])));
        }
        let r = last_result.unwrap();
        assert!(r.promoted_g0, "final cycle promoted G0");
        assert!(
            r.promoted_g1,
            "final cycle also cascaded G1 → Tenured"
        );
        let addr =
            (root[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        let page = h.page_of(addr).unwrap();
        assert_eq!(
            h.desc(page).generation,
            Generation::Tenured,
            "long-lived cons reached Tenured"
        );
        assert_eq!(
            Word::from_raw(unsafe { *(addr as *const u64) }).as_fixnum(),
            Some(99),
            "value preserved through 15 cycles"
        );
    }

    #[test]
    fn major_promotes_g1_and_collects_g0() {
        // Set up state: one rooted cons in G0, one rooted cons
        // already in G1 (achieved by running threshold minors).
        let mut h = small_heap();
        let mut old_root = [one_cons(&mut h, 100)];
        for _ in 0..G0_PROMOTION_THRESHOLD {
            h.collect_minor(|evac| evac.visit(&mut old_root[0]));
        }
        // old_root now in G1.
        let g1_addr =
            (old_root[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        assert_eq!(
            h.desc(h.page_of(g1_addr).unwrap()).generation,
            Generation::G1
        );

        // Allocate a fresh G0 cons, root it.
        let mut new_root = [one_cons(&mut h, 200)];

        // Major GC: G1 → Tenured, G0 → G0.
        let result = h.collect_major(|evac| {
            evac.visit(&mut old_root[0]);
            evac.visit(&mut new_root[0]);
        });
        assert!(
            result.promoted_g1,
            "major reports G1 → Tenured promotion"
        );

        // old_root now in Tenured.
        let t_addr =
            (old_root[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        assert_eq!(
            h.desc(h.page_of(t_addr).unwrap()).generation,
            Generation::Tenured,
            "G1 contents promoted to Tenured"
        );
        // new_root still in G0 (major does G0→G0).
        let g_addr =
            (new_root[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        assert_eq!(
            h.desc(h.page_of(g_addr).unwrap()).generation,
            Generation::G0,
            "G0 contents stay in G0 after major"
        );
        // Both counters reset.
        assert_eq!(h.minors_since_g0_promote(), 0);
        assert_eq!(h.g0_promotes_since_g1_promote(), 0);
    }

    #[test]
    fn major_on_empty_heap_is_a_noop() {
        // Regression: a major cycle on a fresh heap (no G0, no
        // G1, no roots) must not panic and must report no work.
        let mut h = small_heap();
        let result = h.collect_major(|_| { /* no roots */ });
        assert_eq!(result.evac.objects_copied, 0);
        let cas = result.cascade.expect("major always returns cascade");
        assert_eq!(cas.objects_copied, 0);
        assert!(
            !result.promoted_g1,
            "no G1 data → promoted_g1 should be false"
        );
        assert_eq!(h.minors_since_g0_promote(), 0);
        assert_eq!(h.g0_promotes_since_g1_promote(), 0);
    }

    #[test]
    fn cascade_reports_nonzero_objects_copied_when_g1_has_data() {
        // Regression for review comment: `CollectResult.cascade`
        // should report real work done by the G1 → Tenured pass,
        // not just a phantom Some(EvacResult::default()).
        let mut h = small_heap();
        let mut root = [one_cons(&mut h, 7)];

        // Run enough cycles to fire a cascade
        // (G0_PROMOTION_THRESHOLD × G1_PROMOTION_THRESHOLD).
        let total = G0_PROMOTION_THRESHOLD * G1_PROMOTION_THRESHOLD;
        let mut final_result = None;
        for _ in 0..total {
            final_result =
                Some(h.collect_minor(|evac| evac.visit(&mut root[0])));
        }
        let r = final_result.unwrap();
        assert!(r.promoted_g0);
        assert!(r.promoted_g1, "cascade fired");
        let cas = r.cascade.expect("cascading minor reports cascade");
        assert!(
            cas.objects_copied >= 1,
            "G1 → Tenured copied at least the rooted cons; got {cas:?}"
        );
    }

    #[test]
    fn unrooted_g1_cons_reclaimed_on_cascade() {
        // Place two conses in G1 (one rooted, one not). When the
        // G1 → Tenured cascade fires, the rooted one survives to
        // Tenured and the unrooted one's page is reclaimed.
        let mut h = small_heap();
        let mut rooted = [one_cons(&mut h, 1)];
        let mut transient = [one_cons(&mut h, 2)];

        // Promote both to G1 (cohort cycle).
        for _ in 0..G0_PROMOTION_THRESHOLD {
            h.collect_minor(|evac| {
                evac.visit(&mut rooted[0]);
                evac.visit(&mut transient[0]);
            });
        }
        let rooted_g1_addr =
            (rooted[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        let transient_g1_addr =
            (transient[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        assert_eq!(
            h.desc(h.page_of(rooted_g1_addr).unwrap()).generation,
            Generation::G1
        );
        assert_eq!(
            h.desc(h.page_of(transient_g1_addr).unwrap()).generation,
            Generation::G1
        );
        let g1_pages_before = h.count_pages_in_gen(Generation::G1);

        // Run more minors until cascade. Visit ONLY `rooted` —
        // the transient becomes unreachable.
        let remaining =
            (G1_PROMOTION_THRESHOLD - 1) * G0_PROMOTION_THRESHOLD;
        for _ in 0..remaining {
            h.collect_minor(|evac| evac.visit(&mut rooted[0]));
        }
        // Now should be in Tenured.
        let rooted_t_addr =
            (rooted[0].raw() & crate::word::PAYLOAD_MASK) as *const u8;
        assert_eq!(
            h.desc(h.page_of(rooted_t_addr).unwrap()).generation,
            Generation::Tenured,
            "rooted cons in Tenured after cascade"
        );
        // G1 should be empty — transient was unrooted and got
        // reclaimed during the cascade.
        assert_eq!(
            h.count_pages_in_gen(Generation::G1),
            0,
            "unrooted G1 cons collected during G1 → Tenured pass"
        );
        let _ = g1_pages_before;
    }

    #[test]
    fn many_cycles_dont_overflow_counter() {
        // saturating_add guard: even if the user runs an absurd
        // number of cycles with no roots, the counter shouldn't
        // wrap around to 0 and accidentally skip a promotion.
        // (For unrooted heap, every cycle frees everything, so
        // promotion is a no-op semantically — but the counter
        // still needs to be well-behaved.)
        let mut h = small_heap();
        for _ in 0..100 {
            h.collect_minor(|_| { /* no roots */ });
        }
        // 100 / 3 = 33 promotions; counter sits at 100 mod 3 = 1.
        assert_eq!(h.minors_since_g0_promote(), 100 % G0_PROMOTION_THRESHOLD);
    }
}
