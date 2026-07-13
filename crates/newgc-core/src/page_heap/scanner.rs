//! Root scanner adapter for the page heap.
//!
//! Matches the surface of NCL's semispace `RootScanner` (upstream) so
//! coordinator-side closures can write
//! `&mut RootScanner<'_, '_>` once and have it resolve to the right
//! type under whichever Cargo feature is active (`gc::RootScanner`
//! re-exports the active backend's scanner).
//!
//! The page-heap implementation is a thin wrapper over
//! `PageEvacuator`; `visit(&mut Word)` delegates to
//! `PageEvacuator::visit`.

use crate::traits::HeapLayout;
use crate::word::Word;

use super::evac::PageEvacuator;

/// Page-heap root scanner. The two lifetimes mirror the semispace
/// shape:
///   - `'s`: lifetime of the outer borrow (how long the scanner
///     itself is alive — typically the closure body).
///   - `'a`: lifetime of the evacuator's borrow on the heap.
pub struct RootScanner<'s, 'a: 's, L: HeapLayout> {
    evac: &'s mut PageEvacuator<'a, L>,
}

impl<'s, 'a: 's, L: HeapLayout> RootScanner<'s, 'a, L> {
    /// Construct a scanner targeting the given evacuator. The
    /// scanner exists only for the duration of a `visit_roots`
    /// callback; when the callback returns and the scanner drops,
    /// the evacuator becomes usable again for the BFS drain.
    pub fn new(evac: &'s mut PageEvacuator<'a, L>) -> Self {
        RootScanner { evac }
    }

    /// Visit a root slot. Same contract as the semispace scanner:
    /// reads `*slot`, possibly evacuates the referenced object,
    /// and rewrites `*slot` with the post-evac Word.
    pub fn visit(&mut self, slot: &mut Word) {
        self.evac.visit(slot);
    }
}
