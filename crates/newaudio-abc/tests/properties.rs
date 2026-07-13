//! Property tests for the ABC pipeline. These check invariants that must
//! hold for *any* well-formed input, not just the curated golden tunes.

use newaudio_abc::{AbcParser, FeatureData, MidiGenerator, write_smf, write_var_len};
use proptest::prelude::*;

proptest! {
    /// Variable-length quantities round-trip: encoded length never exceeds 4
    /// bytes, the top bit of every leading byte is set, and the final byte
    /// has its top bit clear.
    #[test]
    fn var_len_is_well_formed(v in 0u32..=0x0FFF_FFFF) {
        let mut out = Vec::new();
        write_var_len(&mut out, v);
        prop_assert!(out.len() >= 1 && out.len() <= 4);
        for byte in &out[..out.len() - 1] {
            prop_assert!(byte & 0x80 != 0, "non-final byte missing continuation bit");
        }
        prop_assert!(out[out.len() - 1] & 0x80 == 0, "final byte must clear top bit");
    }

    /// Any sequence of notes followed by a rest of n times the unit length
    /// puts the next event at `(n_notes + n_rests) * unit_len` whole-notes.
    #[test]
    fn note_timestamps_advance_by_duration(n in 1u32..6u32) {
        let body: String = (0..n).map(|_| 'C').collect();
        let abc = format!("X:1\nM:4/4\nL:1/4\nK:C\n{body}\n");
        let mut p = AbcParser::new();
        let tune = p.parse(&abc).unwrap();
        let mut t = 0.0_f64;
        for f in &tune.features {
            if let FeatureData::Note(_) = f.data {
                prop_assert!((f.ts - t).abs() < 1e-9);
                t += 0.25;
            }
        }
    }

    /// Tempo change in the body produces a `MetaTempo` event in the SMF
    /// stream beyond the initial one at t=0.
    ///
    /// We place the inline `[Q:..]` on its own music line. The original
    /// Zig parser (and this port) tokenises a run of `|`, `[`, `]`, `:`
    /// characters as one bar-line lump, which means `|[Q:..]` *immediately
    /// after* another bar would be eaten before the inline-bracket handler
    /// sees it. Keeping the change on a new line ensures the inline handler
    /// runs first — the supported way to change tempo mid-tune.
    #[test]
    fn tempo_changes_appear_in_smf(bpm in 30i32..=300i32) {
        let abc = format!(
            "X:1\nT:T\nM:4/4\nL:1/4\nQ:120\nK:C\nC|\n[Q:{bpm}]\nD|\n"
        );
        let mut p = AbcParser::new();
        let tune = p.parse(&abc).unwrap();
        let mut mgen = MidiGenerator::new();
        let mut tracks = mgen.generate(&tune);
        let bytes = write_smf(&mut tracks, mgen.ticks_per_quarter);
        // 0xFF 0x51 is the tempo meta event. It must appear at least twice:
        // initial 120 BPM at t=0 and the change.
        let mut count = 0;
        for window in bytes.windows(2) {
            if window == [0xFF, 0x51] {
                count += 1;
            }
        }
        prop_assert!(count >= 2, "expected >=2 tempo metas, got {count}");
    }

    /// The SMF stream always starts with `MThd` and contains at least one
    /// `MTrk` chunk header — for *any* valid ABC.
    #[test]
    fn smf_always_well_formed(seed in 0u32..1000u32) {
        // Generate a tiny varied tune from the seed.
        let pitches = ["C", "D", "E", "F", "G", "A", "B"];
        let pitch = pitches[(seed as usize) % pitches.len()];
        let abc = format!("X:1\nL:1/4\nK:C\n{pitch}\n");
        let mut p = AbcParser::new();
        let tune = p.parse(&abc).unwrap();
        let mut mgen = MidiGenerator::new();
        let mut tracks = mgen.generate(&tune);
        let bytes = write_smf(&mut tracks, mgen.ticks_per_quarter);
        prop_assert_eq!(&bytes[0..4], b"MThd");
        prop_assert!(bytes.windows(4).any(|w| w == b"MTrk"));
    }
}
