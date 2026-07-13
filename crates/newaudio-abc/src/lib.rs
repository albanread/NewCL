//! ABC notation parser and MIDI generator/writer.
//!
//! # Overview
//!
//! `newaudio-abc` turns [ABC notation] strings into Standard MIDI File
//! bytes. The pipeline is:
//!
//! ```text
//!   "X:1\nK:C\nCDEF|"  →  AbcParser  →  AbcTune
//!                                          │
//!                                          ▼
//!                              MidiGenerator  →  Vec<MidiTrack>
//!                                                       │
//!                                                       ▼
//!                                               write_smf  →  Vec<u8>
//! ```
//!
//! The whole flow is wrapped by the convenience function
//! [`abc_to_smf`]:
//!
//! ```
//! use newaudio_abc::abc_to_smf;
//!
//! let abc = "X:1\nT:Demo\nM:4/4\nL:1/4\nQ:120\nK:C\nCDEF|GAGE|";
//! let (_tune, smf_bytes) = abc_to_smf(abc).unwrap();
//! assert_eq!(&smf_bytes[0..4], b"MThd");
//! ```
//!
//! # Supported ABC
//!
//! Standard headers (`X`, `T`, `C`, `M`, `L`, `Q`, `K`, `V`), single and
//! multi-voice tunes, accidentals with within-bar persistence, durations
//! (`C2`, `C/2`, dotted), broken rhythm (`A>B`/`A<B`), rests, chords
//! (`[CEG]`), guitar chords (`"Cm7"`), ties, slurs, tuplets, grace
//! groups, repeat bars (`|: … :|`), and `%%MIDI` directives.
//!
//! # Quirks preserved from the original Zig parser
//!
//! - Uppercase `C` is MIDI 48 (octave 3), not middle C. Lowercase `c` is
//!   middle C (MIDI 60).
//! - `|`, `[`, `]`, `:` characters tokenise as bar lines as a single
//!   run. As a result, `|[Q:60]` is *not* a tempo change — the `[`
//!   gets consumed by the bar parser. Place inline directives on their
//!   own music line to make them effective.
//!
//! Both behaviours are guaranteed to match the upstream Zig
//! implementation byte-for-byte (see `tests/zig_cross_check.rs` in the
//! workspace).
//!
//! [ABC notation]: https://en.wikipedia.org/wiki/ABC_notation

#![deny(unsafe_op_in_unsafe_fn)]

pub mod fraction;
pub mod midi;
pub mod parser;
pub mod repeat;
pub mod types;

pub use fraction::Fraction;
pub use midi::{EventKind, MidiEvent, MidiGenerator, MidiTrack, TrackType, write_smf, write_var_len};
pub use parser::AbcParser;
pub use repeat::expand_abc_repeats;
pub use types::*;

/// Parse an ABC string and produce the MIDI file bytes in one step.
///
/// Convenience for the common case. Returns the file bytes alongside the
/// parsed tune so callers can also inspect structure (e.g. tempo for UI).
///
/// ```
/// let (tune, bytes) = newaudio_abc::abc_to_smf("X:1\nK:C\nCDEF|").unwrap();
/// assert_eq!(&bytes[..4], b"MThd");
/// assert_eq!(tune.voices.len(), 1);
/// ```
pub fn abc_to_smf(abc: &str) -> Result<(AbcTune, Vec<u8>), String> {
    let mut parser = AbcParser::new();
    let tune = parser.parse(abc)?;
    let mut mgen = MidiGenerator::new();
    let mut tracks = mgen.generate(&tune);
    let bytes = write_smf(&mut tracks, mgen.ticks_per_quarter);
    Ok((tune, bytes))
}
