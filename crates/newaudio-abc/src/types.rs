//! Parsed ABC document types. Names mirror the Zig source.

use std::collections::BTreeMap;

use crate::fraction::Fraction;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tempo {
    pub bpm: i32,
}

impl Default for Tempo {
    fn default() -> Self {
        Self { bpm: 120 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimeSig {
    pub num: u8,
    pub denom: u8,
}

impl Default for TimeSig {
    fn default() -> Self {
        Self { num: 4, denom: 4 }
    }
}

/// Sharps count (positive) or flats count (negative), plus mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeySig {
    pub sharps: i8,
    pub is_major: bool,
}

impl Default for KeySig {
    fn default() -> Self {
        Self {
            sharps: 0,
            is_major: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceContext {
    pub id: i32,
    pub name: String,
    pub key: KeySig,
    pub timesig: TimeSig,
    pub unit_len: Fraction,
    pub transpose: i8,
    pub octave_shift: i8,
    pub instrument: u8,
    pub channel: i8,
    pub velocity: u8,
    pub percussion: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Note {
    pub pitch: u8,
    pub accidental: i8,
    pub octave: i8,
    pub duration: Fraction,
    pub midi_note: u8,
    pub velocity: u8,
    pub is_tied: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rest {
    pub duration: Fraction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chord {
    pub notes: Vec<Note>,
    pub duration: Fraction,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GuitarChord {
    pub symbol: String,
    pub root_note: u8,
    pub chord_type: String,
    pub duration: Fraction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarType {
    Bar1,
    DoubleBar,
    RepStart,
    RepEnd,
    DoubleRep,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BarLine {
    pub bar_type: BarType,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VoiceChange {
    pub voice_number: i32,
    pub voice_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureData {
    Note(Note),
    Rest(Rest),
    Chord(Chord),
    GChord(GuitarChord),
    Bar(BarLine),
    Tempo(Tempo),
    Time(TimeSig),
    Key(KeySig),
    Voice(VoiceChange),
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub voice_id: i32,
    /// Timestamp in whole-note units (matches Zig semantics).
    pub ts: f64,
    pub line_number: usize,
    pub data: FeatureData,
}

#[derive(Debug, Clone, Default)]
pub struct AbcTune {
    pub title: String,
    pub history: String,
    pub composer: String,
    pub origin: String,
    pub rhythm: String,
    pub notes: String,
    pub words: String,
    pub aligned_words: String,
    pub default_key: KeySig,
    pub default_timesig: TimeSig,
    pub default_unit: Fraction,
    pub default_tempo: Tempo,
    pub default_instrument: u8,
    pub default_channel: i8,
    pub default_percussion: bool,
    /// Ordered map: insertion order matches voice creation order so that
    /// MIDI track output is deterministic across runs.
    pub voices: BTreeMap<i32, VoiceContext>,
    pub features: Vec<Feature>,
}

impl AbcTune {
    pub fn new() -> Self {
        Self {
            default_unit: Fraction::new(1, 8),
            default_tempo: Tempo { bpm: 120 },
            default_timesig: TimeSig { num: 4, denom: 4 },
            default_channel: -1,
            ..Default::default()
        }
    }
}
