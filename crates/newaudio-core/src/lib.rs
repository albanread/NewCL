//! Sample-rate-agnostic sound synthesis primitives for NewAudio.
//!
//! # Overview
//!
//! `newaudio-core` is the platform-independent half of NewAudio. It
//! contains:
//!
//! - The synthesis [`Engine`] and its building blocks ([`Effect`],
//!   [`Oscillator`], [`Adsr`], [`Waveform`]).
//! - Stock [`presets`] for game sound effects (`beep`, `coin`, `jump`,
//!   `explode`, `zap`, `hurt`, `shoot`, …) accessed as methods on
//!   [`Engine`].
//! - Source-parity builders in [`synth`] (`tone`, `midi_note`, `noise`,
//!   `fm`, plus `filtered_*` / `reverb_*` / `delay_*` / `distortion_*`).
//! - In-place [`effects`] (filter, echo, distortion).
//! - 2D positional helpers in [`spatial`].
//! - A 16-bit PCM [`wav`] reader/writer.
//!
//! Nothing here depends on Windows; this crate compiles and tests on
//! any target.
//!
//! # Typical use
//!
//! ```
//! use newaudio_core::{Engine, Config, WavParams, write_wav};
//!
//! let mut eng = Engine::new(Config::default());
//! let buffer = eng.coin(1.0, 0.4);
//!
//! let mut out = Vec::new();
//! write_wav(&mut out, &buffer, WavParams::default()).unwrap();
//! assert!(out.starts_with(b"RIFF"));
//! ```
//!
//! # Determinism
//!
//! Every preset is deterministic given a fixed [`Config`] and seed.
//! Construct via [`Engine::with_seed`] when you want byte-identical
//! output across runs — the WAV/MIDI golden tests rely on this.
//!
//! # See also
//!
//! - `newaudio-abc` — ABC notation parser and MIDI generator.
//! - `newaudio-win` (Windows only) — live `waveOut` mixer and `midiOut`
//!   scheduler that consume `Buffer`s and `AbcTune`s from this crate.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod effects;
pub mod engine;
pub mod envelope;
pub mod presets;
pub mod spatial;
pub mod synth;
pub mod wav;
pub mod waveform;

pub use effects::{FilterType, apply_distortion, apply_echo, apply_filter};
pub use engine::{
    Buffer, Config, Effect, Engine, Oscillator, frequency_to_note, normalize, note_to_frequency,
};
pub use envelope::Adsr;
pub use spatial::{PanResult, pan_from_position, pan_from_position_2d};
pub use synth::NoiseType;
pub use waveform::{Lcg, Waveform};
pub use wav::{
    WavParams, WavReadError, read_wav, read_wav_file, write_wav, write_wav_file,
};
