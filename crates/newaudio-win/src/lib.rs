//! Windows runtime for NewAudio.
//!
//! # Overview
//!
//! `newaudio-win` is the platform-specific half of NewAudio. It binds
//! the Win32 multimedia subsystem (`winmm.dll`) and exposes:
//!
//! - [`PcmRuntime`]: background `waveOut` mixer with looping voices,
//!   fade-out, per-voice volume/pan, volume jitter, variations, and an
//!   independent SFX bus gain.
//! - [`MidiRuntime`]: scheduled `midiOut` player that consumes
//!   `newaudio_abc::AbcTune`s, allocates MIDI channels, dispatches
//!   short messages with sub-millisecond cadence, and supports
//!   pause/resume plus a music bus gain.
//! - [`Mixer`]: top-level facade that owns both runtimes and keeps the
//!   master volume in sync.
//! - [`AudioThread`]: non-blocking command queue + worker that drains
//!   [`AudioCommand`]s into a `PcmRuntime`. Use this from game-loop hot
//!   paths to avoid taking the runtime mutex per call.
//! - [`SoundPack`]: typed enum-keyed sound library with built-in
//!   variation support.
//!
//! # Typical use
//!
//! ```no_run
//! use std::sync::Arc;
//! use newaudio_core::{Engine, Config};
//! use newaudio_win::{AudioThread, Mixer, SoundPack};
//!
//! let mixer = Mixer::start().unwrap();
//! mixer.set_master_volume(0.8);
//! mixer.set_sfx_bus_volume(1.0);
//! mixer.set_music_bus_volume(0.4);
//!
//! #[derive(Copy, Clone, Eq, PartialEq, Hash)]
//! enum Sfx { Coin, Jump }
//! let mut eng = Engine::new(Config::default());
//! let pack = SoundPack::builder(mixer.pcm())
//!     .insert(Sfx::Coin, eng.coin(1.0, 0.4))
//!     .insert(Sfx::Jump, eng.jump(1.0, 0.3))
//!     .build();
//!
//! let audio = AudioThread::spawn(mixer.pcm());
//! audio.play_simple(1);              // non-blocking
//! pack.play(Sfx::Coin);              // direct, takes runtime mutex
//! ```
//!
//! # Thread safety
//!
//! `PcmRuntime`, `MidiRuntime`, and `Mixer` are `Send + Sync`. Wrap in
//! `Arc` to share across game threads — see [`AudioThread`] for the
//! non-blocking submission pattern.

#![cfg(windows)]
#![deny(unsafe_op_in_unsafe_fn)]

mod audio_thread;
mod midi_runtime;
mod mixer;
mod pcm_runtime;
mod scheduling;
mod sound_pack;

pub use audio_thread::{AudioCommand, AudioThread};
pub use midi_runtime::{MidiAssetId, MidiPlayback, MidiRuntime, PlaybackState};
pub use mixer::Mixer;
pub use pcm_runtime::{PcmRuntime, PlayOptions, SoundId, VoiceHandle};
pub use scheduling::{ScheduledEvent, compile_asset};
pub use sound_pack::{SoundKey, SoundPack, SoundPackBuilder};
