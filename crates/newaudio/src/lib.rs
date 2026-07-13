//! NewAudio — game-focused sound synthesis and ABC notation playback for Windows.
//!
//! This is the umbrella facade: it re-exports every public type from the
//! three internal crates so you only need one dependency in your
//! `Cargo.toml`. See the workspace's `USER_GUIDE.md` for a tour.
//!
//! - [`newaudio_core`] — synthesis, ADSR, presets, WAV reader/writer,
//!   effect helpers, 2D positional helpers.
//! - [`newaudio_abc`] — ABC parser, MIDI generator, SMF writer.
//! - [`newaudio_win`] (Windows only) — `waveOut` mixer, `midiOut`
//!   scheduler, top-level [`Mixer`], non-blocking [`AudioThread`],
//!   typed [`SoundPack`].
//!
//! # Quick start
//!
//! ```no_run
//! # #[cfg(windows)] {
//! use newaudio::*;
//!
//! // Render a coin SFX.
//! let mut eng = Engine::new(Config::default());
//! let coin = eng.coin(1.0, 0.4);
//!
//! // Play it through the runtime.
//! let mixer = Mixer::start().unwrap();
//! let id = mixer.pcm().register_sound(coin);
//! mixer.pcm().play_simple(id);
//!
//! // Parse + play an ABC tune.
//! let tune = AbcParser::new().parse("X:1\nK:C\nCDEF|").unwrap();
//! let mid = mixer.midi().load(&tune);
//! mixer.midi().play_simple(mid);
//! # }
//! ```
//!
//! # See also
//!
//! - `USER_GUIDE.md` in the workspace root — task-oriented walkthrough.
//! - `README.md` — feature summary and status.

pub use newaudio_core::*;
pub use newaudio_abc::*;

#[cfg(windows)]
pub use newaudio_win::*;
