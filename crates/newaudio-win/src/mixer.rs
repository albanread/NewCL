//! Top-level mixer that owns both the PCM (waveOut) and MIDI (midiOut)
//! runtimes and exposes a single master/sfx/music bus surface.
//!
//! `master` is multiplied into both runtimes' bus gains. The intended
//! pattern:
//!
//! ```no_run
//! # #[cfg(windows)] {
//! use newaudio_win::Mixer;
//! let mixer = Mixer::start().unwrap();
//! mixer.set_master_volume(0.8);
//! mixer.set_sfx_bus_volume(1.0);
//! mixer.set_music_bus_volume(0.5);
//! # }
//! ```
//!
//! Use [`Mixer::pcm`] / [`Mixer::midi`] to get `Arc` handles for sharing
//! across threads. Drop the `Mixer` to shut both runtimes down.

use std::sync::Arc;

use crate::midi_runtime::MidiRuntime;
use crate::pcm_runtime::PcmRuntime;

/// Combined PCM + MIDI runtime with three independent bus volumes.
pub struct Mixer {
    pcm: Arc<PcmRuntime>,
    midi: Arc<MidiRuntime>,
}

impl Mixer {
    /// Start both runtimes. Returns `Err` if either `waveOutOpen` or
    /// `midiOutOpen` fails (e.g. on a headless system with no audio
    /// devices). Both handles are kept until the `Mixer` is dropped.
    pub fn start() -> Result<Self, String> {
        let pcm = Arc::new(PcmRuntime::start()?);
        let midi = Arc::new(MidiRuntime::start()?);
        Ok(Self { pcm, midi })
    }

    /// Cheap clones of the inner runtimes for distributing to game threads.
    pub fn pcm(&self) -> Arc<PcmRuntime> {
        Arc::clone(&self.pcm)
    }

    pub fn midi(&self) -> Arc<MidiRuntime> {
        Arc::clone(&self.midi)
    }

    /// Master gain applied on top of both bus volumes. Range `[0, 1]`.
    pub fn set_master_volume(&self, v: f32) {
        self.pcm.set_master_volume(v);
        self.midi.set_master_volume(v);
    }

    /// Returns the master volume currently in effect (the two runtimes
    /// are kept in sync so reading either is sufficient).
    pub fn master_volume(&self) -> f32 {
        self.pcm.master_volume()
    }

    /// PCM/sfx bus volume. Multiplied with master before reaching voices.
    pub fn set_sfx_bus_volume(&self, v: f32) {
        self.pcm.set_sfx_bus_volume(v);
    }

    pub fn sfx_bus_volume(&self) -> f32 {
        self.pcm.sfx_bus_volume()
    }

    /// MIDI/music bus volume.
    pub fn set_music_bus_volume(&self, v: f32) {
        self.midi.set_music_bus_volume(v);
    }

    pub fn music_bus_volume(&self) -> f32 {
        self.midi.music_bus_volume()
    }

    /// Stop all PCM voices and active MIDI playbacks.
    pub fn stop_all(&self) {
        self.pcm.stop_all();
        self.midi.stop_all();
    }
}
