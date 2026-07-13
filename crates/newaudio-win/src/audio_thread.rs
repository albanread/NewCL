//! Non-blocking audio submission.
//!
//! Wraps an [`Arc<PcmRuntime>`] behind a single producer/consumer channel.
//! Game code calls [`AudioThread::play`] (and friends) which only enqueue
//! a command and return immediately — handy for hot loops where you don't
//! want to take a mutex per call. The worker thread drains the queue and
//! forwards commands to the runtime.
//!
//! ```no_run
//! # #[cfg(windows)] {
//! use std::sync::Arc;
//! use newaudio_win::{AudioThread, Mixer};
//! let mixer = Mixer::start().unwrap();
//! let audio = AudioThread::spawn(mixer.pcm());
//!
//! // In game loop:
//! audio.play(42, 1.0, 0.0);            // never blocks
//! audio.set_master_volume(0.7);
//! # }
//! ```

use std::sync::Arc;
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};

use crate::pcm_runtime::{PcmRuntime, PlayOptions, SoundId, VoiceHandle};

/// Operations the audio thread can perform.
///
/// All variants are `Copy` or `Clone` and contain no allocations, so
/// pushing one into the queue is allocation-free in the steady state.
#[derive(Debug, Clone)]
pub enum AudioCommand {
    Play {
        id: SoundId,
        volume: f32,
        pan: f32,
    },
    PlayWith {
        id: SoundId,
        opts: PlayOptions,
    },
    PlayLooped {
        id: SoundId,
        volume: f32,
        pan: f32,
    },
    PlayRandom {
        ids: Vec<SoundId>,
        volume: f32,
        pan: f32,
    },
    StopVoice {
        handle: VoiceHandle,
        fade_out_secs: Option<f32>,
    },
    StopSound {
        id: SoundId,
    },
    StopAll,
    FreeSound {
        id: SoundId,
    },
    FreeAll,
    SetMasterVolume(f32),
    SetSfxBusVolume(f32),
    /// Synchronisation barrier: nothing is done with the value, but
    /// callers can use `Shutdown` to know the worker has reached this
    /// point in the queue (everything queued before it has been applied).
    Noop,
    Shutdown,
}

/// Owned audio worker. Drop to stop the thread cleanly.
pub struct AudioThread {
    sender: Option<Sender<AudioCommand>>,
    worker: Option<JoinHandle<()>>,
    runtime: Arc<PcmRuntime>,
}

impl AudioThread {
    /// Spawn a worker thread that processes commands against `runtime`.
    pub fn spawn(runtime: Arc<PcmRuntime>) -> Self {
        let (tx, rx) = mpsc::channel::<AudioCommand>();
        let worker_rt = Arc::clone(&runtime);
        let worker = thread::spawn(move || {
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    AudioCommand::Play { id, volume, pan } => {
                        let _ = worker_rt.play(id, volume, pan);
                    }
                    AudioCommand::PlayWith { id, opts } => {
                        let _ = worker_rt.play_with(id, opts);
                    }
                    AudioCommand::PlayLooped { id, volume, pan } => {
                        let _ = worker_rt.play_looped(id, volume, pan);
                    }
                    AudioCommand::PlayRandom { ids, volume, pan } => {
                        let _ = worker_rt.play_random(&ids, volume, pan);
                    }
                    AudioCommand::StopVoice {
                        handle,
                        fade_out_secs,
                    } => worker_rt.stop_voice(handle, fade_out_secs),
                    AudioCommand::StopSound { id } => worker_rt.stop_sound(id),
                    AudioCommand::StopAll => worker_rt.stop_all(),
                    AudioCommand::FreeSound { id } => {
                        let _ = worker_rt.free_sound(id);
                    }
                    AudioCommand::FreeAll => worker_rt.free_all(),
                    AudioCommand::SetMasterVolume(v) => worker_rt.set_master_volume(v),
                    AudioCommand::SetSfxBusVolume(v) => worker_rt.set_sfx_bus_volume(v),
                    AudioCommand::Noop => {}
                    AudioCommand::Shutdown => break,
                }
            }
        });
        Self {
            sender: Some(tx),
            worker: Some(worker),
            runtime,
        }
    }

    /// Direct access to the underlying runtime for operations that need
    /// a return value (e.g. registering a sound). These do take the
    /// runtime's mutex and may block briefly — use them off the hot path.
    pub fn runtime(&self) -> &Arc<PcmRuntime> {
        &self.runtime
    }

    /// Enqueue a raw command.
    pub fn submit(&self, cmd: AudioCommand) {
        if let Some(tx) = self.sender.as_ref() {
            let _ = tx.send(cmd);
        }
    }

    pub fn play(&self, id: SoundId, volume: f32, pan: f32) {
        self.submit(AudioCommand::Play { id, volume, pan });
    }

    pub fn play_simple(&self, id: SoundId) {
        self.play(id, 1.0, 0.0);
    }

    pub fn play_with(&self, id: SoundId, opts: PlayOptions) {
        self.submit(AudioCommand::PlayWith { id, opts });
    }

    pub fn play_looped(&self, id: SoundId, volume: f32, pan: f32) {
        self.submit(AudioCommand::PlayLooped { id, volume, pan });
    }

    pub fn play_random(&self, ids: Vec<SoundId>, volume: f32, pan: f32) {
        self.submit(AudioCommand::PlayRandom { ids, volume, pan });
    }

    pub fn stop_voice(&self, handle: VoiceHandle, fade_out_secs: Option<f32>) {
        self.submit(AudioCommand::StopVoice {
            handle,
            fade_out_secs,
        });
    }

    pub fn stop_sound(&self, id: SoundId) {
        self.submit(AudioCommand::StopSound { id });
    }

    pub fn stop_all(&self) {
        self.submit(AudioCommand::StopAll);
    }

    pub fn set_master_volume(&self, v: f32) {
        self.submit(AudioCommand::SetMasterVolume(v));
    }

    pub fn set_sfx_bus_volume(&self, v: f32) {
        self.submit(AudioCommand::SetSfxBusVolume(v));
    }
}

impl Drop for AudioThread {
    fn drop(&mut self) {
        if let Some(tx) = self.sender.as_ref() {
            let _ = tx.send(AudioCommand::Shutdown);
        }
        // Drop the sender so the worker recv() loop terminates cleanly
        // even if the Shutdown send failed (e.g. receiver gone).
        self.sender.take();
        if let Some(j) = self.worker.take() {
            let _ = j.join();
        }
    }
}
