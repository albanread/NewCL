//! `midiOut`-backed scheduled MIDI player.
//!
//! Workflow:
//!  1. Caller passes an [`AbcTune`] to [`MidiRuntime::load`].
//!  2. The tune is compiled into a sorted list of [`ScheduledEvent`]s.
//!  3. [`MidiRuntime::play`] reserves hardware channels, then a background
//!     worker dispatches due events through `midiOutShortMsg`.

use std::collections::HashMap;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use newaudio_abc::AbcTune;
use windows_sys::Win32::Media::Audio::*;

use crate::scheduling::{ScheduledEvent, compile_asset};

/// `mmsystem.h` constant: success return code for `waveOut*` / `midiOut*`.
const MMSYSERR_NOERROR: u32 = 0;
/// `mmsystem.h` constant: ask the OS to pick a MIDI output device.
const MIDI_MAPPER: u32 = 0xFFFF_FFFF;

pub type MidiAssetId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackState {
    Stopped,
    Playing,
    Paused,
}

#[derive(Debug)]
struct StoredAsset {
    title: String,
    tempo_bpm: f32,
    used_channels_mask: u16,
    events: Vec<ScheduledEvent>,
}

#[derive(Debug)]
pub struct MidiPlayback {
    pub asset_id: MidiAssetId,
    pub state: PlaybackState,
}

#[derive(Debug)]
struct Playback {
    asset_id: MidiAssetId,
    volume: f32,
    state: PlaybackState,
    event_index: usize,
    /// Time the playback started, in monotonic-clock instants. While
    /// paused, [`Self::elapsed_at_pause`] holds the offset and start is
    /// recomputed on resume.
    start: Instant,
    elapsed_at_pause: Duration,
    channel_map: [u8; 16], // 0xFF = unmapped
}

/// Wrapper that makes the raw `HMIDIOUT` handle movable across threads.
/// Synchronisation is provided by the surrounding `Mutex`.
#[repr(transparent)]
struct MidiOutHandle(HMIDIOUT);
unsafe impl Send for MidiOutHandle {}

struct RuntimeShared {
    midi_out: Mutex<MidiOutHandle>,
    master_volume: Mutex<f32>,
    music_bus_volume: Mutex<f32>,
    next_asset_id: Mutex<MidiAssetId>,
    assets: Mutex<HashMap<MidiAssetId, StoredAsset>>,
    playbacks: Mutex<Vec<Playback>>,
    channels_in_use: Mutex<[bool; 16]>,
}

pub struct MidiRuntime {
    shared: Arc<RuntimeShared>,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl MidiRuntime {
    pub fn start() -> Result<Self, String> {
        let mut midi_out: HMIDIOUT = null_mut();
        let rc = unsafe { midiOutOpen(&mut midi_out as *mut _, MIDI_MAPPER, 0, 0, 0) };
        if rc != MMSYSERR_NOERROR {
            return Err(format!("midiOutOpen failed: {rc}"));
        }

        let shared = Arc::new(RuntimeShared {
            midi_out: Mutex::new(MidiOutHandle(midi_out)),
            master_volume: Mutex::new(1.0),
            music_bus_volume: Mutex::new(1.0),
            next_asset_id: Mutex::new(1),
            assets: Mutex::new(HashMap::new()),
            playbacks: Mutex::new(Vec::new()),
            channels_in_use: Mutex::new([false; 16]),
        });
        let shutdown = Arc::new(AtomicBool::new(false));

        let worker_shared = Arc::clone(&shared);
        let worker_shutdown = Arc::clone(&shutdown);
        let worker = thread::spawn(move || worker_main(worker_shared, worker_shutdown));

        Ok(Self {
            shared,
            shutdown,
            worker: Some(worker),
        })
    }

    /// Compile a parsed [`AbcTune`] into the runtime's asset bank. The
    /// returned id can be passed to [`Self::play`] one or more times.
    ///
    /// Compilation pre-computes a sorted list of MIDI short messages
    /// with absolute wall-clock timestamps (nanoseconds since playback
    /// start), so the worker thread doesn't have to re-derive timing
    /// at play time.
    pub fn load(&self, tune: &AbcTune) -> MidiAssetId {
        let (events, mask) = compile_asset(tune);
        let mut id_lock = self.shared.next_asset_id.lock().unwrap();
        let id = *id_lock;
        *id_lock += 1;
        drop(id_lock);
        let asset = StoredAsset {
            title: tune.title.clone(),
            tempo_bpm: tune.default_tempo.bpm as f32,
            used_channels_mask: mask,
            events,
        };
        self.shared.assets.lock().unwrap().insert(id, asset);
        id
    }

    /// Free an asset previously returned by [`Self::load`]. Stops any
    /// active playbacks of it first. Returns `true` if the id was known.
    pub fn free(&self, id: MidiAssetId) -> bool {
        {
            let mut pbs = self.shared.playbacks.lock().unwrap();
            let mut in_use = self.shared.channels_in_use.lock().unwrap();
            pbs.retain(|p| {
                if p.asset_id == id {
                    for &m in &p.channel_map {
                        if m != 0xff {
                            in_use[m as usize] = false;
                        }
                    }
                    false
                } else {
                    true
                }
            });
        }
        self.shared.assets.lock().unwrap().remove(&id).is_some()
    }

    pub fn free_all(&self) {
        self.stop_all();
        self.shared.assets.lock().unwrap().clear();
    }

    pub fn count(&self) -> usize {
        self.shared.assets.lock().unwrap().len()
    }

    pub fn exists(&self, id: MidiAssetId) -> bool {
        self.shared.assets.lock().unwrap().contains_key(&id)
    }

    pub fn tempo_bpm(&self, id: MidiAssetId) -> Option<f32> {
        self.shared
            .assets
            .lock()
            .unwrap()
            .get(&id)
            .map(|a| a.tempo_bpm)
    }

    pub fn title(&self, id: MidiAssetId) -> Option<String> {
        self.shared
            .assets
            .lock()
            .unwrap()
            .get(&id)
            .map(|a| a.title.clone())
    }

    /// Start playing a loaded asset at the given `volume` (0..1.5, clamped).
    ///
    /// Returns `false` if the id is unknown or if there aren't enough free
    /// MIDI channels for the tune's voices. Channel 9 (drums) is reserved
    /// when the tune has any percussion voice and is allocated exclusively.
    pub fn play(&self, id: MidiAssetId, volume: f32) -> bool {
        let mask = {
            let assets = self.shared.assets.lock().unwrap();
            match assets.get(&id) {
                Some(a) => a.used_channels_mask,
                None => return false,
            }
        };

        let mut channel_map = [0xff_u8; 16];
        {
            let mut in_use = self.shared.channels_in_use.lock().unwrap();
            if (mask & (1 << 9)) != 0 {
                if in_use[9] {
                    return false;
                }
                in_use[9] = true;
                channel_map[9] = 9;
            }
            for source in 0..16 {
                if source == 9 {
                    continue;
                }
                if (mask & (1 << source)) == 0 {
                    continue;
                }
                let mut found = None;
                for candidate in 0..16 {
                    if candidate == 9 || in_use[candidate] {
                        continue;
                    }
                    found = Some(candidate);
                    break;
                }
                match found {
                    Some(ch) => {
                        in_use[ch] = true;
                        channel_map[source] = ch as u8;
                    }
                    None => {
                        // Roll back any reservations we already made.
                        for &m in &channel_map {
                            if m != 0xff {
                                in_use[m as usize] = false;
                            }
                        }
                        return false;
                    }
                }
            }
        }

        let pb = Playback {
            asset_id: id,
            volume: volume.clamp(0.0, 1.5),
            state: PlaybackState::Playing,
            event_index: 0,
            start: Instant::now(),
            elapsed_at_pause: Duration::ZERO,
            channel_map,
        };
        self.shared.playbacks.lock().unwrap().push(pb);
        true
    }

    pub fn play_simple(&self, id: MidiAssetId) -> bool {
        self.play(id, 1.0)
    }

    /// Pause every currently-playing asset. Emits "all notes off" on
    /// each reserved channel to silence sustaining notes.
    pub fn pause(&self) {
        let mut pbs = self.shared.playbacks.lock().unwrap();
        let now = Instant::now();
        let handle = self.shared.midi_out.lock().unwrap().0;
        for pb in pbs.iter_mut() {
            if pb.state != PlaybackState::Playing {
                continue;
            }
            pb.elapsed_at_pause = now - pb.start;
            pb.state = PlaybackState::Paused;
            for &m in &pb.channel_map {
                if m != 0xff {
                    send_all_notes_off(handle, m);
                }
            }
        }
    }

    /// Resume any paused playbacks. Inverse of [`Self::pause`].
    pub fn resume(&self) {
        let mut pbs = self.shared.playbacks.lock().unwrap();
        let now = Instant::now();
        for pb in pbs.iter_mut() {
            if pb.state != PlaybackState::Paused {
                continue;
            }
            pb.start = now - pb.elapsed_at_pause;
            pb.state = PlaybackState::Playing;
        }
    }

    /// Halt every active playback and reset the MIDI device. Asset
    /// definitions stay loaded — use [`Self::free_all`] to clear those too.
    pub fn stop_all(&self) {
        let handle = self.shared.midi_out.lock().unwrap().0;
        if !handle.is_null() {
            unsafe { midiOutReset(handle) };
        }
        let mut pbs = self.shared.playbacks.lock().unwrap();
        let mut in_use = self.shared.channels_in_use.lock().unwrap();
        for pb in pbs.iter() {
            for &m in &pb.channel_map {
                if m != 0xff {
                    in_use[m as usize] = false;
                }
            }
        }
        pbs.clear();
    }

    pub fn is_playing(&self) -> bool {
        let pbs = self.shared.playbacks.lock().unwrap();
        !pbs.is_empty()
    }

    /// `true` if at least one playback of `id` is active (playing or
    /// paused). Returns `false` if `id` is unknown, or if every playback
    /// has finished and the dispatcher has reaped it.
    pub fn is_asset_playing(&self, id: MidiAssetId) -> bool {
        self.shared
            .playbacks
            .lock()
            .unwrap()
            .iter()
            .any(|p| p.asset_id == id)
    }

    pub fn state(&self) -> PlaybackState {
        let pbs = self.shared.playbacks.lock().unwrap();
        if pbs.iter().any(|p| p.state == PlaybackState::Playing) {
            PlaybackState::Playing
        } else if pbs.iter().any(|p| p.state == PlaybackState::Paused) {
            PlaybackState::Paused
        } else {
            PlaybackState::Stopped
        }
    }

    pub fn set_master_volume(&self, v: f32) {
        *self.shared.master_volume.lock().unwrap() = v.clamp(0.0, 1.0);
    }

    pub fn master_volume(&self) -> f32 {
        *self.shared.master_volume.lock().unwrap()
    }

    pub fn set_music_bus_volume(&self, v: f32) {
        *self.shared.music_bus_volume.lock().unwrap() = v.clamp(0.0, 1.0);
    }

    pub fn music_bus_volume(&self) -> f32 {
        *self.shared.music_bus_volume.lock().unwrap()
    }
}

impl Drop for MidiRuntime {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(j) = self.worker.take() {
            let _ = j.join();
        }
        let handle = self.shared.midi_out.lock().unwrap().0;
        if !handle.is_null() {
            unsafe { midiOutReset(handle) };
            unsafe { midiOutClose(handle) };
        }
    }
}

fn send_short(handle: HMIDIOUT, status: u8, channel: u8, data1: u8, data2: u8) {
    if handle.is_null() {
        return;
    }
    let msg: u32 = ((status | (channel & 0x0F)) as u32)
        | ((data1 as u32) << 8)
        | ((data2 as u32) << 16);
    unsafe {
        midiOutShortMsg(handle, msg);
    }
}

fn send_all_notes_off(handle: HMIDIOUT, channel: u8) {
    send_short(handle, 0xB0, channel, 123, 0);
    send_short(handle, 0xB0, channel, 120, 0);
    send_short(handle, 0xB0, channel, 64, 0);
}

fn worker_main(shared: Arc<RuntimeShared>, shutdown: Arc<AtomicBool>) {
    while !shutdown.load(Ordering::SeqCst) {
        dispatch_due_events(&shared);
        let sleep_ms = if shared.playbacks.lock().unwrap().is_empty() {
            10
        } else {
            1
        };
        thread::sleep(Duration::from_millis(sleep_ms));
    }
}

fn dispatch_due_events(shared: &RuntimeShared) {
    let handle = shared.midi_out.lock().unwrap().0;
    let master = *shared.master_volume.lock().unwrap();
    let music_bus = *shared.music_bus_volume.lock().unwrap();
    let bus_gain = master * music_bus;
    let now = Instant::now();

    let mut pbs = shared.playbacks.lock().unwrap();
    let assets = shared.assets.lock().unwrap();
    let mut channels = shared.channels_in_use.lock().unwrap();

    let mut i = 0;
    while i < pbs.len() {
        if pbs[i].state == PlaybackState::Paused {
            i += 1;
            continue;
        }

        let pb = &mut pbs[i];
        let asset = match assets.get(&pb.asset_id) {
            Some(a) => a,
            None => {
                for &m in &pb.channel_map {
                    if m != 0xff {
                        channels[m as usize] = false;
                    }
                }
                pbs.swap_remove(i);
                continue;
            }
        };

        while pb.event_index < asset.events.len() {
            let ev = &asset.events[pb.event_index];
            let deadline = pb.start + Duration::from_nanos(ev.time_ns);
            if deadline > now {
                break;
            }
            let mapped = pb.channel_map[ev.source_channel as usize];
            if mapped != 0xff {
                let mut velocity = ev.data2;
                if (ev.status & 0xF0) == 0x90 && velocity != 0 {
                    let scaled = velocity as f32 * (bus_gain * pb.volume).clamp(0.0, 1.0);
                    velocity = scaled.round().clamp(0.0, 127.0) as u8;
                }
                send_short(handle, ev.status, mapped, ev.data1, velocity);
            }
            pb.event_index += 1;
        }

        if pb.event_index >= asset.events.len() {
            for &m in &pb.channel_map {
                if m != 0xff {
                    channels[m as usize] = false;
                }
            }
            pbs.swap_remove(i);
            continue;
        }
        i += 1;
    }
}
