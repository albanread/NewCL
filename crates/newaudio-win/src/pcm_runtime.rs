//! `waveOut`-backed background mixer.
//!
//! Holds a [`SoundBank`] of pre-rendered [`Buffer`]s keyed by `SoundId`,
//! plus a list of active voices. A worker thread fills WaveOut blocks
//! continuously and mixes any active voices into them.
//!
//! Each call to [`PcmRuntime::play`] returns a [`VoiceHandle`] that
//! uniquely identifies that playback instance — needed for individually
//! stopping or fading looping sounds.

use std::collections::HashMap;
use std::mem::{size_of, zeroed};
use std::ptr::null_mut;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use newaudio_core::{Buffer, Lcg};
use windows_sys::Win32::Media::Audio::*;

/// `mmsystem.h` constant: success return code for `waveOut*` calls.
const MMSYSERR_NOERROR: u32 = 0;
/// `mmsystem.h` constant: ask the OS to pick a waveform output device.
const WAVE_MAPPER: u32 = 0xFFFF_FFFF;

const SAMPLE_RATE: u32 = 44_100;
const CHANNELS: u32 = 2;
const FRAMES_PER_BLOCK: usize = 2048;
const MIXER_BLOCKS: usize = 4;

pub type SoundId = u32;

/// Per-call instance id. Use it to address a specific looping or fading
/// playback without affecting other instances of the same sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoiceHandle(pub u64);

/// Optional parameters for [`PcmRuntime::play_with`].
#[derive(Debug, Clone, Copy)]
pub struct PlayOptions {
    pub volume: f32,
    pub pan: f32,
    /// If true, the voice restarts at frame 0 when it reaches the end of
    /// the buffer. Use [`PcmRuntime::stop_voice`] to end it.
    pub looping: bool,
    /// Fraction (`0.0..=1.0`) by which the volume is randomly perturbed
    /// per play. `0.05` = ±5%. Prevents repetitive SFX sounding identical.
    pub volume_jitter: f32,
}

impl Default for PlayOptions {
    fn default() -> Self {
        Self {
            volume: 1.0,
            pan: 0.0,
            looping: false,
            volume_jitter: 0.0,
        }
    }
}

#[derive(Debug)]
struct ActiveVoice {
    handle: VoiceHandle,
    sound_id: SoundId,
    frame_pos: usize,
    gain_left: f32,
    gain_right: f32,
    looping: bool,
    fade: Option<FadeOut>,
}

#[derive(Debug, Clone, Copy)]
struct FadeOut {
    /// Current gain multiplier, applied on top of per-voice gain.
    gain: f32,
    /// How much `gain` drops per frame.
    decrement: f32,
}

#[derive(Debug, Default)]
struct SoundBank {
    sounds: HashMap<SoundId, Arc<Buffer>>,
    next_id: SoundId,
}

impl SoundBank {
    fn insert(&mut self, buf: Buffer) -> SoundId {
        self.next_id = self.next_id.wrapping_add(1).max(1);
        let id = self.next_id;
        self.sounds.insert(id, Arc::new(buf));
        id
    }
    fn get(&self, id: SoundId) -> Option<Arc<Buffer>> {
        self.sounds.get(&id).cloned()
    }
    fn remove(&mut self, id: SoundId) -> bool {
        self.sounds.remove(&id).is_some()
    }
    fn memory_usage(&self) -> usize {
        self.sounds
            .values()
            .map(|b| b.samples.len() * size_of::<f32>())
            .sum()
    }
}

struct MixerBlock {
    header: WAVEHDR,
    samples: Vec<i16>,
}

unsafe impl Send for MixerBlock {}

#[repr(transparent)]
struct WaveOutHandle(HWAVEOUT);
unsafe impl Send for WaveOutHandle {}

struct RuntimeShared {
    bank: Mutex<SoundBank>,
    voices: Mutex<Vec<ActiveVoice>>,
    master_volume: Mutex<f32>,
    sfx_bus_volume: Mutex<f32>,
    rng: Mutex<Lcg>,
    next_voice_handle: AtomicU64,
    blocks: Mutex<Vec<Box<MixerBlock>>>,
    wave_out: Mutex<WaveOutHandle>,
}

/// Public handle to the background mixer. Drop to shut down.
///
/// All methods take `&self` and use internal locking — wrap in [`Arc`] to
/// share across game threads, or use [`crate::AudioThread`] for
/// non-blocking command submission.
pub struct PcmRuntime {
    shared: Arc<RuntimeShared>,
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl PcmRuntime {
    /// Start the mixer thread. Returns `Err` if `waveOutOpen` fails.
    pub fn start() -> Result<Self, String> {
        let mut wave_out: HWAVEOUT = null_mut();
        let mut format: WAVEFORMATEX = unsafe { zeroed() };
        format.wFormatTag = WAVE_FORMAT_PCM as u16;
        format.nChannels = CHANNELS as u16;
        format.nSamplesPerSec = SAMPLE_RATE;
        format.wBitsPerSample = 16;
        format.nBlockAlign = (format.nChannels * format.wBitsPerSample) / 8;
        format.nAvgBytesPerSec = format.nSamplesPerSec * format.nBlockAlign as u32;

        let rc = unsafe {
            waveOutOpen(
                &mut wave_out as *mut _,
                WAVE_MAPPER,
                &format as *const _,
                0,
                0,
                CALLBACK_NULL,
            )
        };
        if rc != MMSYSERR_NOERROR {
            return Err(format!("waveOutOpen failed: {rc}"));
        }

        let mut blocks: Vec<Box<MixerBlock>> = Vec::with_capacity(MIXER_BLOCKS);
        for _ in 0..MIXER_BLOCKS {
            let samples = vec![0i16; FRAMES_PER_BLOCK * CHANNELS as usize];
            let mut block = Box::new(MixerBlock {
                header: unsafe { zeroed() },
                samples,
            });
            block.header.lpData = block.samples.as_mut_ptr() as *mut _;
            block.header.dwBufferLength = (block.samples.len() * size_of::<i16>()) as u32;
            let prep = unsafe {
                waveOutPrepareHeader(
                    wave_out,
                    &mut block.header as *mut _,
                    size_of::<WAVEHDR>() as u32,
                )
            };
            if prep != MMSYSERR_NOERROR {
                unsafe { waveOutClose(wave_out) };
                return Err(format!("waveOutPrepareHeader failed: {prep}"));
            }
            blocks.push(block);
        }

        let shared = Arc::new(RuntimeShared {
            bank: Mutex::new(SoundBank::default()),
            voices: Mutex::new(Vec::new()),
            master_volume: Mutex::new(1.0),
            sfx_bus_volume: Mutex::new(1.0),
            rng: Mutex::new(Lcg::new(0xC0FFEE_u32)),
            next_voice_handle: AtomicU64::new(1),
            blocks: Mutex::new(blocks),
            wave_out: Mutex::new(WaveOutHandle(wave_out)),
        });
        let shutdown = Arc::new(AtomicBool::new(false));

        let worker_shared = Arc::clone(&shared);
        let worker_shutdown = Arc::clone(&shutdown);
        let worker = thread::spawn(move || mixer_thread(worker_shared, worker_shutdown));

        Ok(Self {
            shared,
            shutdown,
            worker: Some(worker),
        })
    }

    // -- Sound bank ---------------------------------------------------------

    pub fn register_sound(&self, buf: Buffer) -> SoundId {
        self.shared.bank.lock().unwrap().insert(buf)
    }

    pub fn free_sound(&self, id: SoundId) -> bool {
        {
            let mut voices = self.shared.voices.lock().unwrap();
            voices.retain(|v| v.sound_id != id);
        }
        self.shared.bank.lock().unwrap().remove(id)
    }

    pub fn free_all(&self) {
        self.shared.voices.lock().unwrap().clear();
        self.shared.bank.lock().unwrap().sounds.clear();
    }

    pub fn sound_count(&self) -> usize {
        self.shared.bank.lock().unwrap().sounds.len()
    }

    pub fn memory_usage(&self) -> usize {
        self.shared.bank.lock().unwrap().memory_usage()
    }

    pub fn duration_secs(&self, id: SoundId) -> f32 {
        self.shared
            .bank
            .lock()
            .unwrap()
            .get(id)
            .map(|b| b.duration)
            .unwrap_or(0.0)
    }

    // -- Playback -----------------------------------------------------------

    /// Play `id` at the given volume and pan. Returns a [`VoiceHandle`]
    /// for the new instance, or `None` if the sound id is unknown.
    pub fn play(&self, id: SoundId, volume: f32, pan: f32) -> Option<VoiceHandle> {
        self.play_with(
            id,
            PlayOptions {
                volume,
                pan,
                ..PlayOptions::default()
            },
        )
    }

    pub fn play_simple(&self, id: SoundId) -> Option<VoiceHandle> {
        self.play(id, 1.0, 0.0)
    }

    /// Play `id` with [`PlayOptions`]. Returns the [`VoiceHandle`] for the
    /// new instance, or `None` if the sound id is unknown.
    pub fn play_with(&self, id: SoundId, opts: PlayOptions) -> Option<VoiceHandle> {
        if self.shared.bank.lock().unwrap().get(id).is_none() {
            return None;
        }
        let mut volume = opts.volume.max(0.0);
        if opts.volume_jitter > 0.0 {
            let jitter = self.shared.rng.lock().unwrap().next_range(
                1.0 - opts.volume_jitter,
                1.0 + opts.volume_jitter,
            );
            volume *= jitter.max(0.0);
        }
        let pan = opts.pan.clamp(-1.0, 1.0);

        let handle = VoiceHandle(self.shared.next_voice_handle.fetch_add(1, Ordering::Relaxed));
        let voice = ActiveVoice {
            handle,
            sound_id: id,
            frame_pos: 0,
            gain_left: volume * if pan <= 0.0 { 1.0 } else { 1.0 - pan },
            gain_right: volume * if pan >= 0.0 { 1.0 } else { 1.0 + pan },
            looping: opts.looping,
            fade: None,
        };
        self.shared.voices.lock().unwrap().push(voice);
        Some(handle)
    }

    /// Play a looping voice. Stop it later with [`Self::stop_voice`].
    pub fn play_looped(&self, id: SoundId, volume: f32, pan: f32) -> Option<VoiceHandle> {
        self.play_with(
            id,
            PlayOptions {
                volume,
                pan,
                looping: true,
                ..PlayOptions::default()
            },
        )
    }

    /// Pick a random sound from the slice and play it. Returns `None` if
    /// the slice is empty or none of the ids are registered.
    ///
    /// Useful for variation packs: register N slightly different versions
    /// of a "hit" sound and call this each time you want an impact.
    pub fn play_random(
        &self,
        ids: &[SoundId],
        volume: f32,
        pan: f32,
    ) -> Option<VoiceHandle> {
        if ids.is_empty() {
            return None;
        }
        let idx = {
            let mut rng = self.shared.rng.lock().unwrap();
            (rng.next_unit() * ids.len() as f32) as usize
        };
        let idx = idx.min(ids.len() - 1);
        self.play(ids[idx], volume, pan)
    }

    /// Stop a specific voice. If `fade_out_secs` is `Some`, the voice
    /// fades over that duration before being removed; if `None`, it stops
    /// immediately.
    pub fn stop_voice(&self, handle: VoiceHandle, fade_out_secs: Option<f32>) {
        let mut voices = self.shared.voices.lock().unwrap();
        let Some(v) = voices.iter_mut().find(|v| v.handle == handle) else {
            return;
        };
        match fade_out_secs {
            None => {
                let h = v.handle;
                voices.retain(|x| x.handle != h);
            }
            Some(secs) => {
                let frames = (secs.max(0.0) * SAMPLE_RATE as f32).max(1.0);
                v.fade = Some(FadeOut {
                    gain: 1.0,
                    decrement: 1.0 / frames,
                });
                v.looping = false;
            }
        }
    }

    pub fn is_voice_active(&self, handle: VoiceHandle) -> bool {
        self.shared
            .voices
            .lock()
            .unwrap()
            .iter()
            .any(|v| v.handle == handle)
    }

    pub fn is_playing(&self, id: SoundId) -> bool {
        self.shared
            .voices
            .lock()
            .unwrap()
            .iter()
            .any(|v| v.sound_id == id)
    }

    pub fn stop_all(&self) {
        self.shared.voices.lock().unwrap().clear();
        let wave_out = self.shared.wave_out.lock().unwrap().0;
        if !wave_out.is_null() {
            unsafe { waveOutReset(wave_out) };
        }
    }

    pub fn stop_sound(&self, id: SoundId) {
        self.shared
            .voices
            .lock()
            .unwrap()
            .retain(|v| v.sound_id != id);
    }

    // -- Bus volumes --------------------------------------------------------

    pub fn set_master_volume(&self, v: f32) {
        *self.shared.master_volume.lock().unwrap() = v.clamp(0.0, 1.0);
    }
    pub fn master_volume(&self) -> f32 {
        *self.shared.master_volume.lock().unwrap()
    }
    pub fn set_sfx_bus_volume(&self, v: f32) {
        *self.shared.sfx_bus_volume.lock().unwrap() = v.clamp(0.0, 1.0);
    }
    pub fn sfx_bus_volume(&self) -> f32 {
        *self.shared.sfx_bus_volume.lock().unwrap()
    }
}

impl Drop for PcmRuntime {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(j) = self.worker.take() {
            let _ = j.join();
        }
        let wave_out = self.shared.wave_out.lock().unwrap().0;
        if !wave_out.is_null() {
            unsafe { waveOutReset(wave_out) };
            let mut blocks = self.shared.blocks.lock().unwrap();
            for block in blocks.iter_mut() {
                unsafe {
                    waveOutUnprepareHeader(
                        wave_out,
                        &mut block.header as *mut _,
                        size_of::<WAVEHDR>() as u32,
                    );
                }
            }
            unsafe { waveOutClose(wave_out) };
        }
    }
}

// ---------------------------------------------------------------------------
// Mixer thread internals
// ---------------------------------------------------------------------------

fn fill_and_queue_block(shared: &RuntimeShared, block: &mut MixerBlock) {
    for s in &mut block.samples {
        *s = 0;
    }
    let mut mixed = vec![0.0_f32; block.samples.len()];

    let master = *shared.master_volume.lock().unwrap();
    let sfx = *shared.sfx_bus_volume.lock().unwrap();
    let global_gain = master * sfx;

    {
        let mut voices = shared.voices.lock().unwrap();
        let bank = shared.bank.lock().unwrap();

        let mut i = 0;
        while i < voices.len() {
            let buffer = match bank.get(voices[i].sound_id) {
                Some(b) => b,
                None => {
                    voices.swap_remove(i);
                    continue;
                }
            };
            let total = buffer.frame_count();
            if total == 0 {
                voices.swap_remove(i);
                continue;
            }
            let channels = buffer.channels as usize;

            let mut remove = false;
            let v = &mut voices[i];
            let mut write_frame = 0_usize;
            while write_frame < FRAMES_PER_BLOCK {
                if v.frame_pos >= total {
                    if v.looping {
                        v.frame_pos = 0;
                    } else {
                        remove = true;
                        break;
                    }
                }
                let src = v.frame_pos * channels;
                let left = buffer.samples[src];
                let right = if channels > 1 {
                    buffer.samples[src + 1]
                } else {
                    left
                };
                // Fade-out: apply current gain, then decrement.
                let fade_gain = match v.fade.as_ref() {
                    Some(f) => f.gain,
                    None => 1.0,
                };
                mixed[write_frame * 2] += left * v.gain_left * global_gain * fade_gain;
                mixed[write_frame * 2 + 1] +=
                    right * v.gain_right * global_gain * fade_gain;

                if let Some(f) = v.fade.as_mut() {
                    f.gain -= f.decrement;
                    if f.gain <= 0.0 {
                        remove = true;
                        break;
                    }
                }
                write_frame += 1;
                v.frame_pos += 1;
            }

            if remove {
                voices.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    for (dst, &src) in block.samples.iter_mut().zip(mixed.iter()) {
        let clamped = src.clamp(-1.0, 1.0);
        *dst = (clamped * 32_767.0).round() as i16;
    }

    let wave_out = shared.wave_out.lock().unwrap().0;
    if !wave_out.is_null() {
        unsafe {
            waveOutWrite(
                wave_out,
                &mut block.header as *mut _,
                size_of::<WAVEHDR>() as u32,
            );
        }
    }
}

fn mixer_thread(shared: Arc<RuntimeShared>, shutdown: Arc<AtomicBool>) {
    while !shutdown.load(Ordering::SeqCst) {
        {
            let mut blocks = shared.blocks.lock().unwrap();
            for block in blocks.iter_mut() {
                let in_queue = (block.header.dwFlags & WHDR_INQUEUE) != 0;
                if in_queue {
                    continue;
                }
                fill_and_queue_block(&shared, block);
            }
        }
        thread::sleep(Duration::from_millis(2));
    }
}
