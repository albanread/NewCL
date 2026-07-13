//! Sample-rate-agnostic synthesis engine. Generates `Buffer`s of f32 samples
//! that can be played by the WinMM runtime or exported to WAV.

use core::f32::consts::PI;

use crate::envelope::Adsr;
use crate::waveform::{Lcg, Waveform, waveform_sample};

/// Engine configuration shared by every rendered [`Buffer`].
///
/// The defaults (`44_100` Hz, stereo, 10 s cap) match the WinMM mixer in
/// [`newaudio-win`](../../newaudio_win/index.html) so registered buffers
/// can be played without resampling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    /// Output sample rate in Hz. Most consumer hardware is happy with
    /// 44 100 or 48 000.
    pub sample_rate: u32,
    /// Channel count. 1 = mono, 2 = stereo (interleaved).
    pub channels: u32,
    /// Safety clamp: any [`Effect`] requesting a longer duration is
    /// truncated to this value. Prevents an out-of-control preset from
    /// allocating gigabytes of `f32` samples.
    pub max_duration_secs: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            sample_rate: 44_100,
            channels: 2,
            max_duration_secs: 10.0,
        }
    }
}

/// One voice in the [`Effect::oscillators`] stack.
///
/// Each oscillator contributes `amplitude × waveform(2π × frequency × t + phase)`
/// to the output. Multiple oscillators on the same effect are summed
/// before envelope and post-processing.
#[derive(Debug, Clone, Copy)]
pub struct Oscillator {
    pub waveform: Waveform,
    /// Frequency in Hz.
    pub frequency: f32,
    /// Linear gain. Final amplitude is also scaled by 0.5 (stereo split)
    /// during rendering, so values up to ~2.0 are sensible.
    pub amplitude: f32,
    /// Phase offset in radians applied to the starting position.
    pub phase: f32,
    /// Duty cycle for [`Waveform::Pulse`] only, in `[0, 1]`. Ignored
    /// otherwise.
    pub pulse_width: f32,
}

impl Default for Oscillator {
    fn default() -> Self {
        Self {
            waveform: Waveform::Sine,
            frequency: 440.0,
            amplitude: 1.0,
            phase: 0.0,
            pulse_width: 0.5,
        }
    }
}

/// Recipe for a single synthesised sound.
///
/// Mirrors `SynthSoundEffect` from the original C++ source but trimmed
/// to the parameters the game-effect presets actually use. Build one
/// manually or grab a starting point from a `*_effect` helper in
/// [`crate::presets`].
///
/// Rendering pipeline ([`Engine::render`]):
/// 1. Sum every [`Oscillator`] in [`Self::oscillators`].
/// 2. Add a sine oscillator that glides from [`Self::pitch_sweep_start`]
///    to [`Self::pitch_sweep_end`] (skipped if equal).
/// 3. Blend in white noise according to [`Self::noise_mix`].
/// 4. Multiply by the [`Self::envelope`] curve.
/// 5. Soft-clip via `tanh` if [`Self::distortion`] > 0.
/// 6. Echo: stack [`Self::echo_count`] taps with decay [`Self::echo_decay`].
/// 7. Peak-normalise to 0.9.
#[derive(Debug, Clone)]
pub struct Effect {
    /// Total length in seconds. Clamped to `[0.01, max_duration_secs]`.
    pub duration: f32,
    pub oscillators: Vec<Oscillator>,
    pub envelope: Adsr,
    /// First-cycle frequency of the pitch-sweep oscillator. 0 disables.
    pub pitch_sweep_start: f32,
    /// Final-cycle frequency of the pitch-sweep oscillator.
    pub pitch_sweep_end: f32,
    /// White-noise blend amount in `[0, 1]`. 0 = pure tone, 1 = pure noise.
    pub noise_mix: f32,
    /// Distortion amount; 0 disables. Values above ~0.3 sound quite gritty.
    pub distortion: f32,
    /// Per-tap echo delay in seconds.
    pub echo_delay: f32,
    /// Per-tap echo gain (multiplied each tap; 0..1).
    pub echo_decay: f32,
    /// Number of echo taps; 0 disables.
    pub echo_count: u32,
}

impl Default for Effect {
    fn default() -> Self {
        Self {
            duration: 0.2,
            oscillators: Vec::new(),
            envelope: Adsr::default(),
            pitch_sweep_start: 0.0,
            pitch_sweep_end: 0.0,
            noise_mix: 0.0,
            distortion: 0.0,
            echo_delay: 0.0,
            echo_decay: 0.0,
            echo_count: 0,
        }
    }
}

/// Rendered audio buffer.
///
/// `samples` is interleaved L/R for stereo (`channels == 2`) or one
/// sample per frame for mono. Values are nominally in `[-1, 1]` but
/// brief overshoots are possible before normalisation.
///
/// Buffers are passed by reference to [`crate::write_wav`] or moved
/// into the WinMM runtime via `register_sound`.
#[derive(Debug, Clone)]
pub struct Buffer {
    pub sample_rate: u32,
    pub channels: u32,
    /// Length in seconds — kept in sync with `samples.len()`.
    pub duration: f32,
    /// Interleaved samples. `samples.len() == frame_count * channels`.
    pub samples: Vec<f32>,
}

impl Buffer {
    pub fn new(cfg: Config, duration: f32) -> Self {
        let mut b = Self {
            sample_rate: cfg.sample_rate,
            channels: cfg.channels,
            duration,
            samples: Vec::new(),
        };
        b.resize(duration);
        b
    }

    pub fn resize(&mut self, duration_secs: f32) {
        self.duration = duration_secs;
        let frame_count = (self.sample_rate as f32 * duration_secs) as usize;
        self.samples
            .resize(frame_count * self.channels as usize, 0.0);
    }

    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / self.channels as usize
        }
    }
}

/// Stateful synthesizer.
///
/// `Engine` holds the sample-rate/channel configuration plus a seeded
/// linear-congruential RNG used by [`Waveform::Noise`] and the variation
/// helpers. Two engines built with [`Engine::with_seed`] using the same
/// seed and config produce **byte-identical** output for the same input,
/// which is what the WAV golden-byte tests rely on.
///
/// Use the impl methods in [`crate::presets`] for stock game SFX, and
/// the methods in [`crate::synth`] for tone / note / noise / FM builders.
#[derive(Debug, Clone)]
pub struct Engine {
    pub config: Config,
    noise: Lcg,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new(Config::default())
    }
}

impl Engine {
    pub fn new(config: Config) -> Self {
        Self {
            config,
            noise: Lcg::new(12_345),
        }
    }

    pub fn with_seed(config: Config, seed: u32) -> Self {
        Self {
            config,
            noise: Lcg::new(seed),
        }
    }

    pub fn reseed(&mut self, seed: u32) {
        self.noise = Lcg::new(seed);
    }

    /// Crate-private mutable access to the engine's LCG. Used by presets
    /// that need deterministic randomness (e.g. `random_beep`).
    pub(crate) fn noise_mut(&mut self) -> &mut Lcg {
        &mut self.noise
    }

    /// Render an `Effect` to a buffer. Mirrors `SynthEngine::generateSound`.
    pub fn render(&mut self, effect: &Effect) -> Buffer {
        let mut clamped = effect
            .duration
            .clamp(0.0, self.config.max_duration_secs);
        if clamped <= 0.0 {
            clamped = 0.01;
        }

        let mut buf = Buffer::new(self.config, clamped);
        let frame_count = buf.frame_count();
        let dt = 1.0 / buf.sample_rate as f32;
        let two_pi = 2.0 * PI;

        for frame in 0..frame_count {
            let time = frame as f32 * dt;
            let mut sample = 0.0_f32;

            // Oscillator stack.
            for osc in &effect.oscillators {
                let phase = two_pi * osc.frequency * time + osc.phase;
                let s = waveform_sample(osc.waveform, phase, osc.pulse_width, &mut self.noise);
                sample += s * osc.amplitude;
            }

            // Pitch sweep — a third "virtual" sine oscillator that glides.
            if effect.pitch_sweep_start != effect.pitch_sweep_end {
                let sweep_t = if clamped > 0.0 { time / clamped } else { 0.0 };
                let freq = effect.pitch_sweep_start
                    + (effect.pitch_sweep_end - effect.pitch_sweep_start) * sweep_t;
                let phase = two_pi * freq * time;
                sample += waveform_sample(Waveform::Sine, phase, 0.5, &mut self.noise) * 0.5;
            }

            // Mix in white noise.
            if effect.noise_mix > 0.0 {
                let n = self.noise.next_signed();
                sample = sample * (1.0 - effect.noise_mix) + n * effect.noise_mix;
            }

            // Apply envelope across the *requested* (unclamped) duration so
            // the curve shape doesn't change just because we clipped the
            // tail at max_duration_secs.
            let env = effect.envelope.value_at(time, effect.duration);
            sample *= env;

            // Soft-clip distortion.
            if effect.distortion > 0.0 {
                let drive = 1.0 + effect.distortion * 10.0;
                let denom = drive.tanh();
                if denom != 0.0 {
                    sample = (sample * drive).tanh() / denom;
                }
            }

            for ch in 0..buf.channels as usize {
                buf.samples[frame * buf.channels as usize + ch] = sample * 0.5;
            }
        }

        // Echo taps.
        if effect.echo_count > 0 && effect.echo_delay > 0.0 {
            let delay_frames = (effect.echo_delay * buf.sample_rate as f32) as usize;
            let channels = buf.channels as usize;
            for echo in 0..effect.echo_count {
                let echo_start = delay_frames * (echo as usize + 1);
                let amp = effect.echo_decay.powi(echo as i32 + 1);
                for frame in 0..frame_count.saturating_sub(echo_start) {
                    for ch in 0..channels {
                        let src = frame * channels + ch;
                        let dst = (frame + echo_start) * channels + ch;
                        let v = buf.samples[src] * amp;
                        buf.samples[dst] += v;
                    }
                }
            }
        }

        normalize(&mut buf, 0.9);
        buf
    }
}

/// Peak-normalise if the buffer exceeds `target`. Otherwise leave it.
pub fn normalize(buf: &mut Buffer, target: f32) {
    let mut peak = 0.0_f32;
    for s in &buf.samples {
        peak = peak.max(s.abs());
    }
    if peak > 1.0 {
        let t = target.clamp(0.0, 1.0);
        let scale = t / peak;
        for s in &mut buf.samples {
            *s *= scale;
        }
    }
}

/// MIDI note → frequency. A4 (note 69) = 440 Hz.
pub fn note_to_frequency(midi: i32) -> f32 {
    440.0 * 2.0_f32.powf((midi - 69) as f32 / 12.0)
}

/// Frequency → nearest MIDI note.
pub fn frequency_to_note(hz: f32) -> i32 {
    if hz <= 0.0 {
        return 0;
    }
    (69.0 + 12.0 * (hz / 440.0).log2()).round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_freq_roundtrip() {
        for midi in 21..=108 {
            let f = note_to_frequency(midi);
            let m = frequency_to_note(f);
            assert_eq!(m, midi, "roundtrip failed for midi {midi}: {f}Hz → {m}");
        }
    }

    #[test]
    fn a4_is_440() {
        assert!((note_to_frequency(69) - 440.0).abs() < 1e-3);
    }

    #[test]
    fn empty_effect_yields_silent_buffer() {
        let mut eng = Engine::new(Config::default());
        let e = Effect {
            duration: 0.1,
            ..Default::default()
        };
        let b = eng.render(&e);
        assert!(b.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn buffer_size_matches_duration() {
        let mut eng = Engine::new(Config::default());
        let e = Effect {
            duration: 0.25,
            oscillators: vec![Oscillator {
                waveform: Waveform::Sine,
                frequency: 440.0,
                amplitude: 0.5,
                ..Default::default()
            }],
            envelope: Adsr {
                attack: 0.01,
                decay: 0.05,
                sustain: 0.7,
                release: 0.1,
            },
            ..Default::default()
        };
        let b = eng.render(&e);
        let expected_frames = (44_100.0 * 0.25) as usize;
        assert_eq!(b.frame_count(), expected_frames);
        assert_eq!(b.samples.len(), expected_frames * 2);
    }

    #[test]
    fn render_is_deterministic_with_same_seed() {
        let cfg = Config::default();
        let e = Effect {
            duration: 0.05,
            noise_mix: 0.5,
            ..Default::default()
        };
        let a = Engine::with_seed(cfg, 7).render(&e);
        let b = Engine::with_seed(cfg, 7).render(&e);
        assert_eq!(a.samples, b.samples);
    }

    #[test]
    fn render_changes_with_different_seed() {
        let cfg = Config::default();
        let e = Effect {
            duration: 0.05,
            noise_mix: 0.5,
            ..Default::default()
        };
        let a = Engine::with_seed(cfg, 1).render(&e);
        let b = Engine::with_seed(cfg, 2).render(&e);
        assert_ne!(a.samples, b.samples);
    }

    #[test]
    fn normalize_caps_peak() {
        let mut b = Buffer {
            sample_rate: 44_100,
            channels: 1,
            duration: 0.001,
            samples: vec![2.0, -2.0, 1.5, -1.5],
        };
        normalize(&mut b, 0.9);
        let peak = b.samples.iter().fold(0.0_f32, |a, s| a.max(s.abs()));
        assert!((peak - 0.9).abs() < 1e-6, "peak after normalize = {peak}");
    }
}
