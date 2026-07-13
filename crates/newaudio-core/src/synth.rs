//! Source-parity sound builders. These complete the `winscheme_sound.h`
//! surface that wasn't in [`presets`](crate::presets): plain tone, MIDI
//! note with custom ADSR, coloured noise, FM, and combined-with-effect
//! variants.

use core::f32::consts::PI;

use crate::effects::{FilterType, apply_distortion, apply_echo, apply_filter};
use crate::engine::{Buffer, Config, Effect, Engine, Oscillator, note_to_frequency};
use crate::envelope::Adsr;
use crate::waveform::Waveform;

/// Coloured noise variant. Matches the `noise_type` codes in the original
/// `snd_noise()` export: 0=white, 1=pink, 2=brown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum NoiseType {
    White = 0,
    Pink = 1,
    Brown = 2,
}

impl NoiseType {
    pub fn from_code(code: i32) -> Self {
        match code {
            1 => Self::Pink,
            2 => Self::Brown,
            _ => Self::White,
        }
    }
}

impl Engine {
    /// Pure oscillator tone with a default punchy envelope.
    pub fn tone(&mut self, frequency: f32, duration: f32, waveform: Waveform) -> Buffer {
        let mut effect = Effect {
            duration,
            envelope: Adsr {
                attack: 0.01,
                decay: 0.05,
                sustain: 0.8,
                release: 0.1,
            },
            ..Default::default()
        };
        effect.oscillators.push(Oscillator {
            waveform,
            frequency: frequency.max(1.0),
            amplitude: 0.75,
            pulse_width: if matches!(waveform, Waveform::Pulse) {
                0.25
            } else {
                0.5
            },
            ..Default::default()
        });
        self.render(&effect)
    }

    /// MIDI note (0..127) with a custom ADSR.
    pub fn midi_note(
        &mut self,
        midi: i32,
        duration: f32,
        waveform: Waveform,
        envelope: Adsr,
    ) -> Buffer {
        let frequency = note_to_frequency(midi);
        let mut effect = Effect {
            duration,
            envelope,
            ..Default::default()
        };
        effect.oscillators.push(Oscillator {
            waveform,
            frequency: frequency.max(1.0),
            amplitude: 0.75,
            pulse_width: if matches!(waveform, Waveform::Pulse) {
                0.25
            } else {
                0.5
            },
            ..Default::default()
        });
        self.render(&effect)
    }

    /// Coloured noise generator. White is uniform; pink uses the Voss-style
    /// 3-pole approximation; brown is a clamped random walk.
    ///
    /// Output is windowed by a quadratic decay envelope to suit short SFX
    /// (matches the original `createNoiseBuffer` behaviour).
    pub fn noise(&mut self, kind: NoiseType, duration: f32) -> Buffer {
        let mut buf = Buffer::new(self.config, duration.max(0.01));
        let channels = buf.channels as usize;
        let frame_count = buf.frame_count();

        let mut pink0 = 0.0_f32;
        let mut pink1 = 0.0_f32;
        let mut pink2 = 0.0_f32;
        let mut brown = 0.0_f32;

        for frame in 0..frame_count {
            let mut value = self.noise_mut().next_signed();
            match kind {
                NoiseType::Pink => {
                    pink0 = 0.99765 * pink0 + value * 0.099_046;
                    pink1 = 0.96300 * pink1 + value * 0.296_516_4;
                    pink2 = 0.57000 * pink2 + value * 1.052_691_3;
                    value = (pink0 + pink1 + pink2 + value * 0.1848) * 0.25;
                }
                NoiseType::Brown => {
                    brown += value * 0.02;
                    brown = brown.clamp(-1.0, 1.0);
                    value = brown;
                }
                NoiseType::White => {}
            }
            let t = frame as f32 / frame_count.max(1) as f32;
            let env = (1.0 - t).powi(2);
            value *= env * 0.6;
            for ch in 0..channels {
                buf.samples[frame * channels + ch] = value;
            }
        }
        crate::engine::normalize(&mut buf, 0.95);
        buf
    }

    /// Simple FM synthesis: carrier sine modulated by a sine modulator.
    /// `mod_index` is in radians and controls the harmonic spread.
    pub fn fm(
        &mut self,
        carrier_hz: f32,
        modulator_hz: f32,
        mod_index: f32,
        duration: f32,
    ) -> Buffer {
        let mut buf = Buffer::new(self.config, duration.max(0.01));
        let channels = buf.channels as usize;
        let frame_count = buf.frame_count();

        let carrier = carrier_hz.max(1.0);
        let modulator = modulator_hz.max(1.0);
        let index = mod_index.max(0.0);

        for frame in 0..frame_count {
            let time = frame as f32 / buf.sample_rate as f32;
            let env_t = frame as f32 / frame_count.max(1) as f32;
            let env = (1.0 - env_t).powf(1.5);
            let modulation = (2.0 * PI * modulator * time).sin() * index;
            let sample = (2.0 * PI * carrier * time + modulation).sin() * env * 0.7;
            for ch in 0..channels {
                buf.samples[frame * channels + ch] = sample;
            }
        }
        crate::engine::normalize(&mut buf, 0.95);
        buf
    }

    /// Tone with a filter applied after rendering.
    pub fn filtered_tone(
        &mut self,
        frequency: f32,
        duration: f32,
        waveform: Waveform,
        filter: FilterType,
        cutoff: f32,
        resonance: f32,
    ) -> Buffer {
        let mut buf = self.tone(frequency, duration, waveform);
        apply_filter(&mut buf, filter, cutoff, resonance);
        crate::engine::normalize(&mut buf, 0.95);
        buf
    }

    /// MIDI note with a filter applied after rendering.
    pub fn filtered_note(
        &mut self,
        midi: i32,
        duration: f32,
        waveform: Waveform,
        envelope: Adsr,
        filter: FilterType,
        cutoff: f32,
        resonance: f32,
    ) -> Buffer {
        let mut buf = self.midi_note(midi, duration, waveform, envelope);
        apply_filter(&mut buf, filter, cutoff, resonance);
        crate::engine::normalize(&mut buf, 0.95);
        buf
    }

    /// Tone with a multi-tap reverb-style echo. `room` and `damping` are
    /// in `[0, 1]` and shape the per-tap delay length and feedback amount.
    pub fn reverb_tone(
        &mut self,
        frequency: f32,
        duration: f32,
        waveform: Waveform,
        room: f32,
        damping: f32,
        wet: f32,
    ) -> Buffer {
        let mut buf = self.tone(frequency, duration, waveform);
        let delay = (room * 0.08).max(0.03);
        let feedback = damping.clamp(0.2, 0.9);
        apply_echo(&mut buf, delay, feedback, wet, 4);
        buf
    }

    /// Tone with a single-tap delay (echo) line.
    pub fn delay_tone(
        &mut self,
        frequency: f32,
        duration: f32,
        waveform: Waveform,
        delay_time: f32,
        feedback: f32,
        mix: f32,
    ) -> Buffer {
        let mut buf = self.tone(frequency, duration, waveform);
        apply_echo(&mut buf, delay_time, feedback, mix, 3);
        buf
    }

    /// Tone with a low-pass + tanh distortion pipeline. `tone` controls
    /// the pre-distortion cutoff (0..1 maps to 0..8 kHz).
    pub fn distortion_tone(
        &mut self,
        frequency: f32,
        duration: f32,
        waveform: Waveform,
        drive: f32,
        tone: f32,
        level: f32,
    ) -> Buffer {
        let mut buf = self.tone(frequency, duration, waveform);
        let cutoff = (tone * 8_000.0).max(200.0);
        apply_filter(&mut buf, FilterType::LowPass, cutoff, 0.0);
        apply_distortion(&mut buf, drive, level);
        buf
    }
}

/// Free function variants for callers that don't want to manage an Engine.
pub fn tone(cfg: Config, frequency: f32, duration: f32, waveform: Waveform) -> Buffer {
    Engine::new(cfg).tone(frequency, duration, waveform)
}
pub fn midi_note(
    cfg: Config,
    midi: i32,
    duration: f32,
    waveform: Waveform,
    envelope: Adsr,
) -> Buffer {
    Engine::new(cfg).midi_note(midi, duration, waveform, envelope)
}
pub fn noise(cfg: Config, kind: NoiseType, duration: f32, seed: u32) -> Buffer {
    Engine::with_seed(cfg, seed).noise(kind, duration)
}
pub fn fm(
    cfg: Config,
    carrier_hz: f32,
    modulator_hz: f32,
    mod_index: f32,
    duration: f32,
) -> Buffer {
    Engine::new(cfg).fm(carrier_hz, modulator_hz, mod_index, duration)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tone_produces_audible_signal() {
        let mut e = Engine::new(Config::default());
        let b = e.tone(440.0, 0.1, Waveform::Sine);
        let peak = b.samples.iter().fold(0.0_f32, |a, &s| a.max(s.abs()));
        assert!(peak > 0.05);
    }

    #[test]
    fn midi_note_matches_tone_frequency() {
        // MIDI 69 = A4 = 440 Hz, so midi_note(69) and tone(440) should be
        // close (they differ in envelope choices and amplitude only).
        let mut e1 = Engine::with_seed(Config::default(), 1);
        let mut e2 = Engine::with_seed(Config::default(), 1);
        let a = e1.tone(440.0, 0.1, Waveform::Sine);
        let b = e2.midi_note(
            69,
            0.1,
            Waveform::Sine,
            Adsr {
                attack: 0.01,
                decay: 0.05,
                sustain: 0.8,
                release: 0.1,
            },
        );
        // The two should be byte-identical with matching ADSR.
        assert_eq!(a.samples, b.samples);
    }

    #[test]
    fn noise_white_is_broadband() {
        let mut e = Engine::with_seed(Config::default(), 12345);
        let b = e.noise(NoiseType::White, 0.05);
        let any_nonzero = b.samples.iter().any(|&s| s.abs() > 0.01);
        assert!(any_nonzero);
    }

    #[test]
    fn noise_brown_is_smoother_than_white() {
        // A crude correlation check: brown noise has high frame-to-frame
        // correlation, white noise does not.
        let mut white = Engine::with_seed(Config::default(), 42).noise(NoiseType::White, 0.1);
        let mut brown = Engine::with_seed(Config::default(), 42).noise(NoiseType::Brown, 0.1);
        // Normalize away the per-buffer scaling so the comparison is fair.
        crate::engine::normalize(&mut white, 1.0);
        crate::engine::normalize(&mut brown, 1.0);

        fn neighbour_diff(b: &Buffer) -> f32 {
            let mut total = 0.0;
            let channels = b.channels as usize;
            for f in 1..b.frame_count() {
                let a = b.samples[(f - 1) * channels];
                let bb = b.samples[f * channels];
                total += (bb - a).abs();
            }
            total / b.frame_count() as f32
        }
        let dw = neighbour_diff(&white);
        let db = neighbour_diff(&brown);
        assert!(
            db < dw,
            "brown ({db}) should be smoother than white ({dw})"
        );
    }

    #[test]
    fn fm_produces_signal() {
        let mut e = Engine::new(Config::default());
        let b = e.fm(440.0, 110.0, 2.0, 0.1);
        let peak = b.samples.iter().fold(0.0_f32, |a, &s| a.max(s.abs()));
        assert!(peak > 0.1, "fm peak = {peak}");
    }

    #[test]
    fn filtered_tone_is_less_bright() {
        let mut e1 = Engine::with_seed(Config::default(), 1);
        let mut e2 = Engine::with_seed(Config::default(), 1);
        let plain = e1.tone(2000.0, 0.1, Waveform::Sawtooth);
        let filt = e2.filtered_tone(2000.0, 0.1, Waveform::Sawtooth, FilterType::LowPass, 400.0, 0.0);
        // The low-passed version should have lower total energy than the
        // raw saw at 2 kHz.
        let energy = |b: &Buffer| -> f32 { b.samples.iter().map(|s| s * s).sum() };
        assert!(energy(&filt) < energy(&plain));
    }

    #[test]
    fn noise_with_same_seed_is_deterministic() {
        let mut e1 = Engine::with_seed(Config::default(), 99);
        let mut e2 = Engine::with_seed(Config::default(), 99);
        let a = e1.noise(NoiseType::Pink, 0.05);
        let b = e2.noise(NoiseType::Pink, 0.05);
        assert_eq!(a.samples, b.samples);
    }
}
