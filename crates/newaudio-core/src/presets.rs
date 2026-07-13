//! Game sound-effect presets.
//!
//! Each preset is a small `Effect` recipe — a stack of oscillators plus
//! an ADSR envelope, optional pitch sweep, noise mix, and distortion —
//! that renders to a `Buffer` ready to play through [`crate::Engine`]
//! and the WinMM runtime.
//!
//! Parameter conventions:
//! - `duration` is in seconds.
//! - Most other parameters are dimensionless "intensity" scalars between
//!   ~0.5 and ~2 with 1.0 as the nominal value. They subtly shape the
//!   preset; the audible character is fixed.
//! - To customise further, build your own [`Effect`] and call
//!   `Engine::render` directly.

use crate::engine::{Effect, Engine, Oscillator};
use crate::envelope::Adsr;
use crate::waveform::Waveform;

/// `Engine`-bound builder methods for the stock game presets.
///
/// Every method returns a fully rendered [`crate::Buffer`]. The buffer
/// is interleaved stereo `f32` and can be played through the WinMM
/// runtime, written to a `.wav` file, or post-processed via the helpers
/// in [`crate::effects`].
impl Engine {
    /// Pure sine beep with a snappy ADSR. Good for menu confirmations
    /// and UI clicks at low frequencies, lasers at high frequencies.
    pub fn beep(&mut self, frequency: f32, duration: f32) -> crate::engine::Buffer {
        let e = beep_effect(frequency, duration);
        self.render(&e)
    }

    /// Two-oscillator chime (B5 → E6). Classic pickup/coin sound.
    /// The `pitch` parameter is accepted for source-compatibility but
    /// the preset hard-codes the interval.
    pub fn coin(&mut self, _pitch: f32, duration: f32) -> crate::engine::Buffer {
        let e = coin_effect(duration);
        self.render(&e)
    }

    /// Upward pitch sweep (300 → 600 Hz). Mario-style jump.
    pub fn jump(&mut self, _power: f32, duration: f32) -> crate::engine::Buffer {
        let e = jump_effect(duration);
        self.render(&e)
    }

    /// Low sine + triangle body + noise + downward sweep + soft distortion.
    /// `size` scales the noise mix.
    pub fn explode(&mut self, size: f32, duration: f32) -> crate::engine::Buffer {
        let e = explode_effect(size, duration);
        self.render(&e)
    }

    /// Heavier variant of [`Self::explode`] with an added sub-oscillator,
    /// longer release, and more distortion.
    pub fn big_explosion(&mut self, size: f32, duration: f32) -> crate::engine::Buffer {
        let e = big_explosion_effect(size, duration);
        self.render(&e)
    }

    /// Lighter variant of [`Self::explode`] without the extra sub.
    pub fn small_explosion(&mut self, intensity: f32, duration: f32) -> crate::engine::Buffer {
        let e = explode_effect(intensity, duration);
        self.render(&e)
    }

    /// `explode` with elevated noise mix to suggest air dispersal.
    pub fn distant_explosion(&mut self, distance: f32, duration: f32) -> crate::engine::Buffer {
        let mut e = explode_effect(distance, duration);
        e.noise_mix = 0.4;
        self.render(&e)
    }

    /// `explode` with heavy noise and distortion — metallic shrapnel.
    pub fn metal_explosion(&mut self, shrapnel: f32, duration: f32) -> crate::engine::Buffer {
        let mut e = explode_effect(shrapnel, duration);
        e.noise_mix = 0.7;
        e.distortion = 0.3;
        self.render(&e)
    }

    /// Downward 1 kHz → 100 Hz sweep with light noise. Laser zap.
    pub fn zap(&mut self, _frequency: f32, duration: f32) -> crate::engine::Buffer {
        let e = zap_effect(duration);
        self.render(&e)
    }

    /// Downward 800 → 200 Hz sweep with heavy noise. Gunshot.
    pub fn shoot(&mut self, _power: f32, duration: f32) -> crate::engine::Buffer {
        let e = shoot_effect(duration);
        self.render(&e)
    }

    /// Upward 200 → 800 Hz square sweep with slow attack. Power-up jingle.
    pub fn powerup(&mut self, _intensity: f32, duration: f32) -> crate::engine::Buffer {
        let e = powerup_effect(duration);
        self.render(&e)
    }

    /// Downward 600 → 200 Hz sweep with medium noise. Hurt grunt.
    pub fn hurt(&mut self, _severity: f32, duration: f32) -> crate::engine::Buffer {
        let e = hurt_effect(duration);
        self.render(&e)
    }

    /// Single-frame noise burst with snap ADSR. UI click.
    pub fn click(&mut self, _sharpness: f32, duration: f32) -> crate::engine::Buffer {
        let e = click_effect(duration);
        self.render(&e)
    }

    /// Wide-band noise pop. Crude bang for impacts that aren't explosions.
    pub fn bang(&mut self, _intensity: f32, duration: f32) -> crate::engine::Buffer {
        let e = bang_effect(duration);
        self.render(&e)
    }

    /// High-pitched short beep. `pitch` multiplies the 800 Hz base.
    pub fn blip(&mut self, pitch: f32, duration: f32) -> crate::engine::Buffer {
        self.beep(800.0 * pitch, duration)
    }

    /// Brighter chime variant of [`Self::coin`] for item-pickup feedback.
    pub fn pickup(&mut self, _brightness: f32, duration: f32) -> crate::engine::Buffer {
        let e = coin_effect(duration);
        self.render(&e)
    }

    /// Linear frequency sweep from `start_hz` to `end_hz` with sustain-y
    /// ADSR. Use [`Self::sweep_down`] if you prefer the down-named alias.
    pub fn sweep_up(
        &mut self,
        start_hz: f32,
        end_hz: f32,
        duration: f32,
    ) -> crate::engine::Buffer {
        let e = sweep_effect(start_hz, end_hz, duration);
        self.render(&e)
    }

    /// Alias of [`Self::sweep_up`]: linear sweep `start_hz` → `end_hz`.
    /// Use whichever name reads better at the call site.
    pub fn sweep_down(
        &mut self,
        start_hz: f32,
        end_hz: f32,
        duration: f32,
    ) -> crate::engine::Buffer {
        let e = sweep_effect(start_hz, end_hz, duration);
        self.render(&e)
    }

    /// Procedural beep: reseeds the engine, then draws frequency
    /// (200..1000 Hz) and a small duration jitter from the LCG.
    /// Identical seeds produce identical output.
    pub fn random_beep(&mut self, seed: u32, duration: f32) -> crate::engine::Buffer {
        self.reseed(seed);
        let base = if duration > 0.0 { duration } else { 0.2 };
        let freq = self.noise_mut().next_range(200.0, 1000.0);
        let dur = self.noise_mut().next_range(base * 0.5, base * 1.5);
        self.beep(freq, dur)
    }
}

// ---------------------------------------------------------------------------
// Effect builders — pure data, easy to unit-test without rendering.
//
// The `*_effect` functions return an [`Effect`] *recipe* without rendering
// it. Modify the result before passing it to [`Engine::render`] when you
// want to tweak a preset (e.g. push the decay time longer, add an extra
// oscillator).
// ---------------------------------------------------------------------------

/// Recipe for a simple sine [`Self::beep`](Engine::beep).
pub fn beep_effect(frequency: f32, duration: f32) -> Effect {
    Effect {
        duration,
        oscillators: vec![Oscillator {
            waveform: Waveform::Sine,
            frequency,
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
    }
}

pub fn coin_effect(duration: f32) -> Effect {
    Effect {
        duration,
        oscillators: vec![
            Oscillator {
                waveform: Waveform::Sine,
                frequency: 987.77, // B5
                amplitude: 0.5,
                ..Default::default()
            },
            Oscillator {
                waveform: Waveform::Sine,
                frequency: 1318.51, // E6
                amplitude: 0.3,
                ..Default::default()
            },
        ],
        envelope: Adsr {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.3,
            release: 0.15,
        },
        ..Default::default()
    }
}

pub fn jump_effect(duration: f32) -> Effect {
    Effect {
        duration,
        pitch_sweep_start: 300.0,
        pitch_sweep_end: 600.0,
        envelope: Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.5,
            release: 0.1,
        },
        ..Default::default()
    }
}

pub fn explode_effect(size: f32, duration: f32) -> Effect {
    Effect {
        duration,
        oscillators: vec![
            Oscillator {
                waveform: Waveform::Sine,
                frequency: 58.0,
                amplitude: 0.95,
                ..Default::default()
            },
            Oscillator {
                waveform: Waveform::Triangle,
                frequency: 86.0,
                amplitude: 0.28,
                ..Default::default()
            },
        ],
        noise_mix: (0.06 * size).clamp(0.0, 0.12),
        pitch_sweep_start: 135.0,
        pitch_sweep_end: 32.0,
        envelope: Adsr {
            attack: 0.0015,
            decay: 0.14,
            sustain: 0.0,
            release: 0.10,
        },
        distortion: 0.08,
        ..Default::default()
    }
}

pub fn big_explosion_effect(size: f32, duration: f32) -> Effect {
    let mut e = explode_effect(size, duration);
    e.oscillators.push(Oscillator {
        waveform: Waveform::Sine,
        frequency: 36.0,
        amplitude: 0.55,
        ..Default::default()
    });
    e.pitch_sweep_start = 110.0;
    e.pitch_sweep_end = 22.0;
    e.noise_mix = e.noise_mix.max(0.14);
    e.envelope.decay = 0.22;
    e.envelope.release = 0.18;
    e.distortion = 0.14;
    e
}

pub fn zap_effect(duration: f32) -> Effect {
    Effect {
        duration,
        pitch_sweep_start: 1000.0,
        pitch_sweep_end: 100.0,
        noise_mix: 0.2,
        envelope: Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.3,
            release: 0.08,
        },
        ..Default::default()
    }
}

pub fn shoot_effect(duration: f32) -> Effect {
    Effect {
        duration,
        pitch_sweep_start: 800.0,
        pitch_sweep_end: 200.0,
        noise_mix: 0.3,
        envelope: Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.4,
            release: 0.08,
        },
        ..Default::default()
    }
}

pub fn powerup_effect(duration: f32) -> Effect {
    Effect {
        duration,
        pitch_sweep_start: 200.0,
        pitch_sweep_end: 800.0,
        oscillators: vec![Oscillator {
            waveform: Waveform::Square,
            frequency: 400.0,
            amplitude: 0.4,
            ..Default::default()
        }],
        envelope: Adsr {
            attack: 0.1,
            decay: 0.1,
            sustain: 0.8,
            release: 0.2,
        },
        ..Default::default()
    }
}

pub fn hurt_effect(duration: f32) -> Effect {
    Effect {
        duration,
        pitch_sweep_start: 600.0,
        pitch_sweep_end: 200.0,
        noise_mix: 0.4,
        envelope: Adsr {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.2,
            release: 0.15,
        },
        ..Default::default()
    }
}

pub fn click_effect(duration: f32) -> Effect {
    Effect {
        duration,
        oscillators: vec![Oscillator {
            waveform: Waveform::Noise,
            amplitude: 0.3,
            ..Default::default()
        }],
        envelope: Adsr {
            attack: 0.001,
            decay: 0.01,
            sustain: 0.0,
            release: 0.03,
        },
        ..Default::default()
    }
}

pub fn bang_effect(duration: f32) -> Effect {
    Effect {
        duration,
        noise_mix: 0.8,
        envelope: Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.0,
            release: 0.1,
        },
        ..Default::default()
    }
}

pub fn sweep_effect(start: f32, end: f32, duration: f32) -> Effect {
    Effect {
        duration,
        pitch_sweep_start: start,
        pitch_sweep_end: end,
        envelope: Adsr {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.8,
            release: 0.1,
        },
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Config;

    #[test]
    fn coin_has_two_oscillators_in_the_chime_interval() {
        let e = coin_effect(0.4);
        assert_eq!(e.oscillators.len(), 2);
        let f0 = e.oscillators[0].frequency;
        let f1 = e.oscillators[1].frequency;
        // The chime is B5 → E6 — a perfect fourth (5 semitones,
        // ratio ≈ 2^(5/12) ≈ 1.3348).
        let ratio = f1 / f0;
        assert!((ratio - 1.33484).abs() < 0.01, "ratio = {ratio}");
    }

    #[test]
    fn explode_sweeps_down() {
        let e = explode_effect(1.0, 0.6);
        assert!(e.pitch_sweep_start > e.pitch_sweep_end);
    }

    #[test]
    fn big_explosion_has_more_low_oscillators() {
        let e = explode_effect(1.0, 0.6);
        let big = big_explosion_effect(1.0, 0.6);
        assert_eq!(e.oscillators.len() + 1, big.oscillators.len());
    }

    #[test]
    fn click_uses_noise() {
        let e = click_effect(0.05);
        assert_eq!(e.oscillators[0].waveform, Waveform::Noise);
    }

    #[test]
    fn jump_sweeps_up() {
        let e = jump_effect(0.3);
        assert!(e.pitch_sweep_end > e.pitch_sweep_start);
    }

    #[test]
    fn render_beep_produces_audible_signal() {
        let mut eng = Engine::new(Config::default());
        let b = eng.beep(440.0, 0.2);
        let peak = b.samples.iter().fold(0.0_f32, |a, s| a.max(s.abs()));
        assert!(peak > 0.05, "expected an audible peak, got {peak}");
    }
}
