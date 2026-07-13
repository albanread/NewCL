//! Waveform generators. Phase is in radians.

use core::f32::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Waveform {
    Sine = 0,
    Square = 1,
    Sawtooth = 2,
    Triangle = 3,
    Noise = 4,
    Pulse = 5,
}

impl Waveform {
    pub fn from_code(code: i32) -> Self {
        match code {
            1 => Self::Square,
            2 => Self::Sawtooth,
            3 => Self::Triangle,
            4 => Self::Noise,
            5 => Self::Pulse,
            _ => Self::Sine,
        }
    }
}

/// `Lcg` matches the C++ `randomSeed = randomSeed * 1103515245 + 12345`
/// pseudo-random source from the original SynthEngine. The lower bits are
/// dropped (>>16) and a 15-bit slice is taken to reproduce the exact
/// sample-by-sample output, which the golden WAV tests rely on.
#[derive(Debug, Clone, Copy)]
pub struct Lcg(pub u32);

impl Lcg {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    /// Step state and return a signed value in `[-1, 1)`.
    pub fn next_signed(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((self.0 / 65_536) % 32_768) as f32 / 16_384.0 - 1.0
    }

    /// Step state and return an unsigned value in `[0, 1)`.
    pub fn next_unit(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((self.0 / 65_536) % 32_768) as f32 / 32_768.0
    }

    pub fn next_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_unit() * (max - min)
    }
}

/// Compute one sample of `wave` at the given (unwrapped) phase in radians.
///
/// `noise_rng` is consulted for `Waveform::Noise`; pass a dedicated `Lcg`
/// when you need reproducibility.
pub fn waveform_sample(wave: Waveform, phase: f32, pulse_width: f32, noise_rng: &mut Lcg) -> f32 {
    let two_pi = 2.0 * PI;
    match wave {
        Waveform::Sine => phase.sin(),
        Waveform::Square => {
            if phase.rem_euclid(two_pi) < PI {
                1.0
            } else {
                -1.0
            }
        }
        Waveform::Sawtooth => {
            let normalized = phase / two_pi;
            2.0 * (normalized - (normalized + 0.5).floor())
        }
        Waveform::Triangle => {
            let t = (phase / two_pi).rem_euclid(1.0);
            4.0 * (t - 0.5).abs() - 1.0
        }
        Waveform::Noise => noise_rng.next_signed(),
        Waveform::Pulse => {
            let t = phase.rem_euclid(two_pi) / two_pi;
            if t < pulse_width { 1.0 } else { -1.0 }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_is_zero_at_origin() {
        let mut rng = Lcg::new(0);
        let s = waveform_sample(Waveform::Sine, 0.0, 0.5, &mut rng);
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn square_alternates() {
        let mut rng = Lcg::new(0);
        let lo = waveform_sample(Waveform::Square, 0.1, 0.5, &mut rng);
        let hi = waveform_sample(Waveform::Square, PI + 0.1, 0.5, &mut rng);
        assert_eq!(lo, 1.0);
        assert_eq!(hi, -1.0);
    }

    #[test]
    fn triangle_peaks_at_half_period() {
        let mut rng = Lcg::new(0);
        // At phase = pi the triangle reaches its negative peak (-1) under
        // the formula `4*|t - 0.5| - 1` with t = 0.5.
        let s = waveform_sample(Waveform::Triangle, PI, 0.5, &mut rng);
        assert!((s + 1.0).abs() < 1e-6, "expected -1.0, got {s}");
    }

    #[test]
    fn sawtooth_in_range() {
        let mut rng = Lcg::new(0);
        for k in 0..200 {
            let phase = k as f32 * 0.1;
            let s = waveform_sample(Waveform::Sawtooth, phase, 0.5, &mut rng);
            assert!(
                s >= -1.0 && s <= 1.0,
                "sawtooth out of range at phase {phase}: {s}"
            );
        }
    }

    #[test]
    fn lcg_matches_cpp_first_samples() {
        // Reproduces the exact sequence the C++ implementation would
        // produce starting from seed 12345 — this is the contract the
        // WAV golden tests depend on.
        let mut rng = Lcg::new(12_345);
        let a = rng.next_signed();
        let b = rng.next_signed();
        let c = rng.next_signed();
        // Compute the same values by hand-stepping the C formula.
        let mut s: u32 = 12_345u32
            .wrapping_mul(1_103_515_245)
            .wrapping_add(12_345);
        let expect_a = ((s / 65_536) % 32_768) as f32 / 16_384.0 - 1.0;
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let expect_b = ((s / 65_536) % 32_768) as f32 / 16_384.0 - 1.0;
        s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        let expect_c = ((s / 65_536) % 32_768) as f32 / 16_384.0 - 1.0;
        assert_eq!(a, expect_a);
        assert_eq!(b, expect_b);
        assert_eq!(c, expect_c);
    }
}
