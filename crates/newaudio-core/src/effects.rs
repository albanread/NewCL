//! In-place audio effects: filters, echo/delay, distortion.
//!
//! These mirror the helpers in the original `sound_runtime_win.cpp` and are
//! used by the `Engine::*_tone` builders below as well as exposed publicly
//! so callers can post-process arbitrary [`Buffer`]s.

use core::f32::consts::PI;

use crate::engine::Buffer;

/// Filter shape passed to [`apply_filter`] and the `filtered_*` builders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum FilterType {
    None = 0,
    LowPass = 1,
    HighPass = 2,
    BandPass = 3,
}

impl FilterType {
    pub fn from_code(code: i32) -> Self {
        match code {
            1 => Self::LowPass,
            2 => Self::HighPass,
            3 => Self::BandPass,
            _ => Self::None,
        }
    }
}

/// Single-pole RC low/high-pass (band-pass = average of both).
///
/// `cutoff_hz` is clamped to ≥ 20 Hz to avoid division by zero. `resonance`
/// is accepted for API compatibility but the RC model doesn't use it.
pub fn apply_filter(buf: &mut Buffer, kind: FilterType, cutoff_hz: f32, _resonance: f32) {
    if matches!(kind, FilterType::None) || buf.channels == 0 || cutoff_hz <= 0.0 {
        return;
    }
    let dt = 1.0 / buf.sample_rate as f32;
    let rc = 1.0 / (2.0 * PI * cutoff_hz.max(20.0));
    let low_alpha = dt / (rc + dt);
    let high_alpha = rc / (rc + dt);

    let channels = buf.channels as usize;
    let mut low_prev = vec![0.0_f32; channels];
    let mut high_prev = vec![0.0_f32; channels];
    let mut input_prev = vec![0.0_f32; channels];

    for frame in 0..buf.frame_count() {
        for ch in 0..channels {
            let idx = frame * channels + ch;
            let input = buf.samples[idx];
            let low = low_prev[ch] + low_alpha * (input - low_prev[ch]);
            let high = high_alpha * (high_prev[ch] + input - input_prev[ch]);
            low_prev[ch] = low;
            high_prev[ch] = high;
            input_prev[ch] = input;

            buf.samples[idx] = match kind {
                FilterType::LowPass => low,
                FilterType::HighPass => high,
                FilterType::BandPass => 0.5 * (low + high),
                FilterType::None => input,
            };
        }
    }
}

/// Multi-tap echo. `delay_secs` is the per-tap delay; `feedback` is clamped
/// to `[0, 0.95]` and `mix` to `[0, 1]`. Output is peak-normalised so
/// stacked echoes don't clip.
pub fn apply_echo(buf: &mut Buffer, delay_secs: f32, feedback: f32, mix: f32, taps: u32) {
    if delay_secs <= 0.0 || mix <= 0.0 || taps == 0 {
        return;
    }
    let delay_frames = (delay_secs * buf.sample_rate as f32) as usize;
    if delay_frames == 0 {
        return;
    }
    let original = buf.samples.clone();
    let fb = feedback.clamp(0.0, 0.95);
    let mix = mix.clamp(0.0, 1.0);
    let channels = buf.channels as usize;
    let total_frames = buf.frame_count();

    for tap in 1..=taps {
        let offset = delay_frames * tap as usize;
        let gain = fb.powi(tap as i32 - 1) * mix;
        if offset >= total_frames {
            break;
        }
        for frame in 0..total_frames.saturating_sub(offset) {
            for ch in 0..channels {
                let src = frame * channels + ch;
                let dst = (frame + offset) * channels + ch;
                buf.samples[dst] += original[src] * gain;
            }
        }
    }
    crate::engine::normalize(buf, 0.95);
}

/// Soft-clip distortion. `drive` adds gain into a `tanh` non-linearity;
/// `level` is a post-gain. Output is peak-normalised.
pub fn apply_distortion(buf: &mut Buffer, drive: f32, level: f32) {
    let eff_drive = (1.0 + drive * 8.0).max(1.0);
    let lvl = level.max(0.0);
    for s in &mut buf.samples {
        *s = (*s * eff_drive).tanh() * lvl;
    }
    crate::engine::normalize(buf, 0.95);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Config, Engine};

    fn impulse_buffer() -> Buffer {
        let mut s = vec![0.0_f32; 44_100];
        s[0] = 1.0;
        Buffer {
            sample_rate: 44_100,
            channels: 1,
            duration: 1.0,
            samples: s,
        }
    }

    #[test]
    fn filter_none_is_passthrough() {
        let mut b = impulse_buffer();
        let expect = b.samples.clone();
        apply_filter(&mut b, FilterType::None, 1000.0, 1.0);
        assert_eq!(b.samples, expect);
    }

    #[test]
    fn lowpass_attenuates_impulse_tail() {
        let mut b = impulse_buffer();
        apply_filter(&mut b, FilterType::LowPass, 1000.0, 1.0);
        // After a single-pole LPF, the impulse smears: first sample < 1.0,
        // and later samples should be non-zero (decaying).
        assert!(b.samples[0] < 1.0);
        assert!(b.samples[1] > 0.0);
        assert!(b.samples[10] > 0.0);
    }

    #[test]
    fn echo_zero_taps_is_passthrough() {
        let mut b = impulse_buffer();
        let expect = b.samples.clone();
        apply_echo(&mut b, 0.1, 0.5, 0.5, 0);
        assert_eq!(b.samples, expect);
    }

    #[test]
    fn echo_adds_delayed_copies() {
        let mut b = impulse_buffer();
        apply_echo(&mut b, 0.01, 0.5, 1.0, 2);
        let delay_frames = (0.01 * 44_100.0) as usize;
        assert!(b.samples[delay_frames] > 0.0);
        assert!(b.samples[delay_frames * 2] > 0.0);
        // Second tap is feedback*1 = 0.5 of first tap.
        assert!(b.samples[delay_frames * 2] < b.samples[delay_frames]);
    }

    #[test]
    fn distortion_clamps_into_tanh_range() {
        let mut b = Buffer {
            sample_rate: 44_100,
            channels: 1,
            duration: 1e-3,
            samples: vec![10.0, -10.0, 5.0, -5.0],
        };
        apply_distortion(&mut b, 1.0, 1.0);
        let peak = b.samples.iter().fold(0.0_f32, |a, &s| a.max(s.abs()));
        assert!(peak <= 1.0 + 1e-6, "peak after distortion = {peak}");
    }

    #[test]
    fn filter_is_deterministic() {
        let mut e1 = Engine::with_seed(Config::default(), 1);
        let mut e2 = Engine::with_seed(Config::default(), 1);
        let mut a = e1.beep(440.0, 0.05);
        let mut b = e2.beep(440.0, 0.05);
        apply_filter(&mut a, FilterType::LowPass, 800.0, 1.0);
        apply_filter(&mut b, FilterType::LowPass, 800.0, 1.0);
        assert_eq!(a.samples, b.samples);
    }
}
