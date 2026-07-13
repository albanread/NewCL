//! Synth core property tests.

use newaudio_core::{
    Adsr, Buffer, Config, Engine, WavParams, frequency_to_note, note_to_frequency, write_wav,
};
use proptest::prelude::*;

proptest! {
    /// ADSR value stays inside `[0, 1]` for any non-pathological inputs.
    #[test]
    fn adsr_is_bounded(
        attack in 0.001f32..1.0,
        decay  in 0.001f32..1.0,
        sustain in 0.0f32..=1.0,
        release in 0.001f32..1.0,
        duration in 0.05f32..5.0,
        t in 0.0f32..6.0,
    ) {
        let env = Adsr { attack, decay, sustain, release };
        let v = env.value_at(t, duration);
        prop_assert!((-1e-6..=1.0 + 1e-6).contains(&v), "adsr out of range: {v}");
    }

    /// MIDI ↔ frequency round-trip is exact for the playable range.
    #[test]
    fn midi_roundtrip_in_range(midi in 21i32..=108i32) {
        let f = note_to_frequency(midi);
        prop_assert_eq!(frequency_to_note(f), midi);
    }

    /// WAV output length always equals `44 + samples * 2` bytes (PCM16).
    #[test]
    fn wav_size_predictable(frames in 1usize..=1000, channels in 1u32..=2u32) {
        let buf = Buffer {
            sample_rate: 44_100,
            channels,
            duration: frames as f32 / 44_100.0,
            samples: vec![0.0; frames * channels as usize],
        };
        let mut out = Vec::new();
        write_wav(&mut out, &buf, WavParams::default()).unwrap();
        prop_assert_eq!(out.len(), 44 + frames * channels as usize * 2);
    }

    /// Rendering the same preset twice with the same seed produces
    /// identical samples — the foundation our golden-byte tests rely on.
    #[test]
    fn engine_is_deterministic(seed in 0u32..1000u32) {
        let cfg = Config::default();
        let mut a = Engine::with_seed(cfg, seed);
        let mut b = Engine::with_seed(cfg, seed);
        prop_assert_eq!(
            a.coin(1.0, 0.05).samples,
            b.coin(1.0, 0.05).samples
        );
    }
}
