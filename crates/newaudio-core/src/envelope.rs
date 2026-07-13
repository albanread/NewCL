//! ADSR envelope.

/// Attack/Decay/Sustain/Release envelope. Times are in seconds, level is
/// normalized to `[0, 1]`. Semantics match the original `EnvelopeADSR` in
/// `SynthEngine.cpp`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Adsr {
    pub attack: f32,
    pub decay: f32,
    pub sustain: f32,
    pub release: f32,
}

impl Default for Adsr {
    fn default() -> Self {
        Self {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.7,
            release: 0.2,
        }
    }
}

impl Adsr {
    /// Envelope value at `time` for a note of total length `note_duration`.
    ///
    /// Layout: A → D → S (variable length) → R. If A+D+R exceeds the note,
    /// sustain phase has zero length but the curve still proceeds in order.
    pub fn value_at(&self, time: f32, note_duration: f32) -> f32 {
        if time < 0.0 {
            return 0.0;
        }

        let total = self.attack + self.decay + self.release;
        let sustain_time = (note_duration - total).max(0.0);

        let mut t = time;

        if t <= self.attack {
            return if self.attack <= 0.0 {
                1.0
            } else {
                t / self.attack
            };
        }
        t -= self.attack;

        if t <= self.decay {
            return if self.decay <= 0.0 {
                self.sustain
            } else {
                let f = t / self.decay;
                1.0 - f * (1.0 - self.sustain)
            };
        }
        t -= self.decay;

        if t <= sustain_time {
            return self.sustain;
        }
        t -= sustain_time;

        if t <= self.release {
            if self.release <= 0.0 {
                return 0.0;
            }
            return self.sustain * (1.0 - t / self.release);
        }

        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_time_is_silent_unless_attack_is_zero() {
        let env = Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.7,
            release: 0.1,
        };
        assert_eq!(env.value_at(0.0, 1.0), 0.0);
    }

    #[test]
    fn attack_peaks_at_one() {
        let env = Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.7,
            release: 0.1,
        };
        assert!((env.value_at(0.01, 1.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sustain_level_after_decay() {
        let env = Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.5,
            release: 0.1,
        };
        let v = env.value_at(0.01 + 0.05 + 0.001, 1.0);
        assert!((v - 0.5).abs() < 1e-3, "sustain returned {v}");
    }

    #[test]
    fn after_full_envelope_is_silent() {
        let env = Adsr {
            attack: 0.01,
            decay: 0.05,
            sustain: 0.5,
            release: 0.1,
        };
        let total = 0.01 + 0.05 + 0.1;
        let after = env.value_at(total + 0.01, total);
        assert!(after.abs() < 1e-6);
    }

    #[test]
    fn release_endpoint_is_zero() {
        let env = Adsr {
            attack: 0.0,
            decay: 0.0,
            sustain: 1.0,
            release: 0.2,
        };
        // sustain_time becomes (duration - release) = 0.3 for duration 0.5
        let v = env.value_at(0.5, 0.5);
        assert!(v.abs() < 1e-6, "expected ~0, got {v}");
    }

    #[test]
    fn negative_time_silent() {
        let env = Adsr::default();
        assert_eq!(env.value_at(-0.1, 1.0), 0.0);
    }
}
