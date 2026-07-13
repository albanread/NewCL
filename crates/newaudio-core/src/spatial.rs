//! Stereo positioning helpers.
//!
//! These are pure utility functions — they compute `(volume, pan)` pairs
//! that the runtime can feed into [`crate::engine`] / WinMM playback.
//! Designed for 2D games where the listener and the sound source live on
//! the same horizontal plane.

/// Output of [`pan_from_position`]: a volume in `[0, 1]` and a pan in
/// `[-1, 1]` (negative is left, positive is right).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanResult {
    pub volume: f32,
    pub pan: f32,
}

/// Compute volume + pan for a sound source at `source_x` heard by a
/// listener at `listener_x`.
///
///  - `max_distance` is the falloff radius. Beyond it the sound is silent.
///  - Volume falls off linearly with distance.
///  - Pan tracks the signed offset (right of listener → pan > 0).
///
/// Choosing simple linear falloff over inverse-square keeps timing and
/// volume curves predictable across game frames; if you want a more
/// realistic curve, post-process the returned `volume` yourself.
pub fn pan_from_position(listener_x: f32, source_x: f32, max_distance: f32) -> PanResult {
    if max_distance <= 0.0 {
        return PanResult {
            volume: if listener_x == source_x { 1.0 } else { 0.0 },
            pan: 0.0,
        };
    }
    let delta = source_x - listener_x;
    let dist = delta.abs();
    let volume = (1.0 - dist / max_distance).clamp(0.0, 1.0);
    let pan = (delta / max_distance).clamp(-1.0, 1.0);
    PanResult { volume, pan }
}

/// 2D variant: like [`pan_from_position`] but with Y/depth to bias the
/// volume curve. Pan is still horizontal-only.
pub fn pan_from_position_2d(
    listener: (f32, f32),
    source: (f32, f32),
    max_distance: f32,
) -> PanResult {
    if max_distance <= 0.0 {
        return PanResult {
            volume: if listener == source { 1.0 } else { 0.0 },
            pan: 0.0,
        };
    }
    let dx = source.0 - listener.0;
    let dy = source.1 - listener.1;
    let dist = (dx * dx + dy * dy).sqrt();
    let volume = (1.0 - dist / max_distance).clamp(0.0, 1.0);
    let pan = (dx / max_distance).clamp(-1.0, 1.0);
    PanResult { volume, pan }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn centered_source_is_full_volume_centre_pan() {
        let p = pan_from_position(0.0, 0.0, 10.0);
        assert_eq!(p.volume, 1.0);
        assert_eq!(p.pan, 0.0);
    }

    #[test]
    fn far_source_is_silent() {
        let p = pan_from_position(0.0, 100.0, 10.0);
        assert_eq!(p.volume, 0.0);
        // Pan is also clamped to right.
        assert!(p.pan > 0.99);
    }

    #[test]
    fn right_source_is_right_panned() {
        let p = pan_from_position(0.0, 5.0, 10.0);
        assert!(p.pan > 0.0);
        assert!((p.volume - 0.5).abs() < 1e-6);
    }

    #[test]
    fn left_source_is_left_panned() {
        let p = pan_from_position(0.0, -5.0, 10.0);
        assert!(p.pan < 0.0);
        assert!((p.volume - 0.5).abs() < 1e-6);
    }

    #[test]
    fn pan_2d_uses_distance() {
        // Source at (3, 4) → distance 5; with max_distance 10, volume 0.5.
        let p = pan_from_position_2d((0.0, 0.0), (3.0, 4.0), 10.0);
        assert!((p.volume - 0.5).abs() < 1e-6);
        // Horizontal offset 3 out of 10 → pan 0.3.
        assert!((p.pan - 0.3).abs() < 1e-6);
    }

    #[test]
    fn zero_max_distance_is_binary() {
        let here = pan_from_position(0.0, 0.0, 0.0);
        let there = pan_from_position(0.0, 1.0, 0.0);
        assert_eq!(here.volume, 1.0);
        assert_eq!(there.volume, 0.0);
    }
}
