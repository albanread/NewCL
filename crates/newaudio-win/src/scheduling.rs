//! Tempo-aware flattening of MIDI tracks. Ported from the `flattenTracks`
//! and `buildTempoSegments` helpers in the original Zig runtime so timing
//! is preserved bit-for-bit. The output is a wall-clock-ordered list of
//! short MIDI messages annotated with the *source* channel (which the
//! runtime later maps onto an actually-free hardware channel).

use newaudio_abc::{AbcTune, EventKind, MidiGenerator, MidiTrack};

#[derive(Debug, Clone, Copy)]
pub struct ScheduledEvent {
    pub time_ns: u64,
    pub source_channel: u8,
    pub status: u8,
    pub data1: u8,
    pub data2: u8,
    /// Priority within identical `time_ns` (tempo < note_off < cc < pc < note_on).
    pub order: u8,
}

#[derive(Debug, Clone, Copy)]
struct TempoSegment {
    start_beat: f64,
    start_ns: f64,
    ns_per_beat: f64,
}

fn event_priority(kind: EventKind) -> u8 {
    match kind {
        EventKind::MetaTempo => 0,
        EventKind::NoteOff => 1,
        EventKind::ControlChange => 2,
        EventKind::ProgramChange => 3,
        EventKind::NoteOn => 4,
        _ => 5,
    }
}

fn extract_tempo_bpm(meta_data: &[u8], fallback: i32) -> i32 {
    if meta_data.len() < 3 {
        return fallback;
    }
    let mpq = ((meta_data[0] as u32) << 16)
        | ((meta_data[1] as u32) << 8)
        | meta_data[2] as u32;
    if mpq == 0 {
        return fallback;
    }
    ((60_000_000 / mpq as i64).max(1)) as i32
}

fn build_tempo_segments(default_bpm: i32, tempo_track: Option<&MidiTrack>) -> Vec<TempoSegment> {
    let mut segments = Vec::new();
    let mut current_bpm = default_bpm.max(1);
    let mut current_start_beat = 0.0_f64;
    let mut current_start_ns = 0.0_f64;

    if let Some(track) = tempo_track {
        for ev in &track.events {
            if !matches!(ev.kind, EventKind::MetaTempo) {
                continue;
            }
            let beat = ev.timestamp.max(0.0);
            if beat > current_start_beat {
                segments.push(TempoSegment {
                    start_beat: current_start_beat,
                    start_ns: current_start_ns,
                    ns_per_beat: 60_000_000_000.0 / current_bpm as f64,
                });
                current_start_ns +=
                    (beat - current_start_beat) * (60_000_000_000.0 / current_bpm as f64);
                current_start_beat = beat;
            }
            current_bpm = extract_tempo_bpm(&ev.meta_data, current_bpm);
        }
    }
    segments.push(TempoSegment {
        start_beat: current_start_beat,
        start_ns: current_start_ns,
        ns_per_beat: 60_000_000_000.0 / current_bpm as f64,
    });
    segments
}

fn beat_to_ns(beat: f64, segments: &[TempoSegment]) -> u64 {
    let mut chosen = segments[0];
    for seg in segments {
        if seg.start_beat > beat {
            break;
        }
        chosen = *seg;
    }
    let ns = chosen.start_ns + (beat - chosen.start_beat) * chosen.ns_per_beat;
    ns.max(0.0) as u64
}

/// Compile a tune to a sorted list of short MIDI messages. The returned
/// events are wall-clock-ordered (with ties broken by event priority,
/// source channel, status, data1, data2 — same as the Zig reference).
pub fn compile_asset(tune: &AbcTune) -> (Vec<ScheduledEvent>, u16) {
    let mut mgen = MidiGenerator::new();
    let mut tracks = mgen.generate(tune);

    let tempo_track = tracks.first().cloned();
    let segments = build_tempo_segments(tune.default_tempo.bpm, tempo_track.as_ref());

    // Sort events within each track by (timestamp, priority) so the
    // delta-time logic is consistent with how the SMF writer orders them.
    for track in tracks.iter_mut() {
        track.events.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| event_priority(a.kind).cmp(&event_priority(b.kind)))
        });
    }

    let mut events: Vec<ScheduledEvent> = Vec::new();
    for track in &tracks {
        for ev in &track.events {
            if ev.channel < 0 || ev.channel > 15 {
                continue;
            }
            let status: u8 = match ev.kind {
                EventKind::NoteOn => 0x90,
                EventKind::NoteOff => 0x80,
                EventKind::ProgramChange => 0xC0,
                EventKind::ControlChange => 0xB0,
                _ => continue,
            };
            events.push(ScheduledEvent {
                time_ns: beat_to_ns(ev.timestamp.max(0.0), &segments),
                source_channel: ev.channel as u8,
                status,
                data1: ev.data1,
                data2: ev.data2,
                order: event_priority(ev.kind),
            });
        }
    }

    events.sort_by(|a, b| {
        a.time_ns
            .cmp(&b.time_ns)
            .then_with(|| a.order.cmp(&b.order))
            .then_with(|| a.source_channel.cmp(&b.source_channel))
            .then_with(|| a.status.cmp(&b.status))
            .then_with(|| a.data1.cmp(&b.data1))
            .then_with(|| a.data2.cmp(&b.data2))
    });

    let mut mask: u16 = 0;
    for ev in &events {
        if ev.source_channel < 16 {
            mask |= 1 << ev.source_channel;
        }
    }
    (events, mask)
}

#[cfg(test)]
mod tests {
    use super::*;
    use newaudio_abc::AbcParser;

    #[test]
    fn compiles_simple_scale_to_chronological_events() {
        let abc = "X:1\nT:Scale\nL:1/4\nQ:60\nK:C\nCDEF|\n";
        let mut p = AbcParser::new();
        let tune = p.parse(abc).unwrap();
        let (events, mask) = compile_asset(&tune);

        // 4 NoteOn + 4 NoteOff + 1 ProgramChange = 9 messages.
        assert_eq!(
            events.iter().filter(|e| e.status == 0x90).count(),
            4
        );
        assert_eq!(
            events.iter().filter(|e| e.status == 0x80).count(),
            4
        );
        assert_eq!(
            events.iter().filter(|e| e.status == 0xC0).count(),
            1
        );

        // At 60 BPM, each quarter note = 1 s = 1e9 ns. So the four NoteOns
        // should fire at 0, 1s, 2s, 3s.
        let note_on_times: Vec<u64> = events
            .iter()
            .filter(|e| e.status == 0x90)
            .map(|e| e.time_ns)
            .collect();
        assert_eq!(note_on_times, vec![0, 1_000_000_000, 2_000_000_000, 3_000_000_000]);

        // Single voice → channel 0 only.
        assert_eq!(mask, 0b1);
    }

    #[test]
    fn note_offs_align_with_durations() {
        let abc = "X:1\nL:1/4\nQ:60\nK:C\nC2D2|\n";
        let mut p = AbcParser::new();
        let tune = p.parse(abc).unwrap();
        let (events, _) = compile_asset(&tune);

        let off_times: Vec<u64> = events
            .iter()
            .filter(|e| e.status == 0x80)
            .map(|e| e.time_ns)
            .collect();
        // C2 starts at 0 and lasts 2 quarters (2 s); D2 starts at 2s and
        // lasts another 2 s.
        assert_eq!(off_times, vec![2_000_000_000, 4_000_000_000]);
    }

    #[test]
    fn ties_break_deterministically() {
        // Two notes at the same start; ordering should be channel asc.
        let abc = "X:1\nL:1/4\nQ:60\nK:C\n[CEG]|\n";
        let mut p = AbcParser::new();
        let tune = p.parse(abc).unwrap();
        let (events, _) = compile_asset(&tune);
        let on_at_zero: Vec<u8> = events
            .iter()
            .filter(|e| e.status == 0x90 && e.time_ns == 0)
            .map(|e| e.data1)
            .collect();
        // Pitches were authored in C-E-G order so the chord's NoteOns
        // should appear in that order.
        assert_eq!(on_at_zero, vec![48, 52, 55]);
    }
}
