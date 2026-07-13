//! MIDI event generation from a parsed [`AbcTune`].
//!
//! Timestamps on tracks are in **beats**, where one beat is the time
//! signature's denominator note (e.g. quarter-note in 4/4). Whole-note
//! timestamps stored on features are converted with the voice's local
//! time signature.

use std::collections::HashMap;

use crate::fraction::Fraction;
use crate::types::*;

/// Per-event ordering used when writing a track. Lower values are emitted
/// first when timestamps tie.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    Notes,
    Tempo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    NoteOn,
    NoteOff,
    ProgramChange,
    ControlChange,
    MetaTempo,
    MetaTimeSignature,
    MetaKeySignature,
    MetaText,
    MetaEndOfTrack,
}

#[derive(Debug, Clone)]
pub struct MidiEvent {
    pub kind: EventKind,
    /// In beats (time-signature-denominator units).
    pub timestamp: f64,
    /// 0..15. `-1` is allowed only for meta events that have no channel.
    pub channel: i8,
    pub data1: u8,
    pub data2: u8,
    pub meta_data: Vec<u8>,
}

impl MidiEvent {
    pub fn new(kind: EventKind, timestamp: f64, channel: i8, data1: u8, data2: u8) -> Self {
        Self {
            kind,
            timestamp,
            channel,
            data1,
            data2,
            meta_data: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MidiTrack {
    pub track_number: i32,
    pub kind: TrackType,
    pub voice_number: i32,
    pub name: String,
    pub channel: i8,
    pub events: Vec<MidiEvent>,
}

impl MidiTrack {
    pub fn new(track_number: i32, kind: TrackType) -> Self {
        Self {
            track_number,
            kind,
            voice_number: 0,
            name: String::new(),
            channel: -1,
            events: Vec::new(),
        }
    }
}

#[derive(Debug)]
pub struct MidiGenerator {
    pub ticks_per_quarter: i32,
    pub default_tempo: i32,
    pub default_velocity: u8,
    channels_in_use: [bool; 16],
    voice_to_channel: HashMap<i32, i8>,
    next_available_channel: i8,
    /// Pending NoteOff events keyed by source channel — flushed when
    /// their `end_time` is reached.
    active_notes: Vec<ActiveNote>,
}

#[derive(Debug, Clone, Copy)]
struct ActiveNote {
    midi_note: u8,
    channel: i8,
    end_time: f64,
}

impl Default for MidiGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl MidiGenerator {
    pub fn new() -> Self {
        Self {
            ticks_per_quarter: 480,
            default_tempo: 120,
            default_velocity: 80,
            channels_in_use: [false; 16],
            voice_to_channel: HashMap::new(),
            next_available_channel: 0,
            active_notes: Vec::new(),
        }
    }

    pub fn reset(&mut self) {
        self.channels_in_use = [false; 16];
        self.voice_to_channel.clear();
        self.next_available_channel = 0;
        self.active_notes.clear();
    }

    pub fn generate(&mut self, tune: &AbcTune) -> Vec<MidiTrack> {
        self.reset();
        let mut tracks = Vec::new();
        self.create_tracks(tune, &mut tracks);
        self.assign_channels(tune, &mut tracks);
        let voice_ids: Vec<i32> = tracks
            .iter()
            .filter(|t| t.track_number != 0)
            .map(|t| t.voice_number)
            .collect();
        for voice_id in voice_ids {
            let idx = tracks
                .iter()
                .position(|t| t.voice_number == voice_id && t.track_number != 0)
                .unwrap();
            self.generate_track_events(tune, &mut tracks, idx);
        }
        if !tracks.is_empty() {
            self.process_tempo_track(tune, &mut tracks[0]);
        }
        tracks
    }

    fn create_tracks(&mut self, tune: &AbcTune, tracks: &mut Vec<MidiTrack>) {
        let mut tempo_track = MidiTrack::new(0, TrackType::Tempo);
        tempo_track.name = "Tempo Track".to_owned();
        tracks.push(tempo_track);

        let mut next = 1;
        for (_, voice) in &tune.voices {
            let mut t = MidiTrack::new(next, TrackType::Notes);
            next += 1;
            t.voice_number = voice.id;
            t.name = if voice.name.is_empty() {
                "Voice".to_owned()
            } else {
                voice.name.clone()
            };
            tracks.push(t);
        }
    }

    fn assign_channels(&mut self, tune: &AbcTune, tracks: &mut [MidiTrack]) {
        // Pass 1: explicit channels.
        for t in tracks.iter_mut() {
            if t.track_number == 0 {
                continue;
            }
            if let Some(v) = tune.voices.get(&t.voice_number) {
                if v.channel >= 0 {
                    t.channel = v.channel;
                    let ch = v.channel as usize;
                    self.channels_in_use[ch] = true;
                    self.voice_to_channel.insert(t.voice_number, v.channel);
                }
            }
        }

        // Pass 2: auto-assign the rest.
        for t in tracks.iter_mut() {
            if t.track_number == 0 || t.channel >= 0 {
                continue;
            }
            t.channel = self.assign_channel(t.voice_number);
        }

        // Pass 3: emit program changes at t=0 for every voice.
        for t in tracks.iter_mut() {
            if t.track_number == 0 {
                continue;
            }
            if let Some(v) = tune.voices.get(&t.voice_number) {
                let pc = MidiEvent::new(EventKind::ProgramChange, 0.0, t.channel, v.instrument, 0);
                t.events.push(pc);
            }
        }
    }

    fn assign_channel(&mut self, voice_id: i32) -> i8 {
        if let Some(&ch) = self.voice_to_channel.get(&voice_id) {
            return ch;
        }
        while self.next_available_channel < 16 {
            let ch = self.next_available_channel;
            if ch != 9 && !self.channels_in_use[ch as usize] {
                self.channels_in_use[ch as usize] = true;
                self.voice_to_channel.insert(voice_id, ch);
                self.next_available_channel += 1;
                return ch;
            }
            self.next_available_channel += 1;
        }
        0
    }

    fn generate_track_events(
        &mut self,
        tune: &AbcTune,
        tracks: &mut [MidiTrack],
        track_idx: usize,
    ) {
        self.active_notes.clear();
        let mut max_end_time: f64 = 0.0;
        let voice_id = tracks[track_idx].voice_number;
        let channel = tracks[track_idx].channel;

        // Collect features for this voice first to avoid borrow gymnastics.
        let voice_features: Vec<&Feature> = tune
            .features
            .iter()
            .filter(|f| f.voice_id == voice_id)
            .collect();

        let voice = match tune.voices.get(&voice_id) {
            Some(v) => v.clone(),
            None => return,
        };

        for feature in voice_features {
            let ts = timestamp_to_beats(feature.ts, &voice);
            match &feature.data {
                FeatureData::Note(n) => {
                    self.flush_due_note_offs(ts, &mut tracks[track_idx]);
                    tracks[track_idx].events.push(MidiEvent::new(
                        EventKind::NoteOn,
                        ts,
                        channel,
                        n.midi_note,
                        n.velocity,
                    ));
                    let end_time = ts + duration_to_beats(n.duration, &voice);
                    self.active_notes.push(ActiveNote {
                        midi_note: n.midi_note,
                        channel,
                        end_time,
                    });
                    if end_time > max_end_time {
                        max_end_time = end_time;
                    }
                }
                FeatureData::Rest(r) => {
                    let end_time = ts + duration_to_beats(r.duration, &voice);
                    self.flush_due_note_offs(end_time, &mut tracks[track_idx]);
                    if end_time > max_end_time {
                        max_end_time = end_time;
                    }
                }
                FeatureData::Chord(c) => {
                    self.flush_due_note_offs(ts, &mut tracks[track_idx]);
                    let end_time = ts + duration_to_beats(c.duration, &voice);
                    for n in &c.notes {
                        tracks[track_idx].events.push(MidiEvent::new(
                            EventKind::NoteOn,
                            ts,
                            channel,
                            n.midi_note,
                            n.velocity,
                        ));
                        self.active_notes.push(ActiveNote {
                            midi_note: n.midi_note,
                            channel,
                            end_time,
                        });
                    }
                    if end_time > max_end_time {
                        max_end_time = end_time;
                    }
                }
                _ => {}
            }
        }
        self.flush_all_note_offs(&mut tracks[track_idx]);
        let eot_time = max_end_time;
        tracks[track_idx].events.push(MidiEvent::new(
            EventKind::MetaEndOfTrack,
            eot_time,
            0,
            0,
            0,
        ));
    }

    fn flush_due_note_offs(&mut self, current_time: f64, track: &mut MidiTrack) {
        let mut i = 0;
        while i < self.active_notes.len() {
            if self.active_notes[i].end_time <= current_time {
                let an = self.active_notes.remove(i);
                track.events.push(MidiEvent::new(
                    EventKind::NoteOff,
                    an.end_time,
                    an.channel,
                    an.midi_note,
                    0,
                ));
            } else {
                i += 1;
            }
        }
    }

    fn flush_all_note_offs(&mut self, track: &mut MidiTrack) {
        for an in self.active_notes.drain(..) {
            track.events.push(MidiEvent::new(
                EventKind::NoteOff,
                an.end_time,
                an.channel,
                an.midi_note,
                0,
            ));
        }
    }

    fn process_tempo_track(&self, tune: &AbcTune, track: &mut MidiTrack) {
        push_tempo(track, tune.default_tempo.bpm, 0.0);
        push_time_signature(
            track,
            tune.default_timesig.num,
            tune.default_timesig.denom,
            0.0,
        );
        push_key_signature(
            track,
            tune.default_key.sharps,
            tune.default_key.is_major,
            0.0,
        );
        if !tune.title.is_empty() {
            push_text(track, &tune.title, 0.0);
        }

        let mut max_end_time = 0.0_f64;
        for feature in &tune.features {
            let voice = tune.voices.get(&feature.voice_id);
            let ts_beats = match voice {
                Some(v) => timestamp_to_beats(feature.ts, v),
                None => feature.ts * tune.default_timesig.denom as f64,
            };
            match &feature.data {
                FeatureData::Tempo(t) => push_tempo(track, t.bpm, ts_beats),
                FeatureData::Time(t) => push_time_signature(track, t.num, t.denom, ts_beats),
                FeatureData::Key(k) => push_key_signature(track, k.sharps, k.is_major, ts_beats),
                _ => {}
            }
            let feature_end = match &feature.data {
                FeatureData::Note(n) => {
                    ts_beats + duration_to_beats(n.duration, voice.unwrap())
                }
                FeatureData::Rest(r) => {
                    ts_beats + duration_to_beats(r.duration, voice.unwrap())
                }
                FeatureData::Chord(c) => {
                    ts_beats + duration_to_beats(c.duration, voice.unwrap())
                }
                FeatureData::GChord(g) => {
                    ts_beats + duration_to_beats(g.duration, voice.unwrap())
                }
                _ => ts_beats,
            };
            if feature_end > max_end_time {
                max_end_time = feature_end;
            }
        }
        track.events.push(MidiEvent::new(
            EventKind::MetaEndOfTrack,
            max_end_time,
            0,
            0,
            0,
        ));
    }
}

fn duration_to_beats(d: Fraction, voice: &VoiceContext) -> f64 {
    d.to_f64() * voice.timesig.denom as f64
}

fn timestamp_to_beats(ts: f64, voice: &VoiceContext) -> f64 {
    ts * voice.timesig.denom as f64
}

fn push_tempo(track: &mut MidiTrack, bpm: i32, ts: f64) {
    let mpq: u32 = (60_000_000 / bpm.max(1)) as u32;
    let mut ev = MidiEvent::new(EventKind::MetaTempo, ts, 0, 0, 0);
    ev.meta_data.push(((mpq >> 16) & 0xFF) as u8);
    ev.meta_data.push(((mpq >> 8) & 0xFF) as u8);
    ev.meta_data.push((mpq & 0xFF) as u8);
    track.events.push(ev);
}

fn push_time_signature(track: &mut MidiTrack, num: u8, denom: u8, ts: f64) {
    let mut ev = MidiEvent::new(EventKind::MetaTimeSignature, ts, 0, 0, 0);
    let mut denom_power: u8 = 0;
    let mut t = denom;
    while t > 1 {
        t /= 2;
        denom_power += 1;
    }
    ev.meta_data.push(num);
    ev.meta_data.push(denom_power);
    ev.meta_data.push(24);
    ev.meta_data.push(8);
    track.events.push(ev);
}

fn push_key_signature(track: &mut MidiTrack, sharps: i8, is_major: bool, ts: f64) {
    let mut ev = MidiEvent::new(EventKind::MetaKeySignature, ts, 0, 0, 0);
    ev.meta_data.push(sharps as u8);
    ev.meta_data.push(if is_major { 0 } else { 1 });
    track.events.push(ev);
}

fn push_text(track: &mut MidiTrack, text: &str, ts: f64) {
    let mut ev = MidiEvent::new(EventKind::MetaText, ts, 0, 0, 0);
    ev.meta_data.extend_from_slice(text.as_bytes());
    track.events.push(ev);
}

// ---------------------------------------------------------------------------
// SMF (.mid) file writer
// ---------------------------------------------------------------------------

pub fn write_smf(tracks: &mut [MidiTrack], ticks_per_quarter: i32) -> Vec<u8> {
    let mut out = Vec::new();
    // Header.
    out.extend_from_slice(b"MThd");
    out.extend_from_slice(&6u32.to_be_bytes());
    out.extend_from_slice(&1u16.to_be_bytes()); // format 1
    out.extend_from_slice(&(tracks.len() as u16).to_be_bytes());
    out.extend_from_slice(&(ticks_per_quarter as u16).to_be_bytes());

    for track in tracks.iter_mut() {
        write_track(&mut out, track, ticks_per_quarter);
    }
    out
}

fn write_track(out: &mut Vec<u8>, track: &mut MidiTrack, ticks_per_quarter: i32) {
    // Sort by (timestamp, priority) to match the Zig reference.
    track.events.sort_by(|a, b| {
        a.timestamp
            .partial_cmp(&b.timestamp)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| event_priority(a.kind).cmp(&event_priority(b.kind)))
    });

    let mut track_data = Vec::new();
    let mut last_time = 0.0_f64;
    for ev in &track.events {
        let delta = ((ev.timestamp - last_time) * ticks_per_quarter as f64) as i32;
        let delta = delta.max(0) as u32;
        write_var_len(&mut track_data, delta);
        match ev.kind {
            EventKind::NoteOn => {
                track_data.push(0x90 | (ev.channel as u8 & 0x0F));
                track_data.push(ev.data1);
                track_data.push(ev.data2);
            }
            EventKind::NoteOff => {
                track_data.push(0x80 | (ev.channel as u8 & 0x0F));
                track_data.push(ev.data1);
                track_data.push(ev.data2);
            }
            EventKind::ProgramChange => {
                track_data.push(0xC0 | (ev.channel as u8 & 0x0F));
                track_data.push(ev.data1);
            }
            EventKind::ControlChange => {
                track_data.push(0xB0 | (ev.channel as u8 & 0x0F));
                track_data.push(ev.data1);
                track_data.push(ev.data2);
            }
            EventKind::MetaTempo => {
                track_data.push(0xFF);
                track_data.push(0x51);
                write_var_len(&mut track_data, ev.meta_data.len() as u32);
                track_data.extend_from_slice(&ev.meta_data);
            }
            EventKind::MetaTimeSignature => {
                track_data.push(0xFF);
                track_data.push(0x58);
                write_var_len(&mut track_data, ev.meta_data.len() as u32);
                track_data.extend_from_slice(&ev.meta_data);
            }
            EventKind::MetaKeySignature => {
                track_data.push(0xFF);
                track_data.push(0x59);
                write_var_len(&mut track_data, ev.meta_data.len() as u32);
                track_data.extend_from_slice(&ev.meta_data);
            }
            EventKind::MetaText => {
                track_data.push(0xFF);
                track_data.push(0x01);
                write_var_len(&mut track_data, ev.meta_data.len() as u32);
                track_data.extend_from_slice(&ev.meta_data);
            }
            EventKind::MetaEndOfTrack => {
                track_data.push(0xFF);
                track_data.push(0x2F);
                track_data.push(0x00);
            }
        }
        last_time = ev.timestamp;
    }

    out.extend_from_slice(b"MTrk");
    out.extend_from_slice(&(track_data.len() as u32).to_be_bytes());
    out.extend_from_slice(&track_data);
}

/// Standard MIDI variable-length quantity encoding (big-endian 7-bit groups).
pub fn write_var_len(out: &mut Vec<u8>, value: u32) {
    let mut buffer = [0u8; 4];
    let mut i = 0_usize;
    let mut v = value;

    buffer[i] = (v & 0x7F) as u8;
    v >>= 7;
    while v > 0 {
        i += 1;
        buffer[i] = ((v & 0x7F) | 0x80) as u8;
        v >>= 7;
    }
    loop {
        out.push(buffer[i]);
        if i == 0 {
            break;
        }
        i -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn var_len_single_byte_values() {
        let mut out = Vec::new();
        write_var_len(&mut out, 0);
        assert_eq!(out, vec![0x00]);
        out.clear();
        write_var_len(&mut out, 0x7F);
        assert_eq!(out, vec![0x7F]);
    }

    #[test]
    fn var_len_multi_byte_values() {
        // Reference values from the MIDI 1.0 spec table.
        let cases: &[(u32, &[u8])] = &[
            (0x80, &[0x81, 0x00]),
            (0x2000, &[0xC0, 0x00]),
            (0x3FFF, &[0xFF, 0x7F]),
            (0x4000, &[0x81, 0x80, 0x00]),
            (0x100000, &[0xC0, 0x80, 0x00]),
            (0x1FFFFF, &[0xFF, 0xFF, 0x7F]),
            (0x200000, &[0x81, 0x80, 0x80, 0x00]),
            (0x0FFFFFFF, &[0xFF, 0xFF, 0xFF, 0x7F]),
        ];
        for &(input, expected) in cases {
            let mut out = Vec::new();
            write_var_len(&mut out, input);
            assert_eq!(&out, expected, "var-len mismatch for 0x{input:X}");
        }
    }

    #[test]
    fn smf_header_is_well_formed() {
        let mut tune = AbcTune::new();
        let v = VoiceContext {
            id: 1,
            name: "1".to_owned(),
            key: tune.default_key,
            timesig: tune.default_timesig,
            unit_len: tune.default_unit,
            transpose: 0,
            octave_shift: 0,
            instrument: 0,
            channel: -1,
            velocity: 80,
            percussion: false,
        };
        tune.voices.insert(1, v);

        let mut mgen = MidiGenerator::new();
        let mut tracks = mgen.generate(&tune);
        let bytes = write_smf(&mut tracks, mgen.ticks_per_quarter);
        assert_eq!(&bytes[0..4], b"MThd");
        assert_eq!(&bytes[4..8], &6u32.to_be_bytes());
        // Format 1
        assert_eq!(&bytes[8..10], &1u16.to_be_bytes());
        // 1 tempo track + 1 voice track = 2.
        assert_eq!(u16::from_be_bytes([bytes[10], bytes[11]]), 2);
        // PPQ = 480
        assert_eq!(u16::from_be_bytes([bytes[12], bytes[13]]), 480);
        assert_eq!(&bytes[14..18], b"MTrk");
    }
}
