//! ABC notation parser. Ported from `parser.zig` + `music_parser.zig` +
//! `voice_manager.zig`. Single-pass, builds an [`AbcTune`] feature stream
//! with timestamps in whole-note units that the MIDI generator consumes.

use std::collections::{BTreeMap, HashMap};

use crate::fraction::Fraction;
use crate::repeat::expand_abc_repeats;
use crate::types::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParseState {
    Header,
    Body,
    Complete,
}

pub struct AbcParser {
    state: ParseState,
    current_line: usize,
    // Voice manager
    pub current_voice: i32,
    next_voice_id: i32,
    voice_times: HashMap<i32, f64>,
    voice_name_to_id: HashMap<String, i32>,
    // Music parser
    current_time: f64,
    voice_bar_accidentals: HashMap<i32, [Option<i8>; 7]>,
    tuplet_num: i32,
    tuplet_denom: i32,
    tuplet_notes_remaining: i32,

    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl AbcParser {
    pub fn new() -> Self {
        Self {
            state: ParseState::Header,
            current_line: 0,
            current_voice: 1,
            next_voice_id: 1,
            voice_times: HashMap::new(),
            voice_name_to_id: HashMap::new(),
            current_time: 0.0,
            voice_bar_accidentals: HashMap::new(),
            tuplet_num: 1,
            tuplet_denom: 1,
            tuplet_notes_remaining: 0,
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    pub fn parse(&mut self, abc: &str) -> Result<AbcTune, String> {
        let mut tune = AbcTune::new();
        let ok = self.parse_into(abc, &mut tune);
        if !ok {
            return Err(self
                .errors
                .first()
                .cloned()
                .unwrap_or_else(|| "ABC parse failed".to_owned()));
        }
        Ok(tune)
    }

    pub fn parse_into(&mut self, abc: &str, tune: &mut AbcTune) -> bool {
        self.reset();
        let expanded = expand_abc_repeats(abc);
        let mut last_field: Option<u8> = None;

        for line in expanded.split('\n') {
            self.current_line += 1;
            let trimmed = line.trim_matches(|c: char| c == ' ' || c == '\t' || c == '\r');
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.starts_with("%%MIDI") && trimmed.len() >= 7 {
                self.parse_midi_directive(&trimmed[6..], tune);
                continue;
            }

            if trimmed.starts_with('%') {
                continue;
            }

            // Strip inline comments.
            let clean: &str = match trimmed.find('%') {
                Some(idx) => trimmed[..idx].trim_matches(|c: char| c == ' ' || c == '\t'),
                None => trimmed,
            };
            if clean.is_empty() {
                continue;
            }

            // `+:continuation`
            if clean.starts_with("+:") {
                if let Some(field) = last_field {
                    if is_string_field(field) {
                        let value = clean[2..].trim_matches(|c: char| c == ' ' || c == '\t');
                        let continued = format!("{}:{}", field as char, value);
                        self.parse_line(&continued, tune);
                    } else {
                        self.warnings.push(
                            "Ignoring +: continuation for non-string field".to_owned(),
                        );
                    }
                } else {
                    self.warnings
                        .push("Ignoring +: continuation with no previous field".to_owned());
                }
                continue;
            }

            if is_header_field(clean) {
                last_field = Some(clean.as_bytes()[0]);
            }

            self.parse_line(clean, tune);
        }

        if self.errors.is_empty() {
            if tune.voices.is_empty() {
                let v = VoiceContext {
                    id: 1,
                    name: "1".to_owned(),
                    key: tune.default_key,
                    timesig: tune.default_timesig,
                    unit_len: tune.default_unit,
                    transpose: 0,
                    octave_shift: 0,
                    instrument: tune.default_instrument,
                    channel: tune.default_channel,
                    velocity: 80,
                    percussion: tune.default_percussion,
                };
                tune.voices.insert(1, v);
            }
            self.state = ParseState::Complete;
        }

        self.errors.is_empty()
    }

    fn reset(&mut self) {
        self.state = ParseState::Header;
        self.current_line = 0;
        self.current_voice = 1;
        self.next_voice_id = 1;
        self.voice_times.clear();
        self.voice_name_to_id.clear();
        self.current_time = 0.0;
        self.voice_bar_accidentals.clear();
        self.tuplet_num = 1;
        self.tuplet_denom = 1;
        self.tuplet_notes_remaining = 0;
        self.errors.clear();
        self.warnings.clear();
    }

    fn parse_line(&mut self, line: &str, tune: &mut AbcTune) {
        // Standalone [V:...]
        if let Some(rest) = line.strip_prefix("[V:") {
            if let Some(end) = rest.find(']') {
                let after = rest[end + 1..].trim();
                if after.is_empty() {
                    let voice_id_str = rest[..end].trim();
                    if !voice_id_str.is_empty() {
                        self.save_current_time();
                        let new_voice = self.switch_to_voice(voice_id_str, tune);
                        self.current_time = self.restore_voice_time(new_voice);
                    }
                    return;
                }
            }
        }

        // V: field
        if line.starts_with("V:") {
            let voice_spec = line[2..].trim();
            let voice_id_str = voice_spec.split_whitespace().next().unwrap_or("");
            if !voice_id_str.is_empty() {
                let has_attributes = voice_spec.contains('=');
                if has_attributes {
                    self.parse_header_line(line, tune);
                    let found = self.find_voice_by_identifier(voice_id_str, tune);
                    if found != 0 {
                        self.register_external_voice(found, voice_id_str);
                        self.current_voice = found;
                    }
                } else {
                    self.save_current_time();
                    let new_voice = self.switch_to_voice(voice_id_str, tune);
                    self.current_time = self.restore_voice_time(new_voice);
                }
            }
            return;
        }

        if self.state == ParseState::Header && is_header_field(line) {
            self.parse_header_line(line, tune);
            if line.as_bytes()[0] == b'K' {
                self.state = ParseState::Body;
            }
            return;
        }

        if self.state == ParseState::Header {
            self.state = ParseState::Body;
        }

        if self.state == ParseState::Body {
            if tune.voices.is_empty() {
                let _ = self.switch_to_voice("1", tune);
            }
            self.parse_music_line(line, tune);
        }
    }

    fn parse_header_line(&mut self, line: &str, tune: &mut AbcTune) {
        let field = line.as_bytes()[0];
        let value = line[2..].trim_matches(|c: char| c == ' ' || c == '\t');

        match field {
            b'T' => append_string_field(&mut tune.title, value),
            b'H' => append_string_field(&mut tune.history, value),
            b'C' => append_string_field(&mut tune.composer, value),
            b'O' => append_string_field(&mut tune.origin, value),
            b'R' => append_string_field(&mut tune.rhythm, value),
            b'N' => append_string_field(&mut tune.notes, value),
            b'W' => append_string_field(&mut tune.words, value),
            b'w' => append_string_field(&mut tune.aligned_words, value),
            b'M' => tune.default_timesig = parse_meter(value).unwrap_or(tune.default_timesig),
            b'L' => {
                if let Some(f) = parse_simple_fraction(value) {
                    tune.default_unit = f;
                }
            }
            b'Q' => {
                let after = value.split('=').next_back().unwrap_or(value).trim();
                if let Ok(bpm) = after.parse::<i32>() {
                    tune.default_tempo = Tempo { bpm };
                }
            }
            b'K' => tune.default_key = parse_key_sig(value),
            b'V' => {
                let mut iter = value.split_whitespace();
                if let Some(voice_id_str) = iter.next() {
                    let voice_id = self.get_or_create_voice(voice_id_str, tune);
                    if let Some(voice) = tune.voices.get_mut(&voice_id) {
                        let attr_offset = voice_id_str.len().min(value.len());
                        let attrs = value[attr_offset..]
                            .trim_start_matches(|c: char| c == ' ' || c == '\t');
                        apply_voice_attributes(attrs, voice);
                    }
                }
            }
            _ => {}
        }
    }

    fn parse_midi_directive(&mut self, rest: &str, tune: &mut AbcTune) {
        let args = rest.trim();
        if args.is_empty() {
            return;
        }
        let mut iter = args.splitn(2, |c: char| c == ' ' || c == '\t');
        let sub = iter.next().unwrap_or("");
        let val_str = iter.next().unwrap_or("").trim();

        // If we have at least one voice, modify the current one; else stash
        // as defaults that lazily-created voices will pick up.
        let target_voice = if tune.voices.is_empty() {
            None
        } else {
            Some(self.current_voice)
        };

        let lower = sub.to_ascii_lowercase();
        match lower.as_str() {
            "program" => {
                if let Ok(p) = val_str.parse::<i32>() {
                    if (0..=127).contains(&p) {
                        match target_voice {
                            Some(id) => {
                                if let Some(v) = tune.voices.get_mut(&id) {
                                    v.instrument = p as u8;
                                }
                            }
                            None => tune.default_instrument = p as u8,
                        }
                    }
                }
            }
            "channel" => {
                if let Ok(ch) = val_str.parse::<i32>() {
                    if (1..=16).contains(&ch) {
                        let zero_based = (ch - 1) as i8;
                        match target_voice {
                            Some(id) => {
                                if let Some(v) = tune.voices.get_mut(&id) {
                                    v.channel = zero_based;
                                }
                            }
                            None => tune.default_channel = zero_based,
                        }
                    }
                }
            }
            "transpose" => {
                if let Ok(t) = val_str.parse::<i8>() {
                    if let Some(id) = target_voice {
                        if let Some(v) = tune.voices.get_mut(&id) {
                            v.transpose = t;
                        }
                    }
                }
            }
            "velocity" | "volume" => {
                if let Ok(v) = val_str.parse::<i32>() {
                    if (0..=127).contains(&v) {
                        if let Some(id) = target_voice {
                            if let Some(vc) = tune.voices.get_mut(&id) {
                                vc.velocity = v as u8;
                            }
                        }
                    }
                }
            }
            "drum" | "percussion" => {
                let on = val_str.is_empty()
                    || val_str.eq_ignore_ascii_case("on")
                    || val_str == "1";
                match target_voice {
                    Some(id) => {
                        if let Some(v) = tune.voices.get_mut(&id) {
                            v.percussion = on;
                            if on {
                                v.channel = 9;
                            }
                        }
                    }
                    None => {
                        tune.default_percussion = on;
                        if on {
                            tune.default_channel = 9;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    // -- Music body parsing ------------------------------------------------

    fn parse_music_line(&mut self, line: &str, tune: &mut AbcTune) {
        // Inline header field (e.g. `[K:G]` at start of bar)
        if line.len() >= 2 && line.as_bytes()[1] == b':' {
            self.apply_inline_field(line.as_bytes()[0], line[2..].trim(), tune);
            return;
        }
        self.parse_note_sequence(line, tune);
    }

    fn apply_inline_field(&mut self, field: u8, value: &str, tune: &mut AbcTune) {
        let voice_id = self.current_voice;
        // Voice may not exist if line came before any K:; the parser ensures one.
        let voice_present = tune.voices.contains_key(&voice_id);
        if !voice_present {
            return;
        }

        match field {
            b'Q' => {
                let after = value.split('=').next_back().unwrap_or(value).trim();
                if let Ok(bpm) = after.parse::<i32>() {
                    tune.default_tempo = Tempo { bpm };
                    tune.features.push(Feature {
                        voice_id,
                        ts: self.current_time,
                        line_number: self.current_line,
                        data: FeatureData::Tempo(Tempo { bpm }),
                    });
                }
            }
            b'M' => {
                let tsig = parse_meter(value).unwrap_or_else(|| {
                    tune.voices.get(&voice_id).map(|v| v.timesig).unwrap_or_default()
                });
                if let Some(v) = tune.voices.get_mut(&voice_id) {
                    v.timesig = tsig;
                }
                tune.features.push(Feature {
                    voice_id,
                    ts: self.current_time,
                    line_number: self.current_line,
                    data: FeatureData::Time(tsig),
                });
            }
            b'L' => {
                if let Some(f) = parse_simple_fraction(value) {
                    if let Some(v) = tune.voices.get_mut(&voice_id) {
                        v.unit_len = f;
                    }
                }
            }
            b'K' => {
                let key = parse_key_sig(value);
                if let Some(v) = tune.voices.get_mut(&voice_id) {
                    v.key = key;
                }
                tune.features.push(Feature {
                    voice_id,
                    ts: self.current_time,
                    line_number: self.current_line,
                    data: FeatureData::Key(key),
                });
            }
            _ => {}
        }
    }

    fn parse_note_sequence(&mut self, seq: &str, tune: &mut AbcTune) {
        let b = seq.as_bytes();
        let mut pos = 0;
        while pos < b.len() {
            pos = skip_ws(b, pos);
            if pos >= b.len() {
                break;
            }

            // Grace group {..}
            if b[pos] == b'{' {
                let _ = skip_grace(b, &mut pos);
                continue;
            }

            // Inline bracket field e.g. [K:G]
            if b[pos] == b'[' {
                if self.parse_inline_bracket(seq, &mut pos, tune) {
                    continue;
                }
            }

            // Guitar chord "Cm7"
            if b[pos] == b'"' {
                if let Some(mut gc) = parse_guitar_chord(b, &mut pos) {
                    let next = skip_ws(b, pos);
                    if next < b.len() && is_note_char(b[next]) {
                        let mut dummy = Note {
                            pitch: 0,
                            accidental: 0,
                            octave: 0,
                            duration: Fraction::new(0, 1),
                            midi_note: 0,
                            velocity: 0,
                            is_tied: false,
                        };
                        let mut tmp_pos = next;
                        if self.parse_note(b, &mut tmp_pos, &mut dummy, tune) {
                            gc.duration = dummy.duration;
                        } else {
                            gc.duration = tune.voices[&self.current_voice].unit_len;
                        }
                    } else {
                        gc.duration = tune.voices[&self.current_voice].unit_len;
                    }
                    let voice_id = self.current_voice;
                    tune.features.push(Feature {
                        voice_id,
                        ts: self.current_time,
                        line_number: self.current_line,
                        data: FeatureData::GChord(gc),
                    });
                    continue;
                }
            }

            // Chord [..]
            if b[pos] == b'[' {
                let mut chord_notes: Vec<Note> = Vec::new();
                let start_pos = pos;
                pos += 1;
                while pos < b.len() && b[pos] != b']' {
                    pos = skip_ws(b, pos);
                    if pos >= b.len() || b[pos] == b']' {
                        break;
                    }
                    let mut n = empty_note();
                    if self.parse_note(b, &mut pos, &mut n, tune) {
                        chord_notes.push(n);
                    } else if pos < b.len() && b[pos] != b']' {
                        pos += 1;
                    }
                    pos = skip_ws(b, pos);
                }
                if pos < b.len() && b[pos] == b']' {
                    pos += 1;
                    let duration =
                        parse_duration(b, &mut pos, tune.voices[&self.current_voice].unit_len);
                    if !chord_notes.is_empty() {
                        let chord = Chord {
                            notes: chord_notes,
                            duration,
                        };
                        let voice_id = self.current_voice;
                        let t = self.current_time;
                        tune.features.push(Feature {
                            voice_id,
                            ts: t,
                            line_number: self.current_line,
                            data: FeatureData::Chord(chord.clone()),
                        });
                        self.current_time += chord.duration.to_f64();
                    }
                } else {
                    pos = (start_pos + 1).min(b.len());
                }
                continue;
            }

            // Bar line
            if is_barline_char(b[pos]) {
                let start = pos;
                while pos < b.len() && is_barline_char(b[pos]) {
                    pos += 1;
                }
                let bar_str = &seq[start..pos];
                let bar_type = match bar_str {
                    "|" => BarType::Bar1,
                    "||" => BarType::DoubleBar,
                    "|:" => BarType::RepStart,
                    ":|" => BarType::RepEnd,
                    ":|:" => BarType::DoubleRep,
                    _ => BarType::Bar1,
                };
                self.reset_bar_accidentals(self.current_voice);
                let voice_id = self.current_voice;
                tune.features.push(Feature {
                    voice_id,
                    ts: self.current_time,
                    line_number: self.current_line,
                    data: FeatureData::Bar(BarLine { bar_type }),
                });
                continue;
            }

            // Rest
            if b[pos] == b'z' || b[pos] == b'Z' {
                pos += 1;
                let duration =
                    parse_duration(b, &mut pos, tune.voices[&self.current_voice].unit_len);
                let rest = Rest { duration };
                let voice_id = self.current_voice;
                tune.features.push(Feature {
                    voice_id,
                    ts: self.current_time,
                    line_number: self.current_line,
                    data: FeatureData::Rest(rest),
                });
                self.current_time += duration.to_f64();
                continue;
            }

            // Slur / tuplet
            if b[pos] == b'(' {
                if pos + 1 < b.len() && b[pos + 1].is_ascii_digit() {
                    let _ = self.parse_tuplet(b, &mut pos, tune);
                    continue;
                }
                pos += 1;
                continue;
            }
            if b[pos] == b')' {
                pos += 1;
                continue;
            }

            // Note (or accidental followed by note)
            if is_note_char(b[pos]) || is_accidental_marker(b[pos]) {
                let mut note = empty_note();
                if self.parse_note(b, &mut pos, &mut note, tune) {
                    // Broken rhythm
                    let mut has_broken = false;
                    let mut first_longer = false;
                    if pos < b.len() {
                        if b[pos] == b'>' {
                            has_broken = true;
                            first_longer = true;
                            pos += 1;
                        } else if b[pos] == b'<' {
                            has_broken = true;
                            pos += 1;
                        } else {
                            // Lookahead through grace groups
                            let mut look = pos;
                            while look < b.len() && skip_grace(b, &mut look) {
                                look = skip_ws(b, look);
                            }
                            if look < b.len() {
                                if b[look] == b'>' {
                                    has_broken = true;
                                    first_longer = true;
                                    pos = look + 1;
                                } else if b[look] == b'<' {
                                    has_broken = true;
                                    pos = look + 1;
                                }
                            }
                        }
                    }

                    if has_broken {
                        if first_longer {
                            note.duration = note.duration.mul(Fraction::new(3, 2));
                        } else {
                            note.duration = note.duration.mul(Fraction::new(1, 2));
                        }
                    }

                    self.apply_tuplet(&mut note.duration);

                    if pos < b.len() && b[pos] == b'-' {
                        note.is_tied = true;
                        pos += 1;
                    }
                    self.merge_tied_notes(b, &mut pos, &mut note, tune);

                    let voice_id = self.current_voice;
                    tune.features.push(Feature {
                        voice_id,
                        ts: self.current_time,
                        line_number: self.current_line,
                        data: FeatureData::Note(note),
                    });
                    self.current_time += note.duration.to_f64();

                    if has_broken {
                        pos = skip_ws(b, pos);
                        while pos < b.len() && skip_grace(b, &mut pos) {
                            pos = skip_ws(b, pos);
                        }
                        if pos < b.len() && is_note_char(b[pos]) {
                            let mut next_note = empty_note();
                            if self.parse_note(b, &mut pos, &mut next_note, tune) {
                                if first_longer {
                                    next_note.duration =
                                        next_note.duration.mul(Fraction::new(1, 2));
                                } else {
                                    next_note.duration =
                                        next_note.duration.mul(Fraction::new(3, 2));
                                }
                                self.apply_tuplet(&mut next_note.duration);
                                if pos < b.len() && b[pos] == b'-' {
                                    next_note.is_tied = true;
                                    pos += 1;
                                }
                                self.merge_tied_notes(b, &mut pos, &mut next_note, tune);
                                let voice_id = self.current_voice;
                                tune.features.push(Feature {
                                    voice_id,
                                    ts: self.current_time,
                                    line_number: self.current_line,
                                    data: FeatureData::Note(next_note),
                                });
                                self.current_time += next_note.duration.to_f64();
                            }
                        }
                    }
                }
                continue;
            }

            pos += 1;
        }
    }

    fn parse_inline_bracket(
        &mut self,
        seq: &str,
        pos: &mut usize,
        tune: &mut AbcTune,
    ) -> bool {
        let b = seq.as_bytes();
        if *pos + 3 >= b.len() || b[*pos] != b'[' {
            return false;
        }
        let field = b[*pos + 1];
        if !field.is_ascii_alphabetic() || b[*pos + 2] != b':' {
            return false;
        }
        if field.to_ascii_uppercase() == b'V' {
            return self.parse_inline_voice_switch(seq, pos, tune);
        }

        let mut p = *pos + 3;
        let value_start = p;
        while p < b.len() && b[p] != b']' {
            p += 1;
        }
        if p >= b.len() {
            return false;
        }

        let value = seq[value_start..p].trim_matches(|c: char| c == ' ' || c == '\t');
        let owned = value.to_owned();
        self.apply_inline_field(field.to_ascii_uppercase(), &owned, tune);
        *pos = p + 1;
        true
    }

    fn parse_inline_voice_switch(
        &mut self,
        seq: &str,
        pos: &mut usize,
        tune: &mut AbcTune,
    ) -> bool {
        let b = seq.as_bytes();
        if *pos + 2 >= b.len() || b[*pos] != b'[' || b[*pos + 1] != b'V' || b[*pos + 2] != b':' {
            return false;
        }
        *pos += 3;
        let start = *pos;
        while *pos < b.len() && b[*pos] != b']' && !b[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        let voice_identifier = seq[start..*pos].to_owned();

        while *pos < b.len() && b[*pos] != b']' {
            *pos += 1;
        }
        if *pos < b.len() && b[*pos] == b']' {
            *pos += 1;
        } else {
            return false;
        }

        if voice_identifier.is_empty() {
            return false;
        }

        self.save_current_time();
        let voice_id = self.switch_to_voice(&voice_identifier, tune);
        self.current_time = self.restore_voice_time(voice_id);

        tune.features.push(Feature {
            voice_id,
            ts: self.current_time,
            line_number: self.current_line,
            data: FeatureData::Voice(VoiceChange {
                voice_number: voice_id,
                voice_name: voice_identifier,
            }),
        });
        true
    }

    fn parse_note(
        &mut self,
        b: &[u8],
        pos: &mut usize,
        note: &mut Note,
        tune: &AbcTune,
    ) -> bool {
        if *pos >= b.len() {
            return false;
        }
        let mut probe = *pos;
        let _ = parse_accidental(b, &mut probe);
        if probe >= b.len() || !is_note_char(b[probe]) {
            return false;
        }

        let explicit = parse_accidental(b, pos);
        let pitch = parse_pitch(b, pos);
        if pitch == 0 {
            return false;
        }
        note.pitch = pitch;
        note.octave = parse_octave(b, pos);

        let voice = match tune.voices.get(&self.current_voice) {
            Some(v) => v,
            None => return false,
        };

        note.duration = parse_duration(b, pos, voice.unit_len);

        let note_index = pitch_index(pitch).unwrap_or(0);
        let mut accidental = key_accidental_for_pitch(voice.key, pitch.to_ascii_uppercase());

        if let Some(a) = explicit {
            accidental = a;
            self.set_bar_accidental(self.current_voice, note_index, a);
        } else if let Some(a) = self.get_bar_accidental(self.current_voice, note_index) {
            accidental = a;
        }

        note.accidental = accidental;
        note.midi_note = calculate_midi_note(pitch, accidental, note.octave, voice.transpose);
        note.velocity = voice.velocity;
        note.is_tied = false;
        true
    }

    fn merge_tied_notes(
        &mut self,
        b: &[u8],
        pos: &mut usize,
        note: &mut Note,
        tune: &AbcTune,
    ) {
        while note.is_tied {
            let mut scan = skip_tie_join(b, *pos);
            if scan >= b.len() || !is_note_char(b[scan]) {
                break;
            }
            let mut next_note = empty_note();
            if !self.parse_note(b, &mut scan, &mut next_note, tune) {
                break;
            }
            if next_note.midi_note != note.midi_note {
                break;
            }
            note.duration = note.duration.add(next_note.duration);
            *pos = scan;
            if *pos < b.len() && b[*pos] == b'-' {
                note.is_tied = true;
                *pos += 1;
            } else {
                note.is_tied = false;
            }
        }
        note.is_tied = false;
    }

    fn parse_tuplet(&mut self, b: &[u8], pos: &mut usize, tune: &AbcTune) -> bool {
        if *pos >= b.len() || b[*pos] != b'(' {
            return false;
        }
        if *pos + 1 >= b.len() || !b[*pos + 1].is_ascii_digit() {
            return false;
        }
        *pos += 1;
        let p = (b[*pos] - b'0') as i32;
        *pos += 1;

        let mut q: Option<i32> = None;
        let mut r: Option<i32> = None;
        if *pos < b.len() && b[*pos] == b':' {
            *pos += 1;
            if *pos < b.len() && b[*pos].is_ascii_digit() {
                q = Some((b[*pos] - b'0') as i32);
                *pos += 1;
            }
            if *pos < b.len() && b[*pos] == b':' {
                *pos += 1;
                if *pos < b.len() && b[*pos].is_ascii_digit() {
                    r = Some((b[*pos] - b'0') as i32);
                    *pos += 1;
                }
            }
        }

        let timesig = tune
            .voices
            .get(&self.current_voice)
            .map(|v| v.timesig)
            .unwrap_or_default();
        let inferred_q = infer_tuplet_q(p, timesig);
        let final_q = q.unwrap_or(inferred_q);
        let final_r = r.unwrap_or(p);

        if p <= 0 || final_q <= 0 || final_r <= 0 {
            return false;
        }
        self.tuplet_num = final_q;
        self.tuplet_denom = p;
        self.tuplet_notes_remaining = final_r;
        true
    }

    fn apply_tuplet(&mut self, duration: &mut Fraction) {
        if self.tuplet_notes_remaining <= 0 {
            return;
        }
        *duration = duration.mul(Fraction::new(self.tuplet_num, self.tuplet_denom));
        self.tuplet_notes_remaining -= 1;
        if self.tuplet_notes_remaining <= 0 {
            self.tuplet_num = 1;
            self.tuplet_denom = 1;
        }
    }

    fn set_bar_accidental(&mut self, voice_id: i32, idx: usize, a: i8) {
        let entry = self.voice_bar_accidentals.entry(voice_id).or_insert([None; 7]);
        entry[idx] = Some(a);
    }

    fn get_bar_accidental(&self, voice_id: i32, idx: usize) -> Option<i8> {
        self.voice_bar_accidentals.get(&voice_id).and_then(|a| a[idx])
    }

    fn reset_bar_accidentals(&mut self, voice_id: i32) {
        if let Some(state) = self.voice_bar_accidentals.get_mut(&voice_id) {
            *state = [None; 7];
        }
    }

    // -- Voice management --------------------------------------------------

    fn save_current_time(&mut self) {
        if self.current_voice != 0 {
            self.voice_times.insert(self.current_voice, self.current_time);
        }
    }

    fn restore_voice_time(&self, voice_id: i32) -> f64 {
        self.voice_times.get(&voice_id).copied().unwrap_or(0.0)
    }

    fn find_voice_by_identifier(&self, ident: &str, tune: &AbcTune) -> i32 {
        if let Ok(id) = ident.parse::<i32>() {
            if tune.voices.contains_key(&id) {
                return id;
            }
        }
        for (id, voice) in &tune.voices {
            if voice.name == ident {
                return *id;
            }
        }
        self.voice_name_to_id.get(ident).copied().unwrap_or(0)
    }

    fn get_or_create_voice(&mut self, ident: &str, tune: &mut AbcTune) -> i32 {
        let existing = self.find_voice_by_identifier(ident, tune);
        if existing != 0 {
            return existing;
        }
        let voice_id = self.next_voice_id;
        self.next_voice_id += 1;

        let voice = VoiceContext {
            id: voice_id,
            name: ident.to_owned(),
            key: tune.default_key,
            timesig: tune.default_timesig,
            unit_len: tune.default_unit,
            transpose: 0,
            octave_shift: 0,
            instrument: tune.default_instrument,
            channel: tune.default_channel,
            velocity: 80,
            percussion: tune.default_percussion,
        };
        tune.voices.insert(voice_id, voice);
        self.voice_name_to_id.insert(ident.to_owned(), voice_id);
        voice_id
    }

    fn switch_to_voice(&mut self, ident: &str, tune: &mut AbcTune) -> i32 {
        let mut id = self.find_voice_by_identifier(ident, tune);
        if id == 0 {
            id = self.get_or_create_voice(ident, tune);
        }
        self.current_voice = id;
        id
    }

    fn register_external_voice(&mut self, voice_id: i32, ident: &str) {
        // The Zig version walks the map; we just upsert the identifier and
        // bump the auto-increment if necessary.
        self.voice_name_to_id.insert(ident.to_owned(), voice_id);
        if voice_id >= self.next_voice_id {
            self.next_voice_id = voice_id + 1;
        }
    }
}

impl Default for AbcParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn is_string_field(field: u8) -> bool {
    matches!(
        field,
        b'A' | b'B' | b'C' | b'D' | b'F' | b'G' | b'H' | b'N' | b'O' | b'R' | b'S' | b'T' | b'W' | b'Z' | b'w'
    )
}

fn is_header_field(line: &str) -> bool {
    line.len() >= 2 && line.as_bytes()[0].is_ascii_alphabetic() && line.as_bytes()[1] == b':'
}

fn append_string_field(dst: &mut String, value: &str) {
    if value.is_empty() {
        return;
    }
    if dst.is_empty() {
        dst.push_str(value);
    } else {
        dst.push(' ');
        dst.push_str(value);
    }
}

fn parse_meter(value: &str) -> Option<TimeSig> {
    let v = value.trim();
    if matches!(v, "C" | "c") {
        return Some(TimeSig { num: 4, denom: 4 });
    }
    if matches!(v, "C|" | "c|") {
        return Some(TimeSig { num: 2, denom: 2 });
    }
    let (n, d) = v.split_once('/')?;
    let num: u8 = n.trim().parse().ok()?;
    let denom: u8 = d.trim().parse().ok()?;
    Some(TimeSig { num, denom })
}

fn parse_simple_fraction(value: &str) -> Option<Fraction> {
    let (n, d) = value.split_once('/')?;
    let num: i32 = n.trim().parse().ok()?;
    let denom: i32 = d.trim().parse().ok()?;
    Some(Fraction::new(num, denom))
}

fn parse_key_sig(value: &str) -> KeySig {
    let v = value.trim();
    let is_minor = v.contains('m') && v != "M";
    let mut sharps: i8 = 0;
    if v.starts_with("C#") {
        sharps = 7;
    } else if v.starts_with("F#") {
        sharps = 6;
    } else if v.starts_with("Cb") {
        sharps = -7;
    } else if v.starts_with("Gb") {
        sharps = -6;
    } else if v.starts_with("Db") {
        sharps = -5;
    } else if v.starts_with("Ab") {
        sharps = -4;
    } else if v.starts_with("Eb") {
        sharps = -3;
    } else if v.starts_with("Bb") {
        sharps = -2;
    } else if v.starts_with('B') {
        sharps = 5;
    } else if v.starts_with('E') {
        sharps = 4;
    } else if v.starts_with('A') {
        sharps = 3;
    } else if v.starts_with('D') {
        sharps = 2;
    } else if v.starts_with('G') {
        sharps = 1;
    } else if v.starts_with('F') {
        sharps = -1;
    }
    KeySig {
        sharps,
        is_major: !is_minor,
    }
}

fn apply_voice_attributes(attrs: &str, voice: &mut VoiceContext) {
    let bytes = attrs.as_bytes();
    let mut pos = 0;
    while pos < bytes.len() {
        while pos < bytes.len() && bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() {
            break;
        }
        let key_start = pos;
        while pos < bytes.len() && bytes[pos] != b'=' && !bytes[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if pos >= bytes.len() || bytes[pos] != b'=' {
            while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            continue;
        }
        let key = &attrs[key_start..pos];
        pos += 1;

        let val_start;
        let val_end;
        let mut quoted = false;
        if pos < bytes.len() && bytes[pos] == b'"' {
            quoted = true;
            pos += 1;
            val_start = pos;
            while pos < bytes.len() && bytes[pos] != b'"' {
                pos += 1;
            }
            val_end = pos;
            if pos < bytes.len() {
                pos += 1;
            }
        } else {
            val_start = pos;
            while pos < bytes.len() && !bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            val_end = pos;
        }
        let raw = &attrs[val_start..val_end];
        let value = if quoted {
            raw.to_owned()
        } else {
            raw.trim().to_owned()
        };

        if key.eq_ignore_ascii_case("name") {
            voice.name = value;
        } else if key.eq_ignore_ascii_case("instrument")
            || key.eq_ignore_ascii_case("program")
            || key.eq_ignore_ascii_case("prog")
        {
            if let Ok(p) = value.parse::<i32>() {
                if (0..=127).contains(&p) {
                    voice.instrument = p as u8;
                }
            }
        }
    }
}

fn skip_ws(b: &[u8], pos: usize) -> usize {
    let mut p = pos;
    while p < b.len() && (b[p] == b' ' || b[p] == b'\t') {
        p += 1;
    }
    p
}

fn is_note_char(c: u8) -> bool {
    (c.is_ascii_uppercase() && c >= b'A' && c <= b'G')
        || (c.is_ascii_lowercase() && c >= b'a' && c <= b'g')
}

fn is_barline_char(c: u8) -> bool {
    matches!(c, b'|' | b':' | b'[' | b']')
}

fn is_accidental_marker(c: u8) -> bool {
    matches!(c, b'^' | b'_' | b'=')
}

fn parse_accidental(b: &[u8], pos: &mut usize) -> Option<i8> {
    if *pos >= b.len() {
        return None;
    }
    match b[*pos] {
        b'^' => {
            if *pos + 1 < b.len() && b[*pos + 1] == b'^' {
                *pos += 2;
                Some(2)
            } else {
                *pos += 1;
                Some(1)
            }
        }
        b'_' => {
            if *pos + 1 < b.len() && b[*pos + 1] == b'_' {
                *pos += 2;
                Some(-2)
            } else {
                *pos += 1;
                Some(-1)
            }
        }
        b'=' => {
            *pos += 1;
            Some(0)
        }
        _ => None,
    }
}

fn parse_pitch(b: &[u8], pos: &mut usize) -> u8 {
    if *pos >= b.len() || !is_note_char(b[*pos]) {
        return 0;
    }
    let p = b[*pos];
    *pos += 1;
    p
}

fn parse_octave(b: &[u8], pos: &mut usize) -> i8 {
    let mut o: i8 = 0;
    while *pos < b.len() && b[*pos] == b'\'' {
        o += 1;
        *pos += 1;
    }
    while *pos < b.len() && b[*pos] == b',' {
        o -= 1;
        *pos += 1;
    }
    o
}

fn parse_duration(b: &[u8], pos: &mut usize, default_duration: Fraction) -> Fraction {
    let mut duration = default_duration;
    if *pos < b.len() && b[*pos].is_ascii_digit() {
        let mut numerator: i32 = 0;
        while *pos < b.len() && b[*pos].is_ascii_digit() {
            numerator = numerator * 10 + (b[*pos] - b'0') as i32;
            *pos += 1;
        }
        if *pos < b.len() && b[*pos] == b'/' {
            *pos += 1;
            let mut denominator: i32 = 2;
            if *pos < b.len() && b[*pos].is_ascii_digit() {
                denominator = 0;
                while *pos < b.len() && b[*pos].is_ascii_digit() {
                    denominator = denominator * 10 + (b[*pos] - b'0') as i32;
                    *pos += 1;
                }
            }
            duration = Fraction::new(numerator, denominator).mul(default_duration);
        } else {
            duration = Fraction::new(numerator, 1).mul(default_duration);
        }
    } else if *pos < b.len() && b[*pos] == b'/' {
        *pos += 1;
        let mut denominator: i32 = 2;
        if *pos < b.len() && b[*pos].is_ascii_digit() {
            denominator = 0;
            while *pos < b.len() && b[*pos].is_ascii_digit() {
                denominator = denominator * 10 + (b[*pos] - b'0') as i32;
                *pos += 1;
            }
        }
        duration = Fraction::new(default_duration.num, default_duration.denom * denominator);
    }
    while *pos < b.len() && b[*pos] == b'.' {
        duration = duration.mul(Fraction::new(3, 2));
        *pos += 1;
    }
    duration
}

fn parse_guitar_chord(b: &[u8], pos: &mut usize) -> Option<GuitarChord> {
    if *pos >= b.len() || b[*pos] != b'"' {
        return None;
    }
    *pos += 1;
    let start = *pos;
    while *pos < b.len() && b[*pos] != b'"' {
        *pos += 1;
    }
    if *pos >= b.len() {
        return None;
    }
    let symbol_bytes = &b[start..*pos];
    *pos += 1;
    let symbol = std::str::from_utf8(symbol_bytes).ok()?.to_owned();
    let root = parse_chord_root(&symbol);
    let kind = parse_chord_type(&symbol);
    Some(GuitarChord {
        symbol,
        root_note: root,
        chord_type: kind,
        duration: Fraction::new(0, 1),
    })
}

fn parse_chord_root(symbol: &str) -> u8 {
    if symbol.is_empty() {
        return 60;
    }
    let bytes = symbol.as_bytes();
    let root = bytes[0].to_ascii_uppercase();
    let mut midi: u8 = match root {
        b'C' => 60,
        b'D' => 62,
        b'E' => 64,
        b'F' => 65,
        b'G' => 67,
        b'A' => 69,
        b'B' => 71,
        _ => 60,
    };
    if bytes.len() > 1 {
        if bytes[1] == b'#' {
            midi += 1;
        } else if bytes[1] == b'b' {
            midi -= 1;
        }
    }
    midi.saturating_sub(12)
}

fn parse_chord_type(symbol: &str) -> String {
    let bytes = symbol.as_bytes();
    let mut start = 1;
    if bytes.len() > 1 && (bytes[1] == b'#' || bytes[1] == b'b') {
        start = 2;
    }
    if start >= bytes.len() {
        return "major".to_owned();
    }
    let t = &symbol[start..];
    match t {
        "m" | "min" => "minor".to_owned(),
        "7" => "dom7".to_owned(),
        "maj7" | "M7" => "maj7".to_owned(),
        "m7" => "m7".to_owned(),
        "dim" | "o" => "dim".to_owned(),
        "aug" | "+" => "aug".to_owned(),
        "" => "major".to_owned(),
        _ => t.to_owned(),
    }
}

fn skip_grace(b: &[u8], pos: &mut usize) -> bool {
    if *pos >= b.len() || b[*pos] != b'{' {
        return false;
    }
    *pos += 1;
    if *pos < b.len() && b[*pos] == b'/' {
        *pos += 1;
    }
    while *pos < b.len() && b[*pos] != b'}' {
        *pos += 1;
    }
    if *pos < b.len() {
        *pos += 1;
    }
    true
}

fn skip_tie_join(b: &[u8], pos: usize) -> usize {
    let mut p = pos;
    while p < b.len() {
        let c = b[p];
        if c == b' ' || c == b'\t' || c == b'\r' || c == b'|' || c == b':' {
            p += 1;
            continue;
        }
        break;
    }
    p
}

fn pitch_index(pitch: u8) -> Option<usize> {
    match pitch.to_ascii_uppercase() {
        b'A' => Some(0),
        b'B' => Some(1),
        b'C' => Some(2),
        b'D' => Some(3),
        b'E' => Some(4),
        b'F' => Some(5),
        b'G' => Some(6),
        _ => None,
    }
}

fn key_accidental_for_pitch(key: KeySig, pitch_upper: u8) -> i8 {
    const SHARP_ORDER: &[u8] = b"FCGDAEB";
    const FLAT_ORDER: &[u8] = b"BEADGCF";
    if key.sharps > 0 {
        let count = key.sharps as usize;
        for &c in SHARP_ORDER.iter().take(count) {
            if c == pitch_upper {
                return 1;
            }
        }
        return 0;
    }
    if key.sharps < 0 {
        let count = (-key.sharps) as usize;
        for &c in FLAT_ORDER.iter().take(count) {
            if c == pitch_upper {
                return -1;
            }
        }
    }
    0
}

fn calculate_midi_note(pitch: u8, accidental: i8, octave: i8, transpose: i8) -> u8 {
    let normalized = pitch.to_ascii_uppercase();
    if !(b'A'..=b'G').contains(&normalized) {
        return 60;
    }
    // A B C D E F G
    let semitones = [9, 11, 0, 2, 4, 5, 7];
    let mut semitone: i32 = semitones[(normalized - b'A') as usize] as i32;
    semitone += accidental as i32;

    let mut base_octave: i32 = 4;
    if pitch.is_ascii_lowercase() {
        base_octave += 1;
    }
    base_octave += octave as i32;

    let mut midi = base_octave * 12 + semitone + transpose as i32;
    midi = midi.clamp(0, 127);
    midi as u8
}

fn empty_note() -> Note {
    Note {
        pitch: 0,
        accidental: 0,
        octave: 0,
        duration: Fraction::new(0, 1),
        midi_note: 0,
        velocity: 0,
        is_tied: false,
    }
}

fn infer_tuplet_q(p: i32, timesig: TimeSig) -> i32 {
    let compound = timesig.denom == 8 && matches!(timesig.num, 6 | 9 | 12);
    match p {
        2 => 3,
        3 => 2,
        4 => 3,
        6 => 2,
        8 => 3,
        5 | 7 | 9 => if compound { 3 } else { 2 },
        _ => if compound { 3 } else { 2 },
    }
}

// Re-export collection used in tests only.
#[doc(hidden)]
pub fn __test_keysig() -> BTreeMap<&'static str, KeySig> {
    BTreeMap::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_tune() {
        let abc = "X:1\nT:Test\nM:4/4\nL:1/4\nK:C\nCDEF|\n";
        let mut p = AbcParser::new();
        let tune = p.parse(abc).unwrap();
        assert_eq!(tune.title, "Test");
        assert_eq!(tune.default_timesig, TimeSig { num: 4, denom: 4 });
        assert_eq!(tune.default_unit, Fraction::new(1, 4));
        assert_eq!(tune.default_key.sharps, 0);
        assert!(tune.default_key.is_major);
        assert_eq!(tune.voices.len(), 1);
        // CDEF = 4 notes + 1 bar feature.
        let notes: Vec<_> = tune
            .features
            .iter()
            .filter(|f| matches!(f.data, FeatureData::Note(_)))
            .collect();
        assert_eq!(notes.len(), 4);
    }

    #[test]
    fn midi_note_calculation_for_middle_c() {
        // This parser uses the same convention as the original Zig
        // implementation: uppercase `C` = MIDI 48 (C3), lowercase `c` =
        // MIDI 60 (middle C). Preserving the convention keeps tunes that
        // were authored against the original module sounding the same.
        let mut p = AbcParser::new();
        let tune = p.parse("X:1\nK:C\nCc\n").unwrap();
        let notes: Vec<u8> = tune
            .features
            .iter()
            .filter_map(|f| match f.data {
                FeatureData::Note(n) => Some(n.midi_note),
                _ => None,
            })
            .collect();
        assert_eq!(notes, vec![48, 60]);
    }

    #[test]
    fn key_signature_shifts_pitches() {
        // K:G should sharpen every F. F in this octave maps to MIDI 53;
        // adding a sharp from the key signature gives 54.
        let mut p = AbcParser::new();
        let tune = p.parse("X:1\nK:G\nF\n").unwrap();
        if let FeatureData::Note(n) = tune.features.first().unwrap().data {
            assert_eq!(n.midi_note, 54, "F in G major should sharpen to MIDI 54");
        } else {
            panic!("expected note");
        }
    }

    #[test]
    fn explicit_accidental_overrides_key() {
        // K:G with `=F` (natural) should give MIDI 53, not 54.
        let mut p = AbcParser::new();
        let tune = p.parse("X:1\nK:G\n=F\n").unwrap();
        if let FeatureData::Note(n) = tune.features.first().unwrap().data {
            assert_eq!(n.midi_note, 53);
        } else {
            panic!("expected note");
        }
    }

    #[test]
    fn rest_advances_time_but_makes_no_note() {
        let mut p = AbcParser::new();
        let tune = p.parse("X:1\nL:1/4\nK:C\nCzC\n").unwrap();
        let notes: Vec<f64> = tune
            .features
            .iter()
            .filter_map(|f| match f.data {
                FeatureData::Note(_) => Some(f.ts),
                _ => None,
            })
            .collect();
        assert_eq!(notes.len(), 2);
        // First C at 0, rest occupies 1/4, second C at 2/4 (== 0.5).
        assert!((notes[0] - 0.0).abs() < 1e-9);
        assert!((notes[1] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn broken_rhythm_first_longer() {
        // C>D in L:1/8 means C * 3/2, D * 1/2.
        let mut p = AbcParser::new();
        let tune = p.parse("X:1\nL:1/8\nK:C\nC>D\n").unwrap();
        let durations: Vec<f64> = tune
            .features
            .iter()
            .filter_map(|f| match f.data {
                FeatureData::Note(n) => Some(n.duration.to_f64()),
                _ => None,
            })
            .collect();
        assert_eq!(durations.len(), 2);
        assert!((durations[0] - 0.1875).abs() < 1e-9); // 3/16
        assert!((durations[1] - 0.0625).abs() < 1e-9); // 1/16
    }

    #[test]
    fn percussion_directive_routes_to_channel_9() {
        let mut p = AbcParser::new();
        let tune = p
            .parse("X:1\nT:Drum\nL:1/8\nK:C\n%%MIDI drum on\nz4\n")
            .unwrap();
        // The drum directive comes after K: but before any notes, so we
        // currently have no voices when it is processed → default_channel
        // should be set to 9.
        assert_eq!(tune.default_channel, 9);
        assert!(tune.default_percussion);
    }
}
