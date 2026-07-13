//! Repeat expansion. `|: ... :|` markers become the inner content emitted
//! twice. Matches the Zig `repeat.zig` algorithm so the downstream parser
//! sees the same input.

fn expand_line_repeats(line: &str) -> String {
    let bytes = line.as_bytes();
    let mut result = String::new();
    let mut pos = 0;

    while pos < bytes.len() {
        match find_subslice(&bytes[pos..], b"|:") {
            Some(rel_start) => {
                let start = pos + rel_start;
                match find_subslice(&bytes[start + 2..], b":|") {
                    Some(rel_end) => {
                        let end = start + 2 + rel_end;
                        result.push_str(&line[pos..start]);
                        let inner = std::str::from_utf8(&bytes[start + 2..end])
                            .unwrap_or("")
                            .trim_matches(|c: char| c == ' ' || c == '\t');
                        result.push_str(inner);
                        result.push(' ');
                        result.push_str(inner);
                        pos = end + 2;
                        continue;
                    }
                    None => break,
                }
            }
            None => break,
        }
    }
    result.push_str(&line[pos..]);
    result
}

fn expand_voice_section_repeats(section: &str) -> String {
    let mut output = String::new();
    let mut buffer: Vec<String> = Vec::new();
    let mut in_repeat = false;

    for line in section.split('\n') {
        if let Some(start) = line.find("|:") {
            in_repeat = true;
            if line[start + 2..].contains(":|") {
                let expanded = expand_line_repeats(line);
                output.push_str(&expanded);
                output.push('\n');
                in_repeat = false;
                continue;
            }
            let mut cleaned = String::new();
            cleaned.push_str(&line[..start]);
            cleaned.push('|');
            cleaned.push_str(&line[start + 2..]);
            buffer.push(cleaned);
            continue;
        }

        if in_repeat {
            if let Some(end) = line.find(":|") {
                let mut cleaned = String::new();
                cleaned.push_str(&line[..end]);
                cleaned.push('|');
                cleaned.push_str(&line[end + 2..]);
                buffer.push(cleaned);

                for _ in 0..2 {
                    for entry in &buffer {
                        output.push_str(entry);
                        output.push('\n');
                    }
                }
                buffer.clear();
                in_repeat = false;
                continue;
            }
            buffer.push(line.to_owned());
            continue;
        }

        output.push_str(line);
        output.push('\n');
    }

    if !buffer.is_empty() {
        for entry in &buffer {
            output.push_str(entry);
            output.push('\n');
        }
    }

    output
}

pub fn expand_abc_repeats(abc: &str) -> String {
    let mut output = String::new();
    let mut current_section = String::new();
    let mut in_header = true;
    let mut in_voice_section = false;

    let flush_section =
        |dst: &mut String, current: &mut String, in_voice_section: &mut bool| {
            if *in_voice_section && !current.is_empty() {
                let expanded = expand_voice_section_repeats(current);
                dst.push_str(&expanded);
                current.clear();
            }
        };

    for line in abc.split('\n') {
        if in_header {
            output.push_str(line);
            output.push('\n');
            if line.len() >= 2 && line.starts_with('K') && line.as_bytes()[1] == b':' {
                in_header = false;
            }
            continue;
        }

        if line.len() >= 2 && line.starts_with('V') && line.as_bytes()[1] == b':' {
            flush_section(&mut output, &mut current_section, &mut in_voice_section);
            output.push_str(line);
            output.push('\n');
            continue;
        }

        if line.starts_with("%%") {
            flush_section(&mut output, &mut current_section, &mut in_voice_section);
            in_voice_section = false;
            output.push_str(line);
            output.push('\n');
            continue;
        }

        if line.starts_with("[V:") {
            flush_section(&mut output, &mut current_section, &mut in_voice_section);
            output.push_str(line);
            output.push('\n');
            in_voice_section = true;
            continue;
        }

        let trimmed = line.trim_matches(|c: char| c == ' ' || c == '\t' || c == '\r');
        if trimmed.is_empty() {
            flush_section(&mut output, &mut current_section, &mut in_voice_section);
            in_voice_section = false;
            output.push_str(line);
            output.push('\n');
            continue;
        }

        if in_voice_section {
            current_section.push_str(line);
            current_section.push('\n');
        } else {
            in_voice_section = true;
            current_section.push_str(line);
            current_section.push('\n');
        }
    }

    flush_section(&mut output, &mut current_section, &mut in_voice_section);

    output
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_line_repeat_duplicates_content() {
        let input = "X:1\nK:C\n|:CDEF:|\n";
        let out = expand_abc_repeats(input);
        // The single-line expansion should produce two copies of the
        // inner phrase joined by a space.
        assert!(out.contains("CDEF CDEF"), "expanded = {out:?}");
    }

    #[test]
    fn header_is_preserved_verbatim() {
        let input = "X:1\nT:Hello\nK:C\nCDEF\n";
        let out = expand_abc_repeats(input);
        assert!(out.starts_with("X:1\nT:Hello\nK:C\n"));
    }

    #[test]
    fn no_repeat_markers_passthrough() {
        let input = "X:1\nK:C\nCDEF|GABc\n";
        let out = expand_abc_repeats(input);
        assert!(out.contains("CDEF|GABc"));
    }

    #[test]
    fn multi_line_repeat_duplicates_block() {
        let input = "X:1\nK:C\n|:\nCDEF|\nGABc:|\n";
        let out = expand_abc_repeats(input);
        // Multi-line repeat: the two lines should appear twice.
        let cdef_count = out.matches("CDEF").count();
        let gabc_count = out.matches("GABc").count();
        assert_eq!(cdef_count, 2, "got: {out:?}");
        assert_eq!(gabc_count, 2, "got: {out:?}");
    }
}
