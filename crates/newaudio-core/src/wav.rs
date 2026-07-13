//! Canonical PCM16 RIFF/WAVE writer + reader.

use std::io::{self, Read, Write};
use std::path::Path;

use crate::engine::Buffer;

/// Errors emitted by [`read_wav`].
#[derive(Debug)]
pub enum WavReadError {
    Io(io::Error),
    InvalidRiff,
    InvalidWave,
    MissingFmt,
    MissingData,
    UnsupportedFormat(u16),
    UnsupportedBitDepth(u16),
    UnsupportedChannels(u16),
}

impl std::fmt::Display for WavReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::InvalidRiff => write!(f, "not a RIFF file"),
            Self::InvalidWave => write!(f, "RIFF type is not WAVE"),
            Self::MissingFmt => write!(f, "missing fmt chunk"),
            Self::MissingData => write!(f, "missing data chunk"),
            Self::UnsupportedFormat(tag) => write!(f, "unsupported audio format tag {tag}"),
            Self::UnsupportedBitDepth(d) => write!(f, "unsupported bit depth {d}"),
            Self::UnsupportedChannels(c) => write!(f, "unsupported channel count {c}"),
        }
    }
}

impl std::error::Error for WavReadError {}
impl From<io::Error> for WavReadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WavParams {
    pub volume: f32,
}

impl Default for WavParams {
    fn default() -> Self {
        Self { volume: 1.0 }
    }
}

/// Write `buf` as a canonical 16-bit PCM WAV.
///
/// Returns the number of bytes written (header + data). The output layout
/// is byte-for-byte identical to the original `sound_runtime_win.cpp`
/// writer: little-endian RIFF, 16-byte fmt chunk, PCM samples scaled by
/// `volume` and clamped to `[-1, 1]` before rounding to `int16`.
pub fn write_wav<W: Write>(
    mut sink: W,
    buf: &Buffer,
    params: WavParams,
) -> io::Result<usize> {
    let channels = buf.channels as u16;
    let bits_per_sample: u16 = 16;
    let block_align: u16 = channels * (bits_per_sample / 8);
    let byte_rate: u32 = buf.sample_rate * block_align as u32;
    let data_size: u32 = (buf.samples.len() * size_of::<i16>()) as u32;
    let riff_size: u32 = 36 + data_size;

    sink.write_all(b"RIFF")?;
    sink.write_all(&riff_size.to_le_bytes())?;
    sink.write_all(b"WAVE")?;

    sink.write_all(b"fmt ")?;
    sink.write_all(&16u32.to_le_bytes())?;
    sink.write_all(&1u16.to_le_bytes())?; // PCM
    sink.write_all(&channels.to_le_bytes())?;
    sink.write_all(&buf.sample_rate.to_le_bytes())?;
    sink.write_all(&byte_rate.to_le_bytes())?;
    sink.write_all(&block_align.to_le_bytes())?;
    sink.write_all(&bits_per_sample.to_le_bytes())?;

    sink.write_all(b"data")?;
    sink.write_all(&data_size.to_le_bytes())?;

    let gain = params.volume.max(0.0);
    for &s in &buf.samples {
        let scaled = (s * gain).clamp(-1.0, 1.0);
        let pcm = (scaled * 32_767.0).round() as i16;
        sink.write_all(&pcm.to_le_bytes())?;
    }

    Ok(44 + data_size as usize)
}

/// Convenience wrapper that opens `path` for writing.
pub fn write_wav_file(
    path: impl AsRef<Path>,
    buf: &Buffer,
    params: WavParams,
) -> io::Result<usize> {
    let file = std::fs::File::create(path)?;
    let bw = std::io::BufWriter::new(file);
    write_wav(bw, buf, params)
}

/// Read a canonical PCM WAV into a [`Buffer`].
///
/// Supports 8-, 16-, 24-, and 32-bit linear PCM (format tag 1) with 1 or 2
/// channels. Other RIFF chunks between `fmt` and `data` (e.g. `LIST`,
/// `bext`) are skipped. Floating-point WAVs (tag 3) and other compressed
/// formats are rejected with [`WavReadError::UnsupportedFormat`].
pub fn read_wav<R: Read>(mut src: R) -> Result<Buffer, WavReadError> {
    let mut riff = [0u8; 4];
    src.read_exact(&mut riff)?;
    if &riff != b"RIFF" {
        return Err(WavReadError::InvalidRiff);
    }
    let mut size_buf = [0u8; 4];
    src.read_exact(&mut size_buf)?;
    let _riff_size = u32::from_le_bytes(size_buf);

    let mut wave = [0u8; 4];
    src.read_exact(&mut wave)?;
    if &wave != b"WAVE" {
        return Err(WavReadError::InvalidWave);
    }

    let mut format_tag = 0u16;
    let mut channels = 0u16;
    let mut sample_rate = 0u32;
    let mut bits_per_sample = 0u16;
    let mut have_fmt = false;
    let mut data: Vec<u8> = Vec::new();

    loop {
        let mut chunk_id = [0u8; 4];
        match src.read_exact(&mut chunk_id) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let mut chunk_size_buf = [0u8; 4];
        src.read_exact(&mut chunk_size_buf)?;
        let chunk_size = u32::from_le_bytes(chunk_size_buf) as usize;

        match &chunk_id {
            b"fmt " => {
                let mut fmt = vec![0u8; chunk_size];
                src.read_exact(&mut fmt)?;
                if fmt.len() < 16 {
                    return Err(WavReadError::MissingFmt);
                }
                format_tag = u16::from_le_bytes([fmt[0], fmt[1]]);
                channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
                bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
                have_fmt = true;
                if chunk_size % 2 == 1 {
                    let mut pad = [0u8; 1];
                    src.read_exact(&mut pad)?;
                }
            }
            b"data" => {
                data.resize(chunk_size, 0);
                src.read_exact(&mut data)?;
                if chunk_size % 2 == 1 {
                    let mut pad = [0u8; 1];
                    src.read_exact(&mut pad)?;
                }
                break;
            }
            _ => {
                // Skip unknown chunks (LIST, bext, etc.). Pad to even size.
                let total = chunk_size + (chunk_size & 1);
                let mut sink = vec![0u8; total];
                src.read_exact(&mut sink)?;
            }
        }
    }

    if !have_fmt {
        return Err(WavReadError::MissingFmt);
    }
    if data.is_empty() {
        return Err(WavReadError::MissingData);
    }
    if format_tag != 1 {
        return Err(WavReadError::UnsupportedFormat(format_tag));
    }
    if !matches!(channels, 1 | 2) {
        return Err(WavReadError::UnsupportedChannels(channels));
    }
    if !matches!(bits_per_sample, 8 | 16 | 24 | 32) {
        return Err(WavReadError::UnsupportedBitDepth(bits_per_sample));
    }

    let bytes_per_sample = (bits_per_sample / 8) as usize;
    let sample_count = data.len() / bytes_per_sample;
    let mut samples = Vec::with_capacity(sample_count);
    let chunks = data.chunks_exact(bytes_per_sample);
    for chunk in chunks {
        let v = match bits_per_sample {
            8 => (chunk[0] as f32 - 128.0) / 128.0,
            16 => {
                let i = i16::from_le_bytes([chunk[0], chunk[1]]);
                i as f32 / 32_768.0
            }
            24 => {
                let lo = chunk[0] as i32;
                let mid = chunk[1] as i32;
                let hi = chunk[2] as i8 as i32; // sign-extend
                let value = lo | (mid << 8) | (hi << 16);
                value as f32 / 8_388_608.0
            }
            32 => {
                let i = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                i as f32 / 2_147_483_648.0
            }
            _ => unreachable!(),
        };
        samples.push(v);
    }

    let frame_count = samples.len() / channels as usize;
    let duration = frame_count as f32 / sample_rate as f32;
    Ok(Buffer {
        sample_rate,
        channels: channels as u32,
        duration,
        samples,
    })
}

/// Open a `.wav` from disk and decode it.
pub fn read_wav_file(path: impl AsRef<Path>) -> Result<Buffer, WavReadError> {
    let file = std::fs::File::open(path)?;
    read_wav(std::io::BufReader::new(file))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{Buffer, Config};

    fn silent_buffer(channels: u32, frames: usize) -> Buffer {
        Buffer {
            sample_rate: 44_100,
            channels,
            duration: frames as f32 / 44_100.0,
            samples: vec![0.0; frames * channels as usize],
        }
    }

    #[test]
    fn writes_riff_header_correctly() {
        let buf = silent_buffer(2, 100);
        let mut out = Vec::new();
        let bytes = write_wav(&mut out, &buf, WavParams::default()).unwrap();
        assert_eq!(bytes, out.len());
        assert_eq!(&out[0..4], b"RIFF");
        assert_eq!(&out[8..12], b"WAVE");
        assert_eq!(&out[12..16], b"fmt ");
        // fmt chunk size == 16
        assert_eq!(u32::from_le_bytes(out[16..20].try_into().unwrap()), 16);
        // PCM format == 1
        assert_eq!(u16::from_le_bytes(out[20..22].try_into().unwrap()), 1);
        // channels == 2
        assert_eq!(u16::from_le_bytes(out[22..24].try_into().unwrap()), 2);
        // sample rate == 44100
        assert_eq!(u32::from_le_bytes(out[24..28].try_into().unwrap()), 44_100);
        // byte rate == 44100 * 4 (stereo, 16-bit)
        assert_eq!(
            u32::from_le_bytes(out[28..32].try_into().unwrap()),
            44_100 * 4
        );
        // block align == 4
        assert_eq!(u16::from_le_bytes(out[32..34].try_into().unwrap()), 4);
        // bits per sample == 16
        assert_eq!(u16::from_le_bytes(out[34..36].try_into().unwrap()), 16);
        assert_eq!(&out[36..40], b"data");
        // data size == 100 frames * 2 channels * 2 bytes
        assert_eq!(
            u32::from_le_bytes(out[40..44].try_into().unwrap()),
            100 * 2 * 2
        );
    }

    #[test]
    fn riff_size_matches_total_length() {
        let buf = silent_buffer(1, 50);
        let mut out = Vec::new();
        write_wav(&mut out, &buf, WavParams::default()).unwrap();
        let riff_size = u32::from_le_bytes(out[4..8].try_into().unwrap()) as usize;
        assert_eq!(riff_size + 8, out.len());
    }

    #[test]
    fn samples_clamp_at_extremes() {
        let buf = Buffer {
            sample_rate: 44_100,
            channels: 1,
            duration: 1.0 / 44_100.0,
            samples: vec![10.0, -10.0, 0.5, -0.5],
        };
        let mut out = Vec::new();
        write_wav(&mut out, &buf, WavParams::default()).unwrap();
        let s0 = i16::from_le_bytes(out[44..46].try_into().unwrap());
        let s1 = i16::from_le_bytes(out[46..48].try_into().unwrap());
        let s2 = i16::from_le_bytes(out[48..50].try_into().unwrap());
        let s3 = i16::from_le_bytes(out[50..52].try_into().unwrap());
        assert_eq!(s0, 32_767);
        assert_eq!(s1, -32_767);
        assert_eq!(s2, (0.5f32 * 32_767.0).round() as i16);
        assert_eq!(s3, (-0.5f32 * 32_767.0).round() as i16);
    }

    #[test]
    fn mono_44100_one_frame() {
        // Minimal valid WAV: 44 header bytes + 2 data bytes for one mono i16.
        let buf = Buffer {
            sample_rate: 44_100,
            channels: 1,
            duration: 1.0 / 44_100.0,
            samples: vec![0.0],
        };
        let mut out = Vec::new();
        write_wav(&mut out, &buf, WavParams::default()).unwrap();
        assert_eq!(out.len(), 44 + 2);
    }

    #[test]
    fn write_then_read_roundtrip_pcm16() {
        let buf = Buffer {
            sample_rate: 44_100,
            channels: 2,
            duration: 16.0 / 44_100.0,
            samples: vec![
                0.0, 0.0, 0.5, -0.5, 1.0, -1.0, -1.0, 1.0, 0.25, -0.25, 0.75, -0.75, 0.1, -0.1,
                0.9, -0.9,
            ],
        };
        let mut bytes = Vec::new();
        write_wav(&mut bytes, &buf, WavParams::default()).unwrap();
        let decoded = read_wav(&bytes[..]).unwrap();
        assert_eq!(decoded.sample_rate, buf.sample_rate);
        assert_eq!(decoded.channels, buf.channels);
        assert_eq!(decoded.samples.len(), buf.samples.len());
        // PCM16 has ~3e-5 quantisation error per sample.
        for (a, b) in decoded.samples.iter().zip(buf.samples.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} != {b}");
        }
    }

    #[test]
    fn read_rejects_non_riff() {
        let bytes = b"BANG\0\0\0\0WAVEfmt ";
        let err = read_wav(&bytes[..]).unwrap_err();
        assert!(matches!(err, WavReadError::InvalidRiff));
    }

    #[test]
    fn deterministic_output_for_same_engine_state() {
        let mut e = crate::engine::Engine::with_seed(Config::default(), 42);
        let b1 = e.beep(440.0, 0.05);
        let mut a = Vec::new();
        write_wav(&mut a, &b1, WavParams::default()).unwrap();

        let mut e2 = crate::engine::Engine::with_seed(Config::default(), 42);
        let b2 = e2.beep(440.0, 0.05);
        let mut a2 = Vec::new();
        write_wav(&mut a2, &b2, WavParams::default()).unwrap();

        assert_eq!(a, a2);
    }
}
