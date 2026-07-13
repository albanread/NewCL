//! Cross-check the Rust ABC→MIDI pipeline against the original Zig
//! implementation. For each `tests/golden_abc/*.abc` file:
//!
//! 1. Run the Zig harness binary (`tests/zig_golden/zig-out/bin/abc_to_midi.exe`)
//!    to produce reference SMF bytes.
//! 2. Run the Rust pipeline on the same input.
//! 3. Assert byte-equality.
//!
//! If the Zig harness isn't built (or we're on a non-Windows host where it
//! wasn't cross-compiled), the test prints a hint and exits successfully —
//! the suite is still useful without the cross-check, just less strict.

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points at the crate dir; the workspace root is two
    // levels up (`crates/newaudio-abc` → workspace root).
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_owned()
}

fn zig_harness_path() -> PathBuf {
    let ws = workspace_root();
    if cfg!(target_os = "windows") {
        ws.join("tests")
            .join("zig_golden")
            .join("zig-out")
            .join("bin")
            .join("abc_to_midi.exe")
    } else {
        ws.join("tests")
            .join("zig_golden")
            .join("zig-out")
            .join("bin")
            .join("abc_to_midi")
    }
}

fn run_zig_harness(abc_path: &Path) -> Option<Vec<u8>> {
    let exe = zig_harness_path();
    if !exe.exists() {
        eprintln!(
            "skipping cross-check: Zig harness not built at {}\n\
             To enable: cd tests/zig_golden && zig build",
            exe.display()
        );
        return None;
    }
    let out_path = std::env::temp_dir().join(format!(
        "newaudio_zig_{}.mid",
        abc_path.file_stem().unwrap().to_string_lossy()
    ));

    let status = Command::new(&exe)
        .arg(abc_path)
        .arg(&out_path)
        .status()
        .expect("spawn Zig harness");
    assert!(status.success(), "Zig harness failed for {:?}", abc_path);
    let bytes = std::fs::read(&out_path).expect("read harness output");
    Some(bytes)
}

#[test]
fn rust_matches_zig_for_all_golden_tunes() {
    let golden_dir = workspace_root().join("tests").join("golden_abc");
    let entries: Vec<_> = std::fs::read_dir(&golden_dir)
        .expect("golden dir exists")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("abc"))
        .collect();
    assert!(!entries.is_empty(), "no .abc tunes in {:?}", golden_dir);

    let mut compared = 0_usize;
    for entry in entries {
        let abc_path = entry.path();
        let abc_text = std::fs::read_to_string(&abc_path).expect("read abc");
        let (_, rust_bytes) = newaudio_abc::abc_to_smf(&abc_text).expect("rust parse");

        let Some(zig_bytes) = run_zig_harness(&abc_path) else {
            continue;
        };

        if rust_bytes != zig_bytes {
            // Helpful diff: show the index of the first divergence and a
            // small window around it.
            let mismatch_at = rust_bytes
                .iter()
                .zip(zig_bytes.iter())
                .position(|(a, b)| a != b)
                .unwrap_or(rust_bytes.len().min(zig_bytes.len()));
            let start = mismatch_at.saturating_sub(8);
            let end = (mismatch_at + 16).min(rust_bytes.len().min(zig_bytes.len()));
            panic!(
                "{}: rust ({} bytes) != zig ({} bytes), first diff at offset 0x{:X}\n\
                 rust[{:X}..{:X}] = {:02X?}\n\
                 zig [{:X}..{:X}] = {:02X?}",
                abc_path.display(),
                rust_bytes.len(),
                zig_bytes.len(),
                mismatch_at,
                start, end, &rust_bytes[start..end.min(rust_bytes.len())],
                start, end, &zig_bytes[start..end.min(zig_bytes.len())],
            );
        }
        compared += 1;
    }

    eprintln!("cross-checked {compared} ABC tunes against the Zig reference");
}
