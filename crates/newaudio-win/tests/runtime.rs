//! Integration tests that boot real Windows audio runtimes.
//!
//! These open `waveOut` and `midiOut` so they need a working audio
//! subsystem. If the runtime fails to start, the test is reported as
//! passing-with-a-skip-note — useful on headless CI without audio.

use std::sync::Arc;
use std::thread::sleep;
use std::time::Duration;

use newaudio_core::{Config, Engine};
use newaudio_win::{AudioThread, Mixer, PcmRuntime, PlayOptions, SoundPack};

fn try_start_pcm() -> Option<Arc<PcmRuntime>> {
    match PcmRuntime::start() {
        Ok(r) => Some(Arc::new(r)),
        Err(e) => {
            eprintln!("skipping: PcmRuntime::start() failed: {e}");
            None
        }
    }
}

fn try_start_mixer() -> Option<Mixer> {
    match Mixer::start() {
        Ok(m) => Some(m),
        Err(e) => {
            eprintln!("skipping: Mixer::start() failed: {e}");
            None
        }
    }
}

#[test]
fn pcm_runtime_play_returns_voice_handle() {
    let Some(rt) = try_start_pcm() else { return };
    let mut eng = Engine::new(Config::default());
    let id = rt.register_sound(eng.beep(440.0, 0.02));
    let h = rt.play(id, 1.0, 0.0).unwrap();
    assert!(rt.is_voice_active(h));
    // Let it play out (20 ms sound, mixer block ≈ 46 ms).
    sleep(Duration::from_millis(150));
    assert!(!rt.is_voice_active(h));
}

#[test]
fn play_with_volume_jitter_does_not_crash() {
    let Some(rt) = try_start_pcm() else { return };
    let mut eng = Engine::new(Config::default());
    let id = rt.register_sound(eng.beep(440.0, 0.02));
    for _ in 0..10 {
        let _ = rt.play_with(
            id,
            PlayOptions {
                volume: 0.5,
                volume_jitter: 0.2,
                ..Default::default()
            },
        );
    }
    // No assertion on audio — just that the runtime handled 10 plays.
    assert!(rt.sound_count() >= 1);
}

#[test]
fn looped_voice_stays_active_until_stopped() {
    let Some(rt) = try_start_pcm() else { return };
    let mut eng = Engine::new(Config::default());
    let id = rt.register_sound(eng.beep(220.0, 0.02));
    let h = rt.play_looped(id, 0.2, 0.0).unwrap();

    sleep(Duration::from_millis(80));
    assert!(rt.is_voice_active(h), "looped voice should still be active");

    rt.stop_voice(h, None);
    sleep(Duration::from_millis(20));
    assert!(!rt.is_voice_active(h));
}

#[test]
fn fade_out_terminates_voice_within_window() {
    let Some(rt) = try_start_pcm() else { return };
    let mut eng = Engine::new(Config::default());
    let id = rt.register_sound(eng.beep(330.0, 0.05));
    let h = rt.play_looped(id, 0.3, 0.0).unwrap();

    // Start a 50 ms fade-out; voice should be gone within ~150 ms.
    rt.stop_voice(h, Some(0.05));
    let mut spent_ms = 0;
    while rt.is_voice_active(h) && spent_ms < 250 {
        sleep(Duration::from_millis(10));
        spent_ms += 10;
    }
    assert!(
        !rt.is_voice_active(h),
        "fade-out did not finish within 250 ms"
    );
}

#[test]
fn play_random_selects_one_of_the_ids() {
    let Some(rt) = try_start_pcm() else { return };
    let mut eng = Engine::new(Config::default());
    let ids = vec![
        rt.register_sound(eng.beep(440.0, 0.01)),
        rt.register_sound(eng.beep(550.0, 0.01)),
        rt.register_sound(eng.beep(660.0, 0.01)),
    ];
    for _ in 0..50 {
        let h = rt.play_random(&ids, 0.5, 0.0);
        assert!(h.is_some());
    }
}

#[test]
fn sfx_bus_volume_is_independent_of_master() {
    let Some(rt) = try_start_pcm() else { return };
    rt.set_master_volume(0.5);
    rt.set_sfx_bus_volume(0.25);
    assert!((rt.master_volume() - 0.5).abs() < 1e-6);
    assert!((rt.sfx_bus_volume() - 0.25).abs() < 1e-6);
}

#[test]
fn audio_thread_play_does_not_block() {
    let Some(rt) = try_start_pcm() else { return };
    let mut eng = Engine::new(Config::default());
    let id = rt.register_sound(eng.beep(440.0, 0.005));

    let audio = AudioThread::spawn(Arc::clone(&rt));
    // Submit 1000 commands rapidly; the call site never takes the runtime
    // mutex so this must complete in tens of ms even on a busy box.
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        audio.play(id, 0.1, 0.0);
    }
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(200),
        "1000 play submissions took {elapsed:?}"
    );

    // Give the worker time to drain and the mixer time to retire voices.
    sleep(Duration::from_millis(50));
    drop(audio);
}

#[test]
fn sound_pack_indexes_by_typed_key() {
    let Some(rt) = try_start_pcm() else { return };

    #[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
    enum GameSound {
        Coin,
        Jump,
        Hurt,
    }

    let mut eng = Engine::new(Config::default());
    let pack = SoundPack::builder(rt.clone())
        .insert(GameSound::Coin, eng.coin(1.0, 0.02))
        .insert(GameSound::Jump, eng.jump(1.0, 0.02))
        .insert(GameSound::Hurt, eng.hurt(1.0, 0.02))
        .build();

    assert_eq!(pack.key_count(), 3);
    assert!(pack.play(GameSound::Coin).is_some());
    assert!(pack.play(GameSound::Jump).is_some());
}

#[test]
fn sound_pack_supports_variations() {
    let Some(rt) = try_start_pcm() else { return };

    #[derive(Copy, Clone, Eq, PartialEq, Hash)]
    enum Sfx {
        Hit,
    }

    let mut eng = Engine::new(Config::default());
    let pack = SoundPack::builder(rt.clone())
        .insert_many(
            Sfx::Hit,
            vec![
                eng.click(1.0, 0.02),
                eng.click(0.8, 0.02),
                eng.click(1.2, 0.02),
            ],
        )
        .build();

    assert_eq!(pack.variation_count(Sfx::Hit), 3);
    for _ in 0..10 {
        assert!(pack.play_random(Sfx::Hit, 0.5, 0.0).is_some());
    }
}

#[test]
fn mixer_keeps_master_in_sync_across_runtimes() {
    let Some(mixer) = try_start_mixer() else {
        return;
    };
    mixer.set_master_volume(0.4);
    assert!((mixer.pcm().master_volume() - 0.4).abs() < 1e-6);
    assert!((mixer.midi().master_volume() - 0.4).abs() < 1e-6);

    mixer.set_sfx_bus_volume(0.6);
    mixer.set_music_bus_volume(0.3);
    assert!((mixer.sfx_bus_volume() - 0.6).abs() < 1e-6);
    assert!((mixer.music_bus_volume() - 0.3).abs() < 1e-6);
}
