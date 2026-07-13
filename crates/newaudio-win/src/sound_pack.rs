//! Typed enum-keyed sound library.
//!
//! Lets game code define an `enum GameSound { Coin, Jump, Hurt, … }` and
//! refer to game sounds by that enum end-to-end. The pack registers each
//! buffer with [`PcmRuntime`] on construction and remembers the resulting
//! [`SoundId`] under the typed key.
//!
//! ```no_run
//! # #[cfg(windows)] {
//! use std::sync::Arc;
//! use newaudio_core::{Config, Engine};
//! use newaudio_win::{PcmRuntime, SoundPack};
//!
//! #[derive(Copy, Clone, Eq, PartialEq, Hash)]
//! enum GameSound { Coin, Jump, Hurt }
//!
//! let rt = Arc::new(PcmRuntime::start().unwrap());
//! let mut eng = Engine::new(Config::default());
//! let pack = SoundPack::builder(rt.clone())
//!     .insert(GameSound::Coin, eng.coin(1.0, 0.4))
//!     .insert(GameSound::Jump, eng.jump(1.0, 0.3))
//!     .insert(GameSound::Hurt, eng.hurt(1.0, 0.4))
//!     .build();
//!
//! pack.play(GameSound::Coin);
//! # }
//! ```
//!
//! For variations on a single key, use [`SoundPackBuilder::insert_many`]
//! and call [`SoundPack::play_random`] to pick one at random per trigger.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;

use newaudio_core::Buffer;

use crate::pcm_runtime::{PcmRuntime, PlayOptions, SoundId, VoiceHandle};

/// Anything `Copy + Eq + Hash` can index a `SoundPack`.
pub trait SoundKey: Copy + Eq + Hash {}
impl<T: Copy + Eq + Hash> SoundKey for T {}

/// Build-time accumulator for a [`SoundPack`].
pub struct SoundPackBuilder<K: SoundKey> {
    runtime: Arc<PcmRuntime>,
    pending: HashMap<K, Vec<SoundId>>,
}

impl<K: SoundKey> SoundPackBuilder<K> {
    /// Register `buf` under `key`. If `key` already has sounds, this
    /// becomes another variation.
    pub fn insert(mut self, key: K, buf: Buffer) -> Self {
        let id = self.runtime.register_sound(buf);
        self.pending.entry(key).or_default().push(id);
        self
    }

    /// Register a batch of variations under a single key.
    pub fn insert_many<I: IntoIterator<Item = Buffer>>(mut self, key: K, bufs: I) -> Self {
        let entry = self.pending.entry(key).or_default();
        for buf in bufs {
            let id = self.runtime.register_sound(buf);
            entry.push(id);
        }
        self
    }

    pub fn build(self) -> SoundPack<K> {
        SoundPack {
            runtime: self.runtime,
            sounds: self.pending,
        }
    }
}

/// Typed key → SoundIds registry. Plays sounds without the caller needing
/// to keep raw [`SoundId`]s in scope.
pub struct SoundPack<K: SoundKey> {
    runtime: Arc<PcmRuntime>,
    sounds: HashMap<K, Vec<SoundId>>,
}

impl<K: SoundKey> SoundPack<K> {
    pub fn builder(runtime: Arc<PcmRuntime>) -> SoundPackBuilder<K> {
        SoundPackBuilder {
            runtime,
            pending: HashMap::new(),
        }
    }

    /// Play the (first registered) sound for `key` at full volume.
    pub fn play(&self, key: K) -> Option<VoiceHandle> {
        self.play_with(key, PlayOptions::default())
    }

    /// Play the first registered sound for `key` with custom options.
    pub fn play_with(&self, key: K, opts: PlayOptions) -> Option<VoiceHandle> {
        let id = *self.sounds.get(&key)?.first()?;
        self.runtime.play_with(id, opts)
    }

    /// Pick a random variation registered under `key` and play it.
    pub fn play_random(&self, key: K, volume: f32, pan: f32) -> Option<VoiceHandle> {
        let ids = self.sounds.get(&key)?;
        self.runtime.play_random(ids, volume, pan)
    }

    /// Number of distinct keys registered.
    pub fn key_count(&self) -> usize {
        self.sounds.len()
    }

    /// Number of variations registered under `key`.
    pub fn variation_count(&self, key: K) -> usize {
        self.sounds.get(&key).map(Vec::len).unwrap_or(0)
    }

    /// Direct access to the underlying runtime — useful for stopping a
    /// specific voice returned by `play`.
    pub fn runtime(&self) -> &Arc<PcmRuntime> {
        &self.runtime
    }
}
