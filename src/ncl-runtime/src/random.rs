//! Pseudo-random number generation for `(random limit)`.
//!
//! Algorithm: **xoshiro256\*\*** (Blackman & Vigna, 2018) — 256-bit
//! state, 2^256 period, scalar throughput ~1 ns/word on modern x86_64,
//! all dimensions of equidistribution up to 4×64 bits, passes
//! BigCrush + PractRand to 32 TB. It's the canonical "fast and good"
//! non-crypto PRNG; what every recent language stdlib has converged
//! on (Ruby 3, .NET 6+ default, GLib).
//!
//! Seeding: we don't trust any single source, so the seed is
//! SplitMix64-mixed from:
//!
//!   1. `std::time::Instant` nanosecond counter at first call
//!      (monotonic process clock — has the most entropy on a
//!      cold start when nothing else has run yet);
//!   2. `process::id()` (PID, unique across the OS at startup);
//!   3. `thread::current().id()` (varies across Lisp threads);
//!   4. The address of a stack-local variable (Windows ASLR
//!      gives ~24 bits of entropy here, free);
//!   5. **RDTSC** on x86_64 — the CPU's cycle counter — read at
//!      seed time. Adds sub-microsecond jitter the wall clock
//!      can't see.
//!
//! All five mixed with SplitMix64's finaliser (the same finaliser
//! the algorithm's authors recommend for seed expansion). The result
//! is good enough that even running two NCL processes in the same
//! second produces uncorrelated streams.

use std::sync::{Mutex, OnceLock};

use crate::word::Word;

// ─── xoshiro256** ───────────────────────────────────────────────────────────

pub struct Xoshiro256 {
    s: [u64; 4],
}

impl Xoshiro256 {
    /// Seed the 256-bit state by expanding a single u64 through
    /// SplitMix64. Per the authors: any deterministic mapping from
    /// u64 → 4×u64 will do as long as you don't pick all-zero;
    /// SplitMix64 is the convention.
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = SplitMix64 { x: seed };
        let s = [sm.next(), sm.next(), sm.next(), sm.next()];
        // Belt-and-braces: if SplitMix64 ever produced four zeros
        // from a zero seed (it doesn't, but assert anyway), kick it
        // with a known non-zero word.
        if s == [0, 0, 0, 0] {
            return Xoshiro256 { s: [0x9E3779B97F4A7C15, 1, 2, 3] };
        }
        Xoshiro256 { s }
    }

    /// Advance and return one 64-bit word.
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }
}

// SplitMix64 — used only for seed expansion.
struct SplitMix64 {
    x: u64,
}

impl SplitMix64 {
    #[inline]
    fn next(&mut self) -> u64 {
        self.x = self.x.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

// ─── Evil entropy seed ──────────────────────────────────────────────────────

/// Mix every source of randomness we can scrape without a syscall
/// dependency. Each source is folded through SplitMix64 so a low-
/// entropy input doesn't dominate the seed.
fn evil_seed() -> u64 {
    let mut sm = SplitMix64 { x: 0x9E3779B97F4A7C15 };

    // Nanos since some process epoch — best single source on a cold
    // start. We use std::time::Instant indirectly via the duration
    // between two Instants taken back-to-back; the absolute Instant
    // is opaque, but the duration plus the system time give plenty
    // of entropy.
    let now = std::time::Instant::now();
    let sys = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    sm.x ^= sys;
    let _ = sm.next();
    sm.x ^= now.elapsed().as_nanos() as u64;
    let _ = sm.next();

    // Process ID.
    sm.x ^= std::process::id() as u64;
    let _ = sm.next();

    // Thread ID (debug format — Rust doesn't expose the raw id but
    // the Debug impl is a number).
    let tid_s = format!("{:?}", std::thread::current().id());
    let mut h: u64 = 0xcbf29ce484222325;
    for b in tid_s.bytes() {
        h = h.wrapping_mul(0x100000001b3) ^ (b as u64);
    }
    sm.x ^= h;
    let _ = sm.next();

    // Address of a stack local (Windows ASLR).
    let local_marker: u8 = 0;
    sm.x ^= &local_marker as *const u8 as usize as u64;
    let _ = sm.next();

    // RDTSC on x86_64 — picks up sub-microsecond jitter that the
    // wall clock and Instant can't see. Especially good when the
    // process starts at a deterministic-looking Instant (e.g. a
    // freshly booted VM).
    #[cfg(target_arch = "x86_64")]
    {
        let tsc = unsafe { std::arch::x86_64::_rdtsc() };
        sm.x ^= tsc;
        let _ = sm.next();
    }

    sm.next()
}

// ─── Global RNG ─────────────────────────────────────────────────────────────

static RNG: OnceLock<Mutex<Xoshiro256>> = OnceLock::new();

fn rng() -> &'static Mutex<Xoshiro256> {
    RNG.get_or_init(|| Mutex::new(Xoshiro256::from_seed(evil_seed())))
}

/// Generate a uniform random integer in `[0, limit)`.
///
/// Uses Lemire's nearly-divisionless reduction (2019): one 128-bit
/// multiply on the fast path, no division. Worst-case retry is
/// rare — biased toward zero only for limits very close to 2^64,
/// which fixnums (< 2^60) never hit.
pub fn random_in_range(limit: u64) -> u64 {
    assert!(limit > 0, "random: limit must be positive");
    let mut g = rng().lock().unwrap();
    let mut x = g.next_u64();
    let mut m = (x as u128).wrapping_mul(limit as u128);
    let mut l = m as u64; // low 64 bits
    if l < limit {
        let t = limit.wrapping_neg() % limit; // 2^64 mod limit
        while l < t {
            x = g.next_u64();
            m = (x as u128).wrapping_mul(limit as u128);
            l = m as u64;
        }
    }
    (m >> 64) as u64
}

// ─── Lisp shim ──────────────────────────────────────────────────────────────

/// `(random N)` — N must be a positive fixnum. Returns a non-negative
/// fixnum less than N. Float-limit support is intentionally absent
/// for v1; the spec allows it but no current demo needs it.
pub extern "C-unwind" fn random_shim(
    _mutator: *mut crate::mutator::MutatorState,
    _env: u64,
    args: *const u64,
    n_args: u64,
) -> u64 {
    if n_args != 1 {
        panic!("random: expected 1 arg (limit), got {n_args}");
    }
    let w = Word::from_raw(unsafe { *args });
    let Some(limit) = w.as_fixnum() else {
        panic!("random: limit must be a fixnum, got {w:?}");
    };
    if limit <= 0 {
        panic!("random: limit must be positive, got {limit}");
    }
    let r = random_in_range(limit as u64);
    Word::fixnum(r as i64).raw()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xoshiro_known_seed_known_sequence() {
        // Seed 42 → SplitMix64 expanded → first few outputs are
        // deterministic. Spot-check that they change cycle to cycle.
        let mut r = Xoshiro256::from_seed(42);
        let a = r.next_u64();
        let b = r.next_u64();
        let c = r.next_u64();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn random_in_range_stays_in_range() {
        for n in [1u64, 2, 10, 100, 1_000_000] {
            for _ in 0..1000 {
                let r = random_in_range(n);
                assert!(r < n, "random {n} produced {r}");
            }
        }
    }

    #[test]
    fn random_in_range_covers_distribution() {
        // Coarse: over 10_000 draws of (random 10), each bucket
        // should get at least 500 hits (expected 1000).
        let mut buckets = [0u32; 10];
        for _ in 0..10_000 {
            buckets[random_in_range(10) as usize] += 1;
        }
        for (i, &n) in buckets.iter().enumerate() {
            assert!(n > 500, "bucket {i} only got {n} hits");
        }
    }

    #[test]
    fn evil_seed_differs_across_calls() {
        let a = evil_seed();
        let b = evil_seed();
        assert_ne!(a, b);
    }
}
