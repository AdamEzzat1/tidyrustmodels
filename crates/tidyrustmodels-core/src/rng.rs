//! Reproducibility policy: explicit seeds and bounded parallelism.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

/// An explicit random seed. Every stochastic API takes one; there is no global RNG.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Seed(pub u64);

impl Seed {
    /// A fresh ChaCha8 generator for this seed. ChaCha is platform-independent, so the
    /// same seed yields the same stream on every OS and architecture.
    pub fn rng(self) -> ChaCha8Rng {
        ChaCha8Rng::seed_from_u64(self.0)
    }

    /// Derive a sub-seed for an independent stream (e.g. one per tuning candidate).
    /// Uses SplitMix64 mixing so neighbouring indices are uncorrelated.
    pub fn derive(self, index: u64) -> Seed {
        let mut z = self
            .0
            .wrapping_add(0x9E37_79B9_7F4A_7C15u64.wrapping_mul(index.wrapping_add(1)));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Seed(z ^ (z >> 31))
    }
}

/// How much parallelism a resampling or tuning call may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Parallelism {
    /// `None` runs serially on the calling thread. `Some(k)` builds a scoped pool of
    /// `k` threads for the duration of the call; results are identical to serial.
    pub threads: Option<usize>,
}

impl Parallelism {
    /// Serial execution.
    pub const SERIAL: Parallelism = Parallelism { threads: None };

    /// Parallel execution with `k` threads.
    pub fn threads(k: usize) -> Self {
        Parallelism {
            threads: Some(k.max(1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Seed(7).rng();
        let mut b = Seed(7).rng();
        assert_eq!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn derived_seeds_differ() {
        let s = Seed(1);
        assert_ne!(s.derive(0), s.derive(1));
        assert_ne!(s.derive(0), s);
    }
}
