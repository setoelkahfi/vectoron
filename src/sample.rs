//! Choosing the next token from a vector of logits.

use std::collections::HashMap;

/// How [`Sampler`] picks a token.
#[derive(Debug, Clone, PartialEq)]
pub struct SamplerConfig {
    /// At or below `1e-7` the sampler is greedy and always takes the most
    /// likely token.
    pub temperature: f32,
    /// Keep only the `k` most likely tokens.
    pub top_k: Option<usize>,
    /// Keep the most likely tokens whose probabilities sum to at least `p`.
    pub top_p: Option<f32>,
    /// Subtracted from a token's logit once per earlier occurrence.
    pub frequency_penalty: f32,
    /// Subtracted once from the logit of any token seen before.
    pub presence_penalty: f32,
    /// Seed for the random draw. The same seed and inputs give the same tokens.
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: None,
            top_p: Some(0.95),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 0,
        }
    }
}

/// Picks tokens from logits according to a [`SamplerConfig`].
#[derive(Debug, Clone)]
pub struct Sampler {
    config: SamplerConfig,
    rng: SplitMix64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        let rng = SplitMix64(config.seed);
        Self { config, rng }
    }

    /// A sampler that always takes the most likely token.
    pub fn greedy() -> Self {
        Self::new(SamplerConfig {
            temperature: 0.0,
            top_p: None,
            ..SamplerConfig::default()
        })
    }

    /// Picks the next token. `history` is the tokens generated so far, used
    /// only by the penalties. `logits` is modified.
    pub fn sample(&mut self, logits: &mut [f32], history: &[u32]) -> u32 {
        let c = &self.config;
        if c.frequency_penalty != 0.0 || c.presence_penalty != 0.0 {
            let mut counts: HashMap<u32, u32> = HashMap::new();
            for &id in history {
                *counts.entry(id).or_default() += 1;
            }
            for (id, n) in counts {
                if let Some(l) = logits.get_mut(id as usize) {
                    *l -= c.frequency_penalty * n as f32 + c.presence_penalty;
                }
            }
        }

        if c.temperature <= 1e-7 {
            return argmax(logits);
        }

        let mut order: Vec<u32> = (0..logits.len() as u32).collect();
        let by_logit = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
        if let Some(k) = c.top_k.filter(|k| *k > 0 && *k < order.len()) {
            order.select_nth_unstable_by(k - 1, by_logit);
            order.truncate(k);
        }
        order.sort_unstable_by(by_logit);

        let max = logits[order[0] as usize];
        let mut probs: Vec<f32> = order
            .iter()
            .map(|&i| ((logits[i as usize] - max) / c.temperature).exp())
            .collect();
        let total: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= total;
        }

        if let Some(top_p) = c.top_p.filter(|p| *p < 1.0) {
            let mut sum = 0.0;
            let mut keep = probs.len();
            for (i, p) in probs.iter().enumerate() {
                sum += p;
                if sum >= top_p {
                    keep = i + 1;
                    break;
                }
            }
            probs.truncate(keep);
            order.truncate(keep);
        }

        let total: f32 = probs.iter().sum();
        let mut r = self.rng.next_f32() * total;
        for (i, p) in probs.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return order[i];
            }
        }
        order[probs.len() - 1]
    }
}

/// Index of the largest value, the first one on ties.
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in logits.iter().enumerate() {
        if *v > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[derive(Debug, Clone)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_takes_the_first_maximum() {
        let mut l = vec![0.1, 3.0, 3.0, -1.0];
        assert_eq!(Sampler::greedy().sample(&mut l, &[]), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(SamplerConfig {
            top_k: Some(1),
            top_p: None,
            ..SamplerConfig::default()
        });
        for _ in 0..20 {
            let mut l = vec![0.1, 0.2, 5.0, 0.3];
            assert_eq!(s.sample(&mut l, &[]), 2);
        }
    }

    #[test]
    fn same_seed_same_tokens() {
        let config = SamplerConfig {
            temperature: 1.0,
            top_p: None,
            seed: 42,
            ..SamplerConfig::default()
        };
        let draw = |config: &SamplerConfig| {
            let mut s = Sampler::new(config.clone());
            (0..16)
                .map(|_| s.sample(&mut [0.0; 10], &[]))
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(&config), draw(&config));
    }

    #[test]
    fn penalties_lower_repeated_tokens() {
        let mut s = Sampler::new(SamplerConfig {
            temperature: 0.0,
            frequency_penalty: 1.0,
            ..SamplerConfig::default()
        });
        let mut l = vec![2.5, 2.0];
        // token 0 appeared once: 2.5 - 1.0 = 1.5 < 2.0
        assert_eq!(s.sample(&mut l, &[0]), 1);
    }
}
