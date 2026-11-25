//! The Qwen2 decoder (also used by Qwen2.5), loaded from GGUF.

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::gguf::Gguf;
use crate::simd;
use crate::tensor::{
    Input, InputCache, Kernels, QMatrix, add_bias, load_vector, rms_norm, rope, silu, softmax,
};

/// Hyperparameters, read from the GGUF metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2Config {
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub context_length: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

impl Qwen2Config {
    /// Reads the configuration of a `qwen2` GGUF file.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let arch = gguf
            .get("general.architecture")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::Metadata("general.architecture".into()))?;
        if arch != "qwen2" {
            return Err(Error::Model(format!(
                "expected a qwen2 model, found `{arch}`"
            )));
        }
        let int = |key: &str| -> Result<usize> {
            gguf.get(key)
                .and_then(|v| v.as_u64())
                .and_then(|v| usize::try_from(v).ok())
                .ok_or_else(|| Error::Metadata(key.into()))
        };
        let float = |key: &str| -> Result<f32> {
            gguf.get(key)
                .and_then(|v| v.as_f32())
                .ok_or_else(|| Error::Metadata(key.into()))
        };

        let n_heads = int("qwen2.attention.head_count")?;
        let n_kv_heads = int("qwen2.attention.head_count_kv")?;
        let hidden = int("qwen2.embedding_length")?;
        if n_heads == 0 || n_kv_heads == 0 || hidden % n_heads != 0 || n_heads % n_kv_heads != 0 {
            return Err(Error::Model(format!(
                "inconsistent head counts: {n_heads} heads, {n_kv_heads} key/value heads, width {hidden}"
            )));
        }
        let head_dim = hidden / n_heads;
        if head_dim % 2 != 0 {
            return Err(Error::Model(format!("head size {head_dim} is odd")));
        }
        let embed = QMatrix::from_gguf(gguf, "token_embd.weight")?;

        Ok(Self {
            n_layers: int("qwen2.block_count")?,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden,
            ffn: int("qwen2.feed_forward_length")?,
            vocab: embed.rows,
            context_length: int("qwen2.context_length")?,
            rms_eps: float("qwen2.attention.layer_norm_rms_epsilon")?,
            rope_theta: float("qwen2.rope.freq_base").unwrap_or(10_000.0),
        })
    }

    fn kv_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
}

struct Layer {
    attn_norm: Vec<f32>,
    wq: QMatrix,
    wk: QMatrix,
    wv: QMatrix,
    bq: Vec<f32>,
    bk: Vec<f32>,
    bv: Vec<f32>,
    wo: QMatrix,
    ffn_norm: Vec<f32>,
    gate: QMatrix,
    up: QMatrix,
    down: QMatrix,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
}

/// A loaded Qwen2 model with its key/value cache.
///
/// The model keeps the GGUF file in memory as it was read and computes
/// with the quantized weights in place.
pub struct Qwen2Model {
    config: Qwen2Config,
    data: Arc<Vec<u8>>,
    embed: QMatrix,
    layers: Vec<Layer>,
    output_norm: Vec<f32>,
    lm_head: QMatrix,
    inv_freq: Vec<f32>,
    cache_len: usize,
    kernels: Kernels,
}

impl Qwen2Model {
    /// Parses a GGUF file and loads the model from it.
    pub fn load(bytes: Vec<u8>) -> Result<Self> {
        Self::from_gguf(&Gguf::parse(bytes)?)
    }

    /// Loads the model from a parsed GGUF file. The file's buffer is
    /// shared, not copied.
    pub fn from_gguf(gguf: &Gguf) -> Result<Self> {
        let config = Qwen2Config::from_gguf(gguf)?;
        let embed = QMatrix::from_gguf(gguf, "token_embd.weight")?;
        // Models with tied embeddings have no separate output matrix.
        let lm_head = match gguf.tensor("output.weight") {
            Some(info) => QMatrix::new(info)?,
            None => embed.clone(),
        };

        let expect = |m: &QMatrix, name: &str, rows: usize, cols: usize| -> Result<()> {
            if m.rows == rows && m.cols == cols {
                Ok(())
            } else {
                Err(Error::Model(format!(
                    "`{name}` is {}x{}, expected {rows}x{cols}",
                    m.rows, m.cols
                )))
            }
        };
        expect(&embed, "token_embd.weight", config.vocab, config.hidden)?;
        expect(&lm_head, "output.weight", config.vocab, config.hidden)?;

        let q_dim = config.n_heads * config.head_dim;
        let kv_dim = config.kv_dim();
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let name = |part: &str| format!("blk.{i}.{part}");
            let mat = |part: &str| QMatrix::from_gguf(gguf, &name(part));
            let vec = |part: &str| load_vector(gguf, &name(part));
            let layer = Layer {
                attn_norm: vec("attn_norm.weight")?,
                wq: mat("attn_q.weight")?,
                wk: mat("attn_k.weight")?,
                wv: mat("attn_v.weight")?,
                bq: vec("attn_q.bias")?,
                bk: vec("attn_k.bias")?,
                bv: vec("attn_v.bias")?,
                wo: mat("attn_output.weight")?,
                ffn_norm: vec("ffn_norm.weight")?,
                gate: mat("ffn_gate.weight")?,
                up: mat("ffn_up.weight")?,
                down: mat("ffn_down.weight")?,
                k_cache: Vec::new(),
                v_cache: Vec::new(),
            };
            expect(&layer.wq, &name("attn_q.weight"), q_dim, config.hidden)?;
            expect(&layer.wk, &name("attn_k.weight"), kv_dim, config.hidden)?;
            expect(&layer.wv, &name("attn_v.weight"), kv_dim, config.hidden)?;
            expect(&layer.wo, &name("attn_output.weight"), config.hidden, q_dim)?;
            expect(
                &layer.gate,
                &name("ffn_gate.weight"),
                config.ffn,
                config.hidden,
            )?;
            expect(&layer.up, &name("ffn_up.weight"), config.ffn, config.hidden)?;
            expect(
                &layer.down,
                &name("ffn_down.weight"),
                config.hidden,
                config.ffn,
            )?;
            if layer.attn_norm.len() != config.hidden
                || layer.ffn_norm.len() != config.hidden
                || layer.bq.len() != q_dim
                || layer.bk.len() != kv_dim
                || layer.bv.len() != kv_dim
            {
                return Err(Error::Model(format!(
                    "layer {i} has mis-sized norms or biases"
                )));
            }
            layers.push(layer);
        }
        let output_norm = load_vector(gguf, "output_norm.weight")?;
        if output_norm.len() != config.hidden {
            return Err(Error::Model("`output_norm.weight` is mis-sized".into()));
        }

        let inv_freq = (0..config.head_dim)
            .step_by(2)
            .map(|i| 1.0 / config.rope_theta.powf(i as f32 / config.head_dim as f32))
            .collect();

        Ok(Self {
            config,
            data: Arc::clone(gguf.bytes()),
            embed,
            layers,
            output_norm,
            lm_head,
            inv_freq,
            cache_len: 0,
            kernels: Kernels::default(),
        })
    }

    /// Chooses the arithmetic of the matrix products. [`Kernels::Integer`]
    /// is the default; [`Kernels::F32`] is the slower reference. The cached
    /// keys and values stay valid, but replies may differ between the two.
    pub fn set_kernels(&mut self, kernels: Kernels) {
        self.kernels = kernels;
    }

    /// The model's hyperparameters.
    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    /// Number of positions held in the key/value cache.
    pub fn cache_len(&self) -> usize {
        self.cache_len
    }

    /// Empties the key/value cache so the next call starts at position 0.
    pub fn clear_kv_cache(&mut self) {
        self.truncate_cache(0);
    }

    fn truncate_cache(&mut self, len: usize) {
        let kv_dim = self.config.kv_dim();
        for layer in &mut self.layers {
            layer.k_cache.truncate(len * kv_dim);
            layer.v_cache.truncate(len * kv_dim);
        }
        self.cache_len = len;
    }

    /// Runs `tokens` at positions `pos..pos + tokens.len()` and returns
    /// the logits for the last one.
    ///
    /// `pos` may be at most the current cache length. A smaller `pos`
    /// drops the cached positions from `pos` on, so a caller can rewind.
    pub fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Vec<f32>> {
        let t = tokens.len();
        if t == 0 {
            return Err(Error::Model("no tokens to run".into()));
        }
        if pos > self.cache_len {
            return Err(Error::Model(format!(
                "position {pos} is past the {} cached positions",
                self.cache_len
            )));
        }
        if pos + t > self.config.context_length {
            return Err(Error::Model(format!(
                "{} positions exceed the context length of {}",
                pos + t,
                self.config.context_length
            )));
        }
        if let Some(&bad) = tokens.iter().find(|&&id| id as usize >= self.config.vocab) {
            return Err(Error::Model(format!(
                "token {bad} is outside the vocabulary"
            )));
        }
        self.truncate_cache(pos);

        let cfg = self.config.clone();
        let data = Arc::clone(&self.data);
        let data = data.as_slice();
        let (hidden, hd) = (cfg.hidden, cfg.head_dim);
        let q_dim = cfg.n_heads * hd;
        let kv_dim = cfg.kv_dim();
        let group = cfg.n_heads / cfg.n_kv_heads;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut x = vec![0f32; t * hidden];
        for (row, &id) in x.chunks_exact_mut(hidden).zip(tokens) {
            self.embed.dequantize_row(data, id as usize, row);
        }

        let mut h = vec![0f32; t * hidden];
        let mut q = vec![0f32; t * q_dim];
        let mut k = vec![0f32; t * kv_dim];
        let mut v = vec![0f32; t * kv_dim];
        let mut attn = vec![0f32; t * q_dim];
        let mut proj = vec![0f32; t * hidden];
        let mut gate = vec![0f32; t * cfg.ffn];
        let mut up = vec![0f32; t * cfg.ffn];
        let mut scores = Vec::with_capacity(pos + t);
        let kernels = self.kernels;
        // Quantized copies of whichever vectors the current matrices read.
        let mut cache = InputCache::default();

        for layer in &mut self.layers {
            h.copy_from_slice(&x);
            rms_norm(&mut h, &layer.attn_norm, cfg.rms_eps);

            let mut input = Input::new(&h, t, kernels, &mut cache);
            layer.wq.matmul(data, &mut input, &mut q);
            layer.wk.matmul(data, &mut input, &mut k);
            layer.wv.matmul(data, &mut input, &mut v);
            add_bias(&mut q, &layer.bq);
            add_bias(&mut k, &layer.bk);
            add_bias(&mut v, &layer.bv);

            for ti in 0..t {
                let p = pos + ti;
                for head in q[ti * q_dim..][..q_dim].chunks_exact_mut(hd) {
                    rope(head, p, &self.inv_freq);
                }
                for head in k[ti * kv_dim..][..kv_dim].chunks_exact_mut(hd) {
                    rope(head, p, &self.inv_freq);
                }
            }
            layer.k_cache.extend_from_slice(&k);
            layer.v_cache.extend_from_slice(&v);

            attn.fill(0.0);
            for ti in 0..t {
                let visible = pos + ti + 1;
                for head in 0..cfg.n_heads {
                    let kv_off = (head / group) * hd;
                    let qh = &q[ti * q_dim + head * hd..][..hd];
                    scores.clear();
                    for p in 0..visible {
                        let kh = &layer.k_cache[p * kv_dim + kv_off..][..hd];
                        scores.push(simd::dot(qh, kh) * scale);
                    }
                    softmax(&mut scores);
                    let out = &mut attn[ti * q_dim + head * hd..][..hd];
                    for (p, &w) in scores.iter().enumerate() {
                        let vh = &layer.v_cache[p * kv_dim + kv_off..][..hd];
                        simd::axpy(w, vh, out);
                    }
                }
            }

            let mut input = Input::new(&attn, t, kernels, &mut cache);
            layer.wo.matmul(data, &mut input, &mut proj);
            for (x, p) in x.iter_mut().zip(&proj) {
                *x += p;
            }

            h.copy_from_slice(&x);
            rms_norm(&mut h, &layer.ffn_norm, cfg.rms_eps);
            let mut input = Input::new(&h, t, kernels, &mut cache);
            layer.gate.matmul(data, &mut input, &mut gate);
            layer.up.matmul(data, &mut input, &mut up);
            for (g, u) in gate.iter_mut().zip(&up) {
                *g = silu(*g) * u;
            }
            let mut input = Input::new(&gate, t, kernels, &mut cache);
            layer.down.matmul(data, &mut input, &mut proj);
            for (x, p) in x.iter_mut().zip(&proj) {
                *x += p;
            }
        }
        self.cache_len = pos + t;

        let mut last = x[(t - 1) * hidden..].to_vec();
        rms_norm(&mut last, &self.output_norm, cfg.rms_eps);
        let mut logits = vec![0f32; cfg.vocab];
        let mut input = Input::new(&last, 1, kernels, &mut cache);
        self.lm_head.matmul(data, &mut input, &mut logits);
        Ok(logits)
    }
}
