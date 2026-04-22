//! Perception NN loader and inference for the neurosymbolic pipeline.
//!
//! Mirrors `aces/perception.py`. Loads a trained MLP that maps 21-dim
//! observations to semantic features consumed by [`crate::fsm::SymbolicFsm`].
//!
//! Binary format is identical to `policy.bin` (see `crates/game/src/policy.rs`
//! and `aces/export.py`).
//!
//! Architecture exported by `perception.py::export_perception`:
//!   Layer 0: 21 → 64  (Tanh)
//!   Layer 1: 64 → 64  (Tanh)
//!   Layer 2: 64 → 8   (split: continuous [0..5) + intent logits [5..8))
//!
//! Post-processing on layer-2 outputs (raw logits):
//!   outputs[0..4] → sigmoid  → threat, opportunity, collision_risk, uncertainty
//!   outputs[4]    → softplus → opponent_distance
//!   outputs[5..8] → argmax  → opponent_intent (u8: 0=approach,1=flee,2=patrol)

use nalgebra::{DMatrix, DVector};

use crate::fsm::SemanticFeatures;

/// Loaded perception MLP.
pub struct PerceptionMlp {
    layers: Vec<(DMatrix<f64>, DVector<f64>)>,
}

impl PerceptionMlp {
    /// Try to load a perception network from a binary file.
    /// Returns `None` on any read/parse error.
    pub fn load(path: &str) -> Option<Self> {
        let data = std::fs::read(path).ok()?;
        let mut cur = 0_usize;

        let read_u32 = |c: &mut usize| -> Option<u32> {
            if *c + 4 > data.len() {
                return None;
            }
            let v = u32::from_le_bytes(data[*c..*c + 4].try_into().ok()?);
            *c += 4;
            Some(v)
        };

        let num_layers = read_u32(&mut cur)? as usize;
        let mut layers = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            let rows = read_u32(&mut cur)? as usize;
            let cols = read_u32(&mut cur)? as usize;

            // Weight matrix: rows × cols f32, row-major
            let w_len = rows * cols;
            let w_bytes = w_len * 4;
            if cur + w_bytes > data.len() {
                return None;
            }
            let mut w_data = Vec::with_capacity(w_len);
            for i in 0..w_len {
                let off = cur + i * 4;
                let v = f32::from_le_bytes(data[off..off + 4].try_into().ok()?);
                w_data.push(v as f64);
            }
            cur += w_bytes;
            let weight = DMatrix::from_row_slice(rows, cols, &w_data);

            // Bias vector: rows f32
            let b_bytes = rows * 4;
            if cur + b_bytes > data.len() {
                return None;
            }
            let mut b_data = Vec::with_capacity(rows);
            for i in 0..rows {
                let off = cur + i * 4;
                let v = f32::from_le_bytes(data[off..off + 4].try_into().ok()?);
                b_data.push(v as f64);
            }
            cur += b_bytes;
            let bias = DVector::from_vec(b_data);

            layers.push((weight, bias));
        }

        // Sanity check: first layer input should be 21 (vector obs)
        if let Some((w, _)) = layers.first() {
            if w.ncols() != 21 {
                eprintln!(
                    "[ACES] Warning: perception input dim is {} (expected 21)",
                    w.ncols()
                );
            }
        }

        Some(Self { layers })
    }

    /// Forward pass: 21-dim observation → [`SemanticFeatures`].
    ///
    /// Hidden layers use Tanh; final layer splits into continuous (sigmoid /
    /// softplus) outputs and intent logits (argmax).
    pub fn infer(&self, obs: &[f64; 21]) -> SemanticFeatures {
        let mut x = DVector::from_column_slice(obs);

        // All layers except the last use Tanh activation.
        let last = self.layers.len().saturating_sub(1);
        for (i, (w, b)) in self.layers.iter().enumerate() {
            x = w * &x + b;
            if i < last {
                x = x.map(|v| v.tanh());
            }
        }

        // x now contains raw outputs of the final (merged) layer.
        // Expected layout: [threat_raw, opp_raw, coll_raw, unc_raw, dist_raw,
        //                    intent0, intent1, intent2]

        let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
        // softplus: ln(1 + exp(v)), numerically stable form
        let softplus = |v: f64| {
            if v > 20.0 {
                v
            } else {
                (1.0_f64 + v.exp()).ln()
            }
        };

        let threat = sigmoid(x[0]);
        let opportunity = sigmoid(x[1]);
        let collision_risk = sigmoid(x[2]);
        let uncertainty = sigmoid(x[3]);
        let opponent_distance = softplus(x[4]);

        // Intent: argmax over logits [5..8)
        let intent_opponent_intent = if x.len() >= 8 {
            let i0 = x[5];
            let i1 = x[6];
            let i2 = x[7];
            if i0 >= i1 && i0 >= i2 {
                0u8
            } else if i1 >= i2 {
                1u8
            } else {
                2u8
            }
        } else {
            0u8
        };

        SemanticFeatures {
            threat,
            opportunity,
            collision_risk,
            uncertainty,
            opponent_distance,
            opponent_intent: intent_opponent_intent,
        }
    }
}
