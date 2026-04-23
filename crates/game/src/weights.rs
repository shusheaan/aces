//! Shared binary MLP weight loader used by both `policy.rs` and `perception.rs`.
//!
//! Binary format (little-endian, written by `aces/export.py`):
//!   u32 num_layers
//!   for each layer:
//!     u32 rows
//!     u32 cols
//!     rows * cols f32 weights (row-major)
//!     rows f32 biases

use nalgebra::{DMatrix, DVector};

/// Load MLP weights from a binary file.
///
/// Returns `Some(layers)` where each entry is `(weight_matrix, bias_vector)`,
/// or `None` on any read/parse error.
pub fn load_mlp_weights(path: &str) -> Option<Vec<(DMatrix<f64>, DVector<f64>)>> {
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

    Some(layers)
}
