//! WGSL shader source + naga structural validation.
//!
//! Phase 2 of the parallel-simulation plan ports MPPI to WGSL compute
//! shaders. This module owns the shader source (via `include_str!`) and
//! validates it structurally using naga — wgpu's reference WGSL
//! parser/validator, re-exported as `wgpu::naga`.
//!
//! Validation is purely structural (no GPU dispatch) because CI and common
//! development environments do not have GPU access. naga parses the WGSL,
//! type-checks every function body, and validates the module. If naga
//! accepts the shader, the WGSL is syntactically valid and type-safe.
//! Numerical correctness remains to be verified once a GPU is available.
//!
//! This module is gated behind the `gpu` feature because naga is pulled in
//! via `wgpu` (re-export), which is itself optional.
//!
//! # Public surface
//!
//! - [`MPPI_HELPERS_WGSL`] — the shader source as a `&'static str`.
//! - [`validate_mppi_helpers`] — parse + validate, return the naga module.
//! - [`ValidationError`] — a simple `Parse` / `Validate` error enum that
//!   stringifies naga's internal error types so callers don't have to depend
//!   on naga details.

use wgpu::naga;

/// The mppi_helpers WGSL source, embedded at compile time.
pub const MPPI_HELPERS_WGSL: &str = include_str!("shaders/mppi_helpers.wgsl");

/// Error returned by [`validate_mppi_helpers`]. The naga error types carry a
/// lot of internal machinery; we stringify them here so callers can just
/// print the message.
#[derive(Debug)]
pub enum ValidationError {
    /// WGSL parse failed.
    Parse(String),
    /// WGSL parsed but module validation failed.
    Validate(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::Parse(msg) => write!(f, "WGSL parse error:\n{msg}"),
            ValidationError::Validate(msg) => write!(f, "WGSL validation error:\n{msg}"),
        }
    }
}

impl std::error::Error for ValidationError {}

/// Parse and validate the mppi_helpers shader.
///
/// Returns the parsed [`naga::Module`] on success. The module can be
/// introspected (e.g. for tests that verify struct sizes or function
/// presence) without dispatching any GPU work.
pub fn validate_mppi_helpers() -> Result<naga::Module, ValidationError> {
    let module = naga::front::wgsl::parse_str(MPPI_HELPERS_WGSL)
        .map_err(|e| ValidationError::Parse(e.emit_to_string(MPPI_HELPERS_WGSL)))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    validator
        .validate(&module)
        .map_err(|e| ValidationError::Validate(format!("{e:?}")))?;

    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 12 helper functions this slice is required to provide. If any of
    /// these are renamed or dropped, downstream kernel code (future slices)
    /// will break, so guard the names here.
    const EXPECTED_FN_NAMES: &[&str] = &[
        "quat_mul",
        "quat_normalize",
        "quat_rotate",
        "motor_mixing",
        "state_derivative",
        "rk4_step",
        "box_sdf",
        "sphere_sdf",
        "cylinder_sdf",
        "obstacle_sdf",
        "boundary_sdf",
        "arena_sdf",
    ];

    /// Structs whose byte layout must match the Rust-side POD structs in
    /// `crates/batch-sim/src/gpu/pipeline.rs`. All three are 48 bytes.
    const EXPECTED_STRUCT_SIZES: &[(&str, u32)] =
        &[("DroneParams", 48), ("CostWeights", 48), ("Obstacle", 48)];

    #[test]
    fn test_wgsl_parses() {
        // This is the top-level structural check: naga parses the shader and
        // validates all function bodies. A failure here means the shader is
        // broken and no GPU would accept it.
        match validate_mppi_helpers() {
            Ok(_) => {}
            Err(e) => panic!("WGSL validation failed:\n{e}"),
        }
    }

    #[test]
    fn test_wgsl_defines_all_helper_fns() {
        let module = validate_mppi_helpers().expect("validation failed");

        for expected in EXPECTED_FN_NAMES {
            let found = module
                .functions
                .iter()
                .any(|(_, f)| f.name.as_deref() == Some(*expected));
            assert!(
                found,
                "expected function `{expected}` not found in shader module; \
                 present functions: {:?}",
                module
                    .functions
                    .iter()
                    .filter_map(|(_, f)| f.name.clone())
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn test_wgsl_struct_sizes_match_rust_pod() {
        let module = validate_mppi_helpers().expect("validation failed");

        // `TypeInner::Struct { span, .. }` stores the total byte size after
        // WGSL layout rules, which is what naga uses for validation and what
        // the GPU would see. We compare against the `repr(C)` sizes of the
        // corresponding Rust POD structs (48 bytes each).
        for (name, expected_size) in EXPECTED_STRUCT_SIZES {
            let ty = module
                .types
                .iter()
                .find(|(_, ty)| ty.name.as_deref() == Some(*name))
                .unwrap_or_else(|| panic!("struct `{name}` not found in shader module"))
                .1;
            let span = match &ty.inner {
                naga::TypeInner::Struct { span, .. } => *span,
                other => panic!("type `{name}` is not a struct: {other:?}"),
            };
            assert_eq!(
                span, *expected_size,
                "WGSL struct `{name}` size {span} != expected {expected_size} \
                 (Rust POD size in pipeline.rs)",
            );
        }
    }

    /// Cross-check: the Rust POD structs really are 48 bytes. If this ever
    /// regresses, the sibling WGSL test will also fail — but this test
    /// fails earlier with a clearer message.
    #[test]
    fn test_rust_pod_sizes_are_48() {
        use crate::gpu::pipeline::{CostWeightsGpu, DroneParamsGpu, ObstacleGpu};
        assert_eq!(std::mem::size_of::<DroneParamsGpu>(), 48);
        assert_eq!(std::mem::size_of::<CostWeightsGpu>(), 48);
        assert_eq!(std::mem::size_of::<ObstacleGpu>(), 48);
    }
}
