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
//! - [`MPPI_HELPERS_WGSL`] — helpers-only shader source.
//! - [`MPPI_ROLLOUT_WGSL`] — rollout-kernel shader source.
//! - [`full_mppi_source`] — concatenation of helpers + rollout, ready to
//!   feed into `wgpu::ShaderSource::Wgsl`.
//! - [`validate_mppi_helpers`] — parse + validate the helpers module.
//! - [`validate_full_mppi`] — parse + validate the concatenated source
//!   (helpers + rollout kernel with all bindings).
//! - [`ValidationError`] — a simple `Parse` / `Validate` error enum that
//!   stringifies naga's internal error types so callers don't have to
//!   depend on naga details.

use wgpu::naga;

/// The mppi_helpers WGSL source, embedded at compile time.
pub const MPPI_HELPERS_WGSL: &str = include_str!("shaders/mppi_helpers.wgsl");

/// The rollout-kernel WGSL source (bindings + `rollout_and_cost`),
/// embedded at compile time. Must be concatenated with
/// [`MPPI_HELPERS_WGSL`] for standalone use — see [`full_mppi_source`].
pub const MPPI_ROLLOUT_WGSL: &str = include_str!("shaders/mppi_rollout.wgsl");

/// The full MPPI shader source: helpers first, then the rollout kernel.
///
/// WGSL has no `#include`, so helpers must appear in the same translation
/// unit as the kernel. The helpers file owns struct definitions
/// (`DroneParams`, `CostWeights`, `Obstacle`, `DroneState`,
/// `StateDerivative`) and all 12 helper functions; the rollout file adds
/// `MppiDims`, the 11 bind-group declarations, stage-cost wrappers, and
/// the `@compute` entry point.
///
/// A newline is inserted between the two files to guard against a file
/// that ends without one — WGSL otherwise fuses the last token of the
/// helpers with the first token of the rollout.
pub fn full_mppi_source() -> String {
    let mut src = String::with_capacity(MPPI_HELPERS_WGSL.len() + MPPI_ROLLOUT_WGSL.len() + 1);
    src.push_str(MPPI_HELPERS_WGSL);
    src.push('\n');
    src.push_str(MPPI_ROLLOUT_WGSL);
    src
}

/// Error returned by the validation helpers. The naga error types carry a
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

fn validate_source(src: &str) -> Result<naga::Module, ValidationError> {
    let module = naga::front::wgsl::parse_str(src)
        .map_err(|e| ValidationError::Parse(e.emit_to_string(src)))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::default(),
    );
    validator
        .validate(&module)
        .map_err(|e| ValidationError::Validate(format!("{e:?}")))?;

    Ok(module)
}

/// Parse and validate the mppi_helpers shader.
///
/// Returns the parsed [`naga::Module`] on success. The module can be
/// introspected (e.g. for tests that verify struct sizes or function
/// presence) without dispatching any GPU work.
pub fn validate_mppi_helpers() -> Result<naga::Module, ValidationError> {
    validate_source(MPPI_HELPERS_WGSL)
}

/// Parse and validate the full MPPI shader (helpers + rollout kernel).
///
/// Returns the parsed [`naga::Module`]. Callers can walk `entry_points`
/// to confirm `rollout_and_cost` is present and `global_variables` to
/// confirm all 11 bindings (0..10) resolve against the concatenated
/// source.
pub fn validate_full_mppi() -> Result<naga::Module, ValidationError> {
    validate_source(&full_mppi_source())
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

    // -----------------------------------------------------------------------
    // Rollout kernel validation tests.
    // -----------------------------------------------------------------------

    #[test]
    fn test_mppi_rollout_wgsl_parses_and_validates() {
        // Concatenated helpers + rollout must parse and validate as a
        // single module. If naga rejects, the GPU will too.
        match validate_full_mppi() {
            Ok(_) => {}
            Err(e) => panic!("full MPPI shader validation failed:\n{e}"),
        }
    }

    #[test]
    fn test_mppi_rollout_entry_point_present() {
        let module = validate_full_mppi().expect("full shader validation failed");

        let entries: Vec<&naga::EntryPoint> = module.entry_points.iter().collect();
        assert_eq!(
            entries.len(),
            1,
            "expected exactly 1 entry point, got {}: {:?}",
            entries.len(),
            entries.iter().map(|e| &e.name).collect::<Vec<_>>()
        );

        let ep = entries[0];
        assert_eq!(ep.name, "rollout_and_cost", "entry point name mismatch");
        assert_eq!(
            ep.stage,
            naga::ShaderStage::Compute,
            "entry point stage must be Compute"
        );
        assert_eq!(
            ep.workgroup_size,
            [1u32, 1u32, 1u32],
            "workgroup_size must be (1,1,1); dispatch is per-sample/per-drone"
        );
    }

    #[test]
    fn test_mppi_rollout_uses_all_bindings() {
        let module = validate_full_mppi().expect("full shader validation failed");

        // Collect (group, binding) pairs from all global variables.
        let mut bindings: Vec<(u32, u32)> = module
            .global_variables
            .iter()
            .filter_map(|(_, gv)| gv.binding.as_ref().map(|rb| (rb.group, rb.binding)))
            .collect();
        bindings.sort();

        // Expect @group(0) @binding(0..=10).
        let expected: Vec<(u32, u32)> = (0u32..=10).map(|b| (0u32, b)).collect();
        for exp in &expected {
            assert!(
                bindings.contains(exp),
                "missing binding {:?}; found: {:?}",
                exp,
                bindings
            );
        }
    }

    /// The `MppiDims` WGSL struct must agree with the Rust-side POD
    /// struct (32 bytes). Checks both sides in one test so a layout
    /// regression in either surfaces here.
    #[test]
    fn test_mppi_dims_struct_size() {
        use crate::gpu::pipeline::MppiDims;
        assert_eq!(
            std::mem::size_of::<MppiDims>(),
            32,
            "Rust MppiDims must be 32 bytes (matches WGSL layout)"
        );

        let module = validate_full_mppi().expect("full shader validation failed");
        let ty = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("MppiDims"))
            .expect("WGSL struct `MppiDims` not found in module")
            .1;
        let span = match &ty.inner {
            naga::TypeInner::Struct { span, .. } => *span,
            other => panic!("type `MppiDims` is not a struct: {other:?}"),
        };
        assert_eq!(
            span, 32,
            "WGSL struct `MppiDims` size {span} != 32 (Rust POD size)"
        );
    }
}
