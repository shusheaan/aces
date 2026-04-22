use bevy::prelude::*;
use serde::Deserialize;
use std::path::PathBuf;

use aces_mppi::cost::CostWeights;
use aces_sim_core::dynamics::DroneParams;
use aces_sim_core::environment::{Arena, Obstacle};
use aces_sim_core::lockon::LockOnParams;
use nalgebra::Vector3;

// ── TOML deserialization structs ──

#[derive(Deserialize)]
struct ArenaToml {
    bounds: BoundsToml,
    drone: Option<DroneRadiusToml>,
    spawn: SpawnToml,
    obstacles: Vec<ObstacleToml>,
}

#[derive(Deserialize)]
struct BoundsToml {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Deserialize)]
struct DroneRadiusToml {
    collision_radius: f64,
}

#[derive(Deserialize)]
struct SpawnToml {
    drone_a: [f64; 3],
    drone_b: [f64; 3],
}

#[derive(Deserialize)]
struct ObstacleToml {
    #[allow(dead_code)]
    r#type: String,
    center: [f64; 3],
    half_extents: [f64; 3],
}

#[derive(Deserialize)]
struct DroneToml {
    physical: PhysicalToml,
    inertia: InertiaToml,
    simulation: SimToml,
}

#[derive(Deserialize)]
struct PhysicalToml {
    mass: f64,
    arm_length: f64,
    max_motor_thrust: f64,
    torque_coefficient: f64,
    drag_coefficient: f64,
    gravity: f64,
}

#[derive(Deserialize)]
struct InertiaToml {
    ixx: f64,
    iyy: f64,
    izz: f64,
}

#[derive(Deserialize)]
struct SimToml {
    #[allow(dead_code)]
    dt_sim: f64,
    dt_ctrl: f64,
    substeps: usize,
}

#[derive(Deserialize)]
struct RulesToml {
    lockon: LockOnToml,
    mppi: MppiToml,
    noise: Option<NoiseToml>,
}

#[derive(Deserialize)]
struct LockOnToml {
    fov_degrees: f64,
    lock_distance: f64,
    lock_duration: f64,
}

#[derive(Deserialize)]
struct MppiToml {
    #[allow(dead_code)]
    num_samples: usize,
    #[allow(dead_code)]
    horizon: usize,
    temperature: f64,
    noise_std: f64,
    weights: MppiWeightsToml,
}

#[derive(Deserialize)]
struct MppiWeightsToml {
    w_dist: f64,
    w_face: f64,
    w_ctrl: f64,
    w_obs: f64,
    d_safe: f64,
}

#[derive(Deserialize)]
struct NoiseToml {
    wind_theta: Option<f64>,
    wind_mu: Option<[f64; 3]>,
    wind_sigma: Option<f64>,
}

// ── Runtime config resource ──

/// MPPI limits for real-time interactive use (vs. offline training).
/// Training uses 1024 samples × 50 horizon; game must be ~100× lighter.
const GAME_MPPI_SAMPLES: usize = 128;
const GAME_MPPI_HORIZON: usize = 15;

#[derive(Resource)]
pub struct GameConfig {
    pub arena: Arena,
    pub drone_params: DroneParams,
    pub lock_on_params: LockOnParams,
    pub spawn_a: Vector3<f64>,
    pub spawn_b: Vector3<f64>,
    pub dt_ctrl: f64,
    pub substeps: usize,
    pub mppi_num_samples: usize,
    pub mppi_horizon: usize,
    pub mppi_noise_std: f64,
    pub mppi_temperature: f64,
    pub cost_weights: CostWeights,
    pub wind_theta: f64,
    pub wind_mu: Vector3<f64>,
    pub wind_sigma: f64,
    pub bounds: [f64; 3],
    pub obstacle_data: Vec<([f64; 3], [f64; 3])>,
}

impl GameConfig {
    pub fn load() -> Self {
        let config_dir = Self::find_config_dir();

        let arena_str = std::fs::read_to_string(config_dir.join("arena.toml"))
            .expect("Failed to read configs/arena.toml");
        let drone_str = std::fs::read_to_string(config_dir.join("drone.toml"))
            .expect("Failed to read configs/drone.toml");
        let rules_str = std::fs::read_to_string(config_dir.join("rules.toml"))
            .expect("Failed to read configs/rules.toml");

        let arena_toml: ArenaToml = toml::from_str(&arena_str).expect("Invalid arena.toml");
        let drone_toml: DroneToml = toml::from_str(&drone_str).expect("Invalid drone.toml");
        let rules_toml: RulesToml = toml::from_str(&rules_str).expect("Invalid rules.toml");

        // Build Arena
        let bounds_vec = Vector3::new(
            arena_toml.bounds.x,
            arena_toml.bounds.y,
            arena_toml.bounds.z,
        );
        let mut arena = Arena::new(bounds_vec);
        if let Some(d) = &arena_toml.drone {
            arena.drone_radius = d.collision_radius;
        }

        let mut obstacle_data = Vec::new();
        for obs in &arena_toml.obstacles {
            arena.obstacles.push(Obstacle::Box {
                center: Vector3::new(obs.center[0], obs.center[1], obs.center[2]),
                half_extents: Vector3::new(
                    obs.half_extents[0],
                    obs.half_extents[1],
                    obs.half_extents[2],
                ),
            });
            obstacle_data.push((obs.center, obs.half_extents));
        }

        // Build DroneParams
        let phys = &drone_toml.physical;
        let inertia = &drone_toml.inertia;
        let drone_params = DroneParams {
            mass: phys.mass,
            arm_length: phys.arm_length,
            inertia: Vector3::new(inertia.ixx, inertia.iyy, inertia.izz),
            max_thrust: phys.max_motor_thrust,
            torque_coeff: phys.torque_coefficient,
            drag_coeff: phys.drag_coefficient,
            gravity: phys.gravity,
        };

        // Build LockOnParams
        let lock_on_params = LockOnParams {
            fov: rules_toml.lockon.fov_degrees.to_radians(),
            lock_distance: rules_toml.lockon.lock_distance,
            lock_duration: rules_toml.lockon.lock_duration,
        };

        // Build CostWeights
        let w = &rules_toml.mppi.weights;
        let cost_weights = CostWeights {
            w_dist: w.w_dist,
            w_face: w.w_face,
            w_ctrl: w.w_ctrl,
            w_obs: w.w_obs,
            d_safe: w.d_safe,
        };

        // Wind
        let noise = rules_toml.noise.as_ref();
        let wind_theta = noise.and_then(|n| n.wind_theta).unwrap_or(2.0);
        let wind_mu_arr = noise.and_then(|n| n.wind_mu).unwrap_or([0.0, 0.0, 0.0]);
        let wind_sigma = noise.and_then(|n| n.wind_sigma).unwrap_or(0.0);

        Self {
            arena,
            drone_params,
            lock_on_params,
            spawn_a: Vector3::new(
                arena_toml.spawn.drone_a[0],
                arena_toml.spawn.drone_a[1],
                arena_toml.spawn.drone_a[2],
            ),
            spawn_b: Vector3::new(
                arena_toml.spawn.drone_b[0],
                arena_toml.spawn.drone_b[1],
                arena_toml.spawn.drone_b[2],
            ),
            dt_ctrl: drone_toml.simulation.dt_ctrl,
            substeps: drone_toml.simulation.substeps,
            mppi_num_samples: GAME_MPPI_SAMPLES,
            mppi_horizon: GAME_MPPI_HORIZON,
            mppi_noise_std: rules_toml.mppi.noise_std,
            mppi_temperature: rules_toml.mppi.temperature,
            cost_weights,
            wind_theta,
            wind_mu: Vector3::new(wind_mu_arr[0], wind_mu_arr[1], wind_mu_arr[2]),
            wind_sigma,
            bounds: [
                arena_toml.bounds.x,
                arena_toml.bounds.y,
                arena_toml.bounds.z,
            ],
            obstacle_data,
        }
    }

    fn find_config_dir() -> PathBuf {
        // Try current dir, then parent, then workspace root
        let candidates = [
            PathBuf::from("configs"),
            PathBuf::from("../../configs"), // from crates/game/
        ];
        for c in &candidates {
            if c.join("arena.toml").exists() {
                return c.clone();
            }
        }
        // Fallback: use CARGO_MANIFEST_DIR
        if let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") {
            let p = PathBuf::from(manifest).join("../../configs");
            if p.join("arena.toml").exists() {
                return p;
            }
        }
        panic!(
            "Cannot find configs/ directory. Run from workspace root or set CARGO_MANIFEST_DIR."
        );
    }
}
