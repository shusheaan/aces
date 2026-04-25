use pyo3::prelude::*;

use aces_estimator::ekf::EKF;
use aces_estimator::particle_filter::ParticleFilter;
use aces_mppi::cost::CostWeights;
use aces_mppi::optimizer::{ChanceConstraintConfig, MppiOptimizer, RiskConfig};
use aces_sim_core::actuator::ActuatorModel;
use aces_sim_core::camera::{render_depth, CameraFrame, CameraParams};
use aces_sim_core::collision::{check_line_of_sight, Visibility};
use aces_sim_core::detection::{detect_opponent, Detection};
use aces_sim_core::dynamics::{step_rk4, DroneParams};
use aces_sim_core::environment::{Arena, Obstacle};
use aces_sim_core::imu_bias::ImuBias;
use aces_sim_core::lockon::{LockOnParams, LockOnTracker};
use aces_sim_core::noise::ObservationNoise;
use aces_sim_core::safety::SafetyEnvelope;
use aces_sim_core::state::DroneState;
use aces_sim_core::wind::WindModel;
use nalgebra::{Vector3, Vector4};
use rand::SeedableRng;

// ---------------------------------------------------------------------------
// StepResult — returned from Simulation.step()
// ---------------------------------------------------------------------------

#[pyclass(get_all)]
#[derive(Clone)]
struct StepResult {
    drone_a_state: [f64; 13],
    drone_b_state: [f64; 13],
    drone_a_forward: [f64; 3],
    drone_b_forward: [f64; 3],
    drone_a_euler: [f64; 3],
    drone_b_euler: [f64; 3],
    drone_a_collision: bool,
    drone_a_oob: bool,
    drone_b_collision: bool,
    drone_b_oob: bool,
    lock_a_progress: f64,
    lock_b_progress: f64,
    kill_a: bool,
    kill_b: bool,
    distance: f64,
    nearest_obs_dist_a: f64,
    nearest_obs_dist_b: f64,
    /// Noisy observation of drone B's position as seen by drone A
    noisy_b_pos_from_a: [f64; 3],
    /// Noisy observation of drone A's position as seen by drone B
    noisy_a_pos_from_b: [f64; 3],
    /// Current wind force on drone A (world frame, N)
    wind_force_a: [f64; 3],
    /// Current wind force on drone B (world frame, N)
    wind_force_b: [f64; 3],
    /// EKF-estimated position of drone B as tracked by drone A
    ekf_b_pos_from_a: [f64; 3],
    /// EKF-estimated velocity of drone B as tracked by drone A
    ekf_b_vel_from_a: [f64; 3],
    /// EKF-estimated position of drone A as tracked by drone B
    ekf_a_pos_from_b: [f64; 3],
    /// EKF-estimated velocity of drone A as tracked by drone B
    ekf_a_vel_from_b: [f64; 3],
    /// Whether drone A can see drone B (line-of-sight not blocked)
    a_sees_b: bool,
    /// Whether drone B can see drone A
    b_sees_a: bool,
    /// Time since drone A last saw drone B (seconds, 0 if currently visible)
    time_since_a_saw_b: f64,
    /// Time since drone B last saw drone A (seconds, 0 if currently visible)
    time_since_b_saw_a: f64,
    /// Belief state: estimated opponent position (particle filter mean when occluded, EKF when visible)
    belief_b_pos_from_a: [f64; 3],
    /// Belief state uncertainty (particle filter variance, 0 when visible)
    belief_b_var_from_a: f64,
    /// Belief state: estimated opponent position for drone B
    belief_a_pos_from_b: [f64; 3],
    /// Belief state uncertainty for drone B
    belief_a_var_from_b: f64,
    // --- Level 4: Camera/FPV ---
    /// Depth image for drone A's camera (flattened 320x240, row-major). None between renders.
    depth_image_a: Option<Vec<f32>>,
    /// Depth image for drone B's camera. None between renders.
    depth_image_b: Option<Vec<f32>>,
    /// Whether drone A's camera rendered a new frame this step.
    camera_rendered_a: bool,
    /// Whether drone B's camera rendered a new frame this step.
    camera_rendered_b: bool,
    // Detection results for drone A (detecting opponent B)
    det_a_detected: bool,
    det_a_bbox: [f32; 4],
    det_a_confidence: f32,
    det_a_depth: f32,
    det_a_pixel_center: [f32; 2],
    // Detection results for drone B (detecting opponent A)
    det_b_detected: bool,
    det_b_bbox: [f32; 4],
    det_b_confidence: f32,
    det_b_depth: f32,
    det_b_pixel_center: [f32; 2],
    // Safety: 0=Ok, 1=Warning, 2=Critical
    safety_a: u8,
    safety_b: u8,
    /// EKF covariance diagonal for A tracking B [P_px, P_py, P_pz, P_vx, P_vy, P_vz]
    ekf_a_cov_diag: [f64; 6],
    /// EKF last innovation for A tracking B [ix, iy, iz]
    ekf_a_innovation: [f64; 3],
    /// IMU accel bias on drone A
    imu_accel_bias_a: [f64; 3],
    /// IMU gyro bias on drone A
    imu_gyro_bias_a: [f64; 3],
}

// ---------------------------------------------------------------------------
// Simulation — holds two drones, arena, lock-on trackers
// ---------------------------------------------------------------------------

#[pyclass]
struct Simulation {
    arena: Arena,
    params: DroneParams,
    drone_a: DroneState,
    drone_b: DroneState,
    lock_a: LockOnTracker,
    lock_b: LockOnTracker,
    dt_ctrl: f64,
    substeps: usize,
    wind_a: WindModel,
    wind_b: WindModel,
    obs_noise: ObservationNoise,
    rng: rand::rngs::StdRng,
    /// EKF: drone A tracking drone B
    ekf_a: EKF,
    /// EKF: drone B tracking drone A
    ekf_b: EKF,
    /// Particle filter: drone A tracking drone B (for occlusion)
    pf_a: ParticleFilter,
    /// Particle filter: drone B tracking drone A (for occlusion)
    pf_b: ParticleFilter,
    /// Time since A last saw B
    time_since_a_saw_b: f64,
    /// Time since B last saw A
    time_since_b_saw_a: f64,
    // --- Level 4: Camera ---
    camera_params: Option<CameraParams>,
    /// Cached last camera frame for drone A (reused between renders).
    last_frame_a: Option<CameraFrame>,
    /// Cached last camera frame for drone B.
    last_frame_b: Option<CameraFrame>,
    /// Cached last detection for drone A.
    last_det_a: Option<Detection>,
    /// Cached last detection for drone B.
    last_det_b: Option<Detection>,
    /// Elapsed sim time since last camera render for A.
    cam_time_since_render_a: f64,
    /// Elapsed sim time since last camera render for B.
    cam_time_since_render_b: f64,
    /// Simulation clock (total elapsed time).
    sim_time: f64,
    /// Opponent radius for SDF and detection.
    cam_drone_radius: f64,
    /// Min confidence distance for detection.
    cam_min_conf_dist: f64,
    /// Safety envelope
    safety: SafetyEnvelope,
    actuator_a: ActuatorModel,
    actuator_b: ActuatorModel,
    imu_bias_a: ImuBias,
    imu_bias_b: ImuBias,
    motor_bias_range: f64,
}

fn build_arena(bounds: [f64; 3], obstacles: Vec<([f64; 3], [f64; 3])>, drone_radius: f64) -> Arena {
    let mut arena = Arena::new(Vector3::new(bounds[0], bounds[1], bounds[2]));
    arena.drone_radius = drone_radius;
    for (center, half_ext) in obstacles {
        arena.obstacles.push(Obstacle::Box {
            center: Vector3::new(center[0], center[1], center[2]),
            half_extents: Vector3::new(half_ext[0], half_ext[1], half_ext[2]),
        });
    }
    arena
}

fn build_params(
    mass: f64,
    arm_length: f64,
    inertia: [f64; 3],
    max_thrust: f64,
    torque_coeff: f64,
    drag_coeff: f64,
) -> DroneParams {
    DroneParams {
        mass,
        arm_length,
        inertia: Vector3::new(inertia[0], inertia[1], inertia[2]),
        max_thrust,
        torque_coeff,
        drag_coeff,
        gravity: 9.81,
    }
}

fn euler_from_quat(state: &DroneState) -> [f64; 3] {
    let (roll, pitch, yaw) = state.attitude.euler_angles();
    [roll, pitch, yaw]
}

fn v3(a: [f64; 3]) -> Vector3<f64> {
    Vector3::new(a[0], a[1], a[2])
}

/// Update both particle filters for one control step.
///
/// Extracted from `Simulation::step` so that `rng`, `pf_a`/`pf_b`, and `arena`
/// can each be borrowed independently without resorting to `mem::replace`.
#[allow(clippy::too_many_arguments)]
fn step_particle_filters(
    pf_a: &mut ParticleFilter,
    pf_b: &mut ParticleFilter,
    arena: &Arena,
    noisy_b: &Vector3<f64>,
    noisy_a: &Vector3<f64>,
    a_sees_b: bool,
    b_sees_a: bool,
    dt_ctrl: f64,
    time_since_a_saw_b: &mut f64,
    time_since_b_saw_a: &mut f64,
    rng: &mut rand::rngs::StdRng,
) {
    pf_a.predict_with_sdf(dt_ctrl, |p| arena.sdf(p), rng);
    if a_sees_b {
        pf_a.update(noisy_b, rng);
        *time_since_a_saw_b = 0.0;
    } else {
        *time_since_a_saw_b += dt_ctrl;
    }

    pf_b.predict_with_sdf(dt_ctrl, |p| arena.sdf(p), rng);
    if b_sees_a {
        pf_b.update(noisy_a, rng);
        *time_since_b_saw_a = 0.0;
    } else {
        *time_since_b_saw_a += dt_ctrl;
    }
}

#[pymethods]
impl Simulation {
    #[new]
    #[pyo3(signature = (
        bounds,
        obstacles,
        mass = 0.027,
        arm_length = 0.04,
        inertia = [1.4e-5, 1.4e-5, 2.17e-5],
        max_thrust = 0.15,
        torque_coeff = 0.005964,
        drag_coeff = 0.01,
        fov = std::f64::consts::FRAC_PI_2,
        lock_distance = 2.0,
        lock_duration = 1.5,
        dt_ctrl = 0.01,
        substeps = 10,
        drone_radius = 0.05,
        wind_theta = 0.0,
        wind_mu = [0.0, 0.0, 0.0],
        wind_sigma = 0.0,
        obs_noise_std = 0.0,
        camera_enabled = false,
        camera_width = 320,
        camera_height = 240,
        camera_fov_deg = 90.0,
        camera_max_depth = 15.0,
        camera_render_hz = 30.0,
        camera_min_conf_dist = 5.0,
        motor_time_constant = 0.0,
        motor_noise_std = 0.0,
        motor_bias_range = 0.0,
        imu_accel_bias_std = 0.0,
        imu_gyro_bias_std = 0.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bounds: [f64; 3],
        obstacles: Vec<([f64; 3], [f64; 3])>,
        mass: f64,
        arm_length: f64,
        inertia: [f64; 3],
        max_thrust: f64,
        torque_coeff: f64,
        drag_coeff: f64,
        fov: f64,
        lock_distance: f64,
        lock_duration: f64,
        dt_ctrl: f64,
        substeps: usize,
        drone_radius: f64,
        wind_theta: f64,
        wind_mu: [f64; 3],
        wind_sigma: f64,
        obs_noise_std: f64,
        camera_enabled: bool,
        camera_width: usize,
        camera_height: usize,
        camera_fov_deg: f64,
        camera_max_depth: f64,
        camera_render_hz: f64,
        camera_min_conf_dist: f64,
        motor_time_constant: f64,
        motor_noise_std: f64,
        motor_bias_range: f64,
        imu_accel_bias_std: f64,
        imu_gyro_bias_std: f64,
    ) -> Self {
        let arena = build_arena(bounds, obstacles, drone_radius);
        let params = build_params(
            mass,
            arm_length,
            inertia,
            max_thrust,
            torque_coeff,
            drag_coeff,
        );
        let lock_params = LockOnParams {
            fov,
            lock_distance,
            lock_duration,
        };

        let wind_enabled = wind_sigma > 0.0;
        let mu = Vector3::new(wind_mu[0], wind_mu[1], wind_mu[2]);
        let mut wind_a = WindModel::new(wind_theta, mu, wind_sigma);
        let mut wind_b = WindModel::new(wind_theta, mu, wind_sigma);
        wind_a.enabled = wind_enabled;
        wind_b.enabled = wind_enabled;

        let obs_noise = ObservationNoise::new(obs_noise_std);

        // EKF: each drone tracks opponent. Use obs_noise_std as measurement noise.
        let ekf_noise = if obs_noise_std > 0.0 {
            obs_noise_std
        } else {
            0.1
        };
        let ekf_a = EKF::new(Vector3::zeros(), ekf_noise);
        let ekf_b = EKF::new(Vector3::zeros(), ekf_noise);

        // Particle filters for belief tracking during occlusion
        let mut rng = rand::rngs::StdRng::from_entropy();
        let mut pf_a = ParticleFilter::new(Vector3::zeros(), 200, 2.0, ekf_noise, &mut rng);
        let mut pf_b = ParticleFilter::new(Vector3::zeros(), 200, 2.0, ekf_noise, &mut rng);
        pf_a.set_bounds(Vector3::zeros(), arena.bounds);
        pf_b.set_bounds(Vector3::zeros(), arena.bounds);

        let actuator_a = ActuatorModel::new(motor_time_constant, motor_noise_std);
        let actuator_b = ActuatorModel::new(motor_time_constant, motor_noise_std);
        let imu_bias_a = ImuBias::new(imu_accel_bias_std, imu_gyro_bias_std);
        let imu_bias_b = ImuBias::new(imu_accel_bias_std, imu_gyro_bias_std);

        let camera_params = if camera_enabled {
            Some(CameraParams::new(
                camera_width,
                camera_height,
                camera_fov_deg,
                camera_max_depth,
                camera_render_hz,
            ))
        } else {
            None
        };

        Self {
            arena,
            params,
            drone_a: DroneState::default(),
            drone_b: DroneState::default(),
            lock_a: LockOnTracker::new(lock_params.clone()),
            lock_b: LockOnTracker::new(lock_params),
            dt_ctrl,
            substeps,
            wind_a,
            wind_b,
            obs_noise,
            rng,
            ekf_a,
            ekf_b,
            pf_a,
            pf_b,
            time_since_a_saw_b: 0.0,
            time_since_b_saw_a: 0.0,
            camera_params,
            last_frame_a: None,
            last_frame_b: None,
            last_det_a: None,
            last_det_b: None,
            cam_time_since_render_a: f64::INFINITY, // force first render
            cam_time_since_render_b: f64::INFINITY,
            sim_time: 0.0,
            cam_drone_radius: drone_radius,
            cam_min_conf_dist: camera_min_conf_dist,
            safety: SafetyEnvelope::crazyflie(),
            actuator_a,
            actuator_b,
            imu_bias_a,
            imu_bias_b,
            motor_bias_range,
        }
    }

    /// Reset both drones to hover at given positions. Returns (state_a, state_b).
    fn reset(&mut self, pos_a: [f64; 3], pos_b: [f64; 3]) -> ([f64; 13], [f64; 13]) {
        self.drone_a = DroneState::hover_at(v3(pos_a));
        self.drone_b = DroneState::hover_at(v3(pos_b));
        self.lock_a.reset();
        self.lock_b.reset();
        self.wind_a.reset();
        self.wind_b.reset();
        // Reset EKFs: A tracks B, B tracks A
        self.ekf_a.reset(v3(pos_b));
        self.ekf_b.reset(v3(pos_a));
        // Reset particle filters
        let mut pf_a = ParticleFilter::new(
            v3(pos_b),
            200,
            2.0,
            self.obs_noise.std_dev.max(0.1),
            &mut self.rng,
        );
        let mut pf_b = ParticleFilter::new(
            v3(pos_a),
            200,
            2.0,
            self.obs_noise.std_dev.max(0.1),
            &mut self.rng,
        );
        pf_a.set_bounds(Vector3::zeros(), self.arena.bounds);
        pf_b.set_bounds(Vector3::zeros(), self.arena.bounds);
        self.pf_a = pf_a;
        self.pf_b = pf_b;
        self.time_since_a_saw_b = 0.0;
        self.time_since_b_saw_a = 0.0;
        self.actuator_a.reset();
        self.actuator_b.reset();
        if self.motor_bias_range > 0.0 {
            self.actuator_a
                .randomize_bias(self.motor_bias_range, &mut self.rng);
            self.actuator_b
                .randomize_bias(self.motor_bias_range, &mut self.rng);
        }
        self.imu_bias_a.reset();
        self.imu_bias_b.reset();
        // Reset camera state
        self.last_frame_a = None;
        self.last_frame_b = None;
        self.last_det_a = None;
        self.last_det_b = None;
        self.cam_time_since_render_a = f64::INFINITY;
        self.cam_time_since_render_b = f64::INFINITY;
        self.sim_time = 0.0;
        (self.drone_a.to_array(), self.drone_b.to_array())
    }

    /// Advance one control step. motors_a/b are [f1,f2,f3,f4] in Newtons.
    fn step(&mut self, motors_a: [f64; 4], motors_b: [f64; 4]) -> StepResult {
        let raw_a = Vector4::new(motors_a[0], motors_a[1], motors_a[2], motors_a[3]);
        let raw_b = Vector4::new(motors_b[0], motors_b[1], motors_b[2], motors_b[3]);
        // Safety: sanitize NaN/Inf motors
        let ua = self
            .safety
            .sanitize_motors(raw_a, &self.drone_a, &self.params);
        let ub = self
            .safety
            .sanitize_motors(raw_b, &self.drone_b, &self.params);

        // Sub-step integration with wind
        let dt_sim = self.dt_ctrl / self.substeps as f64;
        for _ in 0..self.substeps {
            let wind_force_a = self.wind_a.step(dt_sim, &mut self.rng);
            let wind_force_b = self.wind_b.step(dt_sim, &mut self.rng);
            let actual_a = self.actuator_a.apply(&ua, dt_sim, &mut self.rng);
            let actual_b = self.actuator_b.apply(&ub, dt_sim, &mut self.rng);
            self.drone_a = step_rk4(
                &self.drone_a,
                &actual_a,
                &self.params,
                dt_sim,
                &wind_force_a,
            );
            self.drone_b = step_rk4(
                &self.drone_b,
                &actual_b,
                &self.params,
                dt_sim,
                &wind_force_b,
            );
        }

        // Lock-on updates
        let kill_a = self
            .lock_a
            .update(&self.drone_a, &self.drone_b, &self.arena, self.dt_ctrl);
        let kill_b = self
            .lock_b
            .update(&self.drone_b, &self.drone_a, &self.arena, self.dt_ctrl);

        let fwd_a = self.drone_a.forward();
        let fwd_b = self.drone_b.forward();

        // --- Visibility checks ---
        let a_sees_b =
            check_line_of_sight(&self.arena, &self.drone_a.position, &self.drone_b.position)
                == Visibility::Visible;
        let b_sees_a =
            check_line_of_sight(&self.arena, &self.drone_b.position, &self.drone_a.position)
                == Visibility::Visible;

        // --- Noisy observations (only generated when visible) ---
        let noisy_b = if a_sees_b {
            self.obs_noise.apply(&self.drone_b.position, &mut self.rng)
        } else {
            Vector3::zeros() // placeholder, won't be used for EKF update
        };
        let noisy_a = if b_sees_a {
            self.obs_noise.apply(&self.drone_a.position, &mut self.rng)
        } else {
            Vector3::zeros()
        };

        // IMU bias random walk step
        let zero3 = Vector3::zeros();
        let (_biased_accel_a, _biased_gyro_a) = self.imu_bias_a.apply(
            &zero3,
            &self.drone_a.angular_velocity,
            self.dt_ctrl,
            &mut self.rng,
        );
        let (_biased_accel_b, _biased_gyro_b) = self.imu_bias_b.apply(
            &zero3,
            &self.drone_b.angular_velocity,
            self.dt_ctrl,
            &mut self.rng,
        );

        // --- EKF: predict always, update only when visible ---
        self.ekf_a.predict(self.dt_ctrl);
        if a_sees_b {
            self.ekf_a.update(&noisy_b);
        }
        let ekf_b_pos = self.ekf_a.position();
        let ekf_b_vel = self.ekf_a.velocity();

        self.ekf_b.predict(self.dt_ctrl);
        if b_sees_a {
            self.ekf_b.update(&noisy_a);
        }
        let ekf_a_pos = self.ekf_b.position();
        let ekf_a_vel = self.ekf_b.velocity();

        // --- Particle filter: predict with SDF constraints, update only when visible ---
        step_particle_filters(
            &mut self.pf_a,
            &mut self.pf_b,
            &self.arena,
            &noisy_b,
            &noisy_a,
            a_sees_b,
            b_sees_a,
            self.dt_ctrl,
            &mut self.time_since_a_saw_b,
            &mut self.time_since_b_saw_a,
            &mut self.rng,
        );

        // --- Belief state: use EKF when visible, particle filter when occluded ---
        let (belief_b_pos, belief_b_var) = if a_sees_b {
            (ekf_b_pos, 0.0)
        } else {
            (self.pf_a.mean_position(), self.pf_a.position_variance())
        };

        let (belief_a_pos, belief_a_var) = if b_sees_a {
            (ekf_a_pos, 0.0)
        } else {
            (self.pf_b.mean_position(), self.pf_b.position_variance())
        };

        let wf_a = self.wind_a.force;
        let wf_b = self.wind_b.force;

        // --- Level 4: Camera rendering + detection ---
        self.sim_time += self.dt_ctrl;
        self.cam_time_since_render_a += self.dt_ctrl;
        self.cam_time_since_render_b += self.dt_ctrl;

        let mut camera_rendered_a = false;
        let mut camera_rendered_b = false;
        let mut depth_image_a: Option<Vec<f32>> = None;
        let mut depth_image_b: Option<Vec<f32>> = None;

        if let Some(ref cam) = self.camera_params {
            let render_interval = 1.0 / cam.render_hz;

            // Drone A camera
            if self.cam_time_since_render_a >= render_interval {
                let rot_a = self.drone_a.attitude.to_rotation_matrix();
                let mut frame = render_depth(
                    cam,
                    &self.arena,
                    &self.drone_a.position,
                    &rot_a,
                    &self.drone_b.position,
                    self.cam_drone_radius,
                    self.sim_time,
                );
                let det = detect_opponent(
                    cam,
                    &self.arena,
                    &self.drone_a.position,
                    &rot_a,
                    &self.drone_b.position,
                    self.cam_drone_radius,
                    self.cam_min_conf_dist,
                );
                // Move depth data into StepResult to avoid a 300KB clone.
                // last_frame_a retains metadata (width/height/timestamp) with
                // an empty depth vec, since depth is only needed in StepResult.
                let depth = std::mem::take(&mut frame.depth);
                depth_image_a = Some(depth);
                self.last_frame_a = Some(frame);
                self.last_det_a = Some(det);
                self.cam_time_since_render_a = 0.0;
                camera_rendered_a = true;
            }

            // Drone B camera
            if self.cam_time_since_render_b >= render_interval {
                let rot_b = self.drone_b.attitude.to_rotation_matrix();
                let mut frame = render_depth(
                    cam,
                    &self.arena,
                    &self.drone_b.position,
                    &rot_b,
                    &self.drone_a.position,
                    self.cam_drone_radius,
                    self.sim_time,
                );
                let det = detect_opponent(
                    cam,
                    &self.arena,
                    &self.drone_b.position,
                    &rot_b,
                    &self.drone_a.position,
                    self.cam_drone_radius,
                    self.cam_min_conf_dist,
                );
                let depth = std::mem::take(&mut frame.depth);
                depth_image_b = Some(depth);
                self.last_frame_b = Some(frame);
                self.last_det_b = Some(det);
                self.cam_time_since_render_b = 0.0;
                camera_rendered_b = true;
            }
        }

        let det_a = self.last_det_a.as_ref();
        let det_b = self.last_det_b.as_ref();

        StepResult {
            drone_a_state: self.drone_a.to_array(),
            drone_b_state: self.drone_b.to_array(),
            drone_a_forward: [fwd_a.x, fwd_a.y, fwd_a.z],
            drone_b_forward: [fwd_b.x, fwd_b.y, fwd_b.z],
            drone_a_euler: euler_from_quat(&self.drone_a),
            drone_b_euler: euler_from_quat(&self.drone_b),
            drone_a_collision: self.arena.is_collision(&self.drone_a.position),
            drone_a_oob: self.arena.is_out_of_bounds(&self.drone_a.position),
            drone_b_collision: self.arena.is_collision(&self.drone_b.position),
            drone_b_oob: self.arena.is_out_of_bounds(&self.drone_b.position),
            lock_a_progress: self.lock_a.progress(),
            lock_b_progress: self.lock_b.progress(),
            kill_a,
            kill_b,
            distance: self.drone_a.distance_to(&self.drone_b),
            // Combined SDF (boundary + obstacle), matching
            // crates/batch-sim/src/observation.rs::build_observation which uses
            // arena.sdf(). Previously used obstacle_sdf only, which returned
            // +infinity for empty arenas and produced divergent observations
            // between CPU env and batch-sim (obs slot [15]).
            nearest_obs_dist_a: self.arena.sdf(&self.drone_a.position),
            nearest_obs_dist_b: self.arena.sdf(&self.drone_b.position),
            noisy_b_pos_from_a: [noisy_b.x, noisy_b.y, noisy_b.z],
            noisy_a_pos_from_b: [noisy_a.x, noisy_a.y, noisy_a.z],
            wind_force_a: [wf_a.x, wf_a.y, wf_a.z],
            wind_force_b: [wf_b.x, wf_b.y, wf_b.z],
            ekf_b_pos_from_a: [ekf_b_pos.x, ekf_b_pos.y, ekf_b_pos.z],
            ekf_b_vel_from_a: [ekf_b_vel.x, ekf_b_vel.y, ekf_b_vel.z],
            ekf_a_pos_from_b: [ekf_a_pos.x, ekf_a_pos.y, ekf_a_pos.z],
            ekf_a_vel_from_b: [ekf_a_vel.x, ekf_a_vel.y, ekf_a_vel.z],
            a_sees_b,
            b_sees_a,
            time_since_a_saw_b: self.time_since_a_saw_b,
            time_since_b_saw_a: self.time_since_b_saw_a,
            belief_b_pos_from_a: [belief_b_pos.x, belief_b_pos.y, belief_b_pos.z],
            belief_b_var_from_a: belief_b_var,
            belief_a_pos_from_b: [belief_a_pos.x, belief_a_pos.y, belief_a_pos.z],
            belief_a_var_from_b: belief_a_var,
            // Camera / detection
            depth_image_a,
            depth_image_b,
            camera_rendered_a,
            camera_rendered_b,
            det_a_detected: det_a.is_some_and(|d| d.detected),
            det_a_bbox: det_a.map_or([0.0; 4], |d| d.bbox),
            det_a_confidence: det_a.map_or(0.0, |d| d.confidence),
            det_a_depth: det_a.map_or(0.0, |d| d.depth),
            det_a_pixel_center: det_a.map_or([0.0; 2], |d| d.pixel_center),
            det_b_detected: det_b.is_some_and(|d| d.detected),
            det_b_bbox: det_b.map_or([0.0; 4], |d| d.bbox),
            det_b_confidence: det_b.map_or(0.0, |d| d.confidence),
            det_b_depth: det_b.map_or(0.0, |d| d.depth),
            det_b_pixel_center: det_b.map_or([0.0; 2], |d| d.pixel_center),
            // Safety
            safety_a: self.safety.check(&self.drone_a, &self.arena).severity as u8,
            safety_b: self.safety.check(&self.drone_b, &self.arena).severity as u8,
            ekf_a_cov_diag: self.ekf_a.covariance_diagonal(),
            ekf_a_innovation: self
                .ekf_a
                .last_innovation()
                .map(|v| [v.x, v.y, v.z])
                .unwrap_or([0.0; 3]),
            imu_accel_bias_a: [
                self.imu_bias_a.accel_bias.x,
                self.imu_bias_a.accel_bias.y,
                self.imu_bias_a.accel_bias.z,
            ],
            imu_gyro_bias_a: [
                self.imu_bias_a.gyro_bias.x,
                self.imu_bias_a.gyro_bias.y,
                self.imu_bias_a.gyro_bias.z,
            ],
        }
    }

    /// Hover thrust per motor (N).
    fn hover_thrust(&self) -> f64 {
        self.params.hover_thrust()
    }

    /// Max thrust per motor (N).
    fn max_thrust(&self) -> f64 {
        self.params.max_thrust
    }

    /// Current state arrays.
    fn drone_a_state(&self) -> [f64; 13] {
        self.drone_a.to_array()
    }

    fn drone_b_state(&self) -> [f64; 13] {
        self.drone_b.to_array()
    }

    /// SDF at a point (for debugging / observation).
    fn arena_sdf(&self, point: [f64; 3]) -> f64 {
        self.arena.sdf(&v3(point))
    }

    /// Get particle filter positions for visualization (A's belief about B).
    fn belief_particles_a(&self) -> Vec<[f64; 3]> {
        self.pf_a.particle_positions()
    }

    /// Get particle filter positions for visualization (B's belief about A).
    fn belief_particles_b(&self) -> Vec<[f64; 3]> {
        self.pf_b.particle_positions()
    }

    /// Get camera intrinsics (fx, fy, cx, cy). Returns None if camera is disabled.
    fn camera_intrinsics(&self) -> Option<(f64, f64, f64, f64)> {
        self.camera_params
            .as_ref()
            .map(|c| (c.fx, c.fy, c.cx, c.cy))
    }

    /// Get camera image dimensions (width, height). Returns None if camera is disabled.
    fn camera_resolution(&self) -> Option<(usize, usize)> {
        self.camera_params.as_ref().map(|c| (c.width, c.height))
    }

    /// Get camera max depth. Returns None if camera is disabled.
    fn camera_max_depth(&self) -> Option<f64> {
        self.camera_params.as_ref().map(|c| c.max_depth)
    }

    /// Check line-of-sight between two arbitrary points.
    /// Returns true if visible, false if occluded by obstacles.
    fn check_los(&self, pos_a: [f64; 3], pos_b: [f64; 3]) -> bool {
        check_line_of_sight(&self.arena, &v3(pos_a), &v3(pos_b)) == Visibility::Visible
    }
}

// ---------------------------------------------------------------------------
// MppiController — wraps the MPPI optimizer for Python
// ---------------------------------------------------------------------------

#[pyclass]
struct MppiController {
    optimizer: MppiOptimizer,
}

#[pymethods]
impl MppiController {
    #[new]
    #[pyo3(signature = (
        bounds,
        obstacles,
        num_samples = 1024,
        horizon = 50,
        noise_std = 0.03,
        temperature = 10.0,
        mass = 0.027,
        arm_length = 0.04,
        inertia = [1.4e-5, 1.4e-5, 2.17e-5],
        max_thrust = 0.15,
        torque_coeff = 0.005964,
        drag_coeff = 0.01,
        dt_ctrl = 0.01,
        substeps = 10,
        drone_radius = 0.05,
        w_dist = 1.0,
        w_face = 5.0,
        w_ctrl = 0.01,
        w_obs = 1000.0,
        d_safe = 0.3,
        risk_wind_theta = 0.0,
        risk_wind_sigma = 0.0,
        risk_cvar_alpha = 0.0,
        risk_cvar_penalty = 0.0,
        cc_delta = None,
        cc_lambda_lr = 0.1,
        cc_lambda_init = 100.0,
        seed = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        bounds: [f64; 3],
        obstacles: Vec<([f64; 3], [f64; 3])>,
        num_samples: usize,
        horizon: usize,
        noise_std: f64,
        temperature: f64,
        mass: f64,
        arm_length: f64,
        inertia: [f64; 3],
        max_thrust: f64,
        torque_coeff: f64,
        drag_coeff: f64,
        dt_ctrl: f64,
        substeps: usize,
        drone_radius: f64,
        w_dist: f64,
        w_face: f64,
        w_ctrl: f64,
        w_obs: f64,
        d_safe: f64,
        risk_wind_theta: f64,
        risk_wind_sigma: f64,
        risk_cvar_alpha: f64,
        risk_cvar_penalty: f64,
        cc_delta: Option<f64>,
        cc_lambda_lr: f64,
        cc_lambda_init: f64,
        seed: Option<u64>,
    ) -> Self {
        let arena = build_arena(bounds, obstacles, drone_radius);
        let params = build_params(
            mass,
            arm_length,
            inertia,
            max_thrust,
            torque_coeff,
            drag_coeff,
        );
        let weights = CostWeights {
            w_dist,
            w_face,
            w_ctrl,
            w_obs,
            d_safe,
        };
        let mut optimizer = if let Some(s) = seed {
            MppiOptimizer::with_seed(
                num_samples,
                horizon,
                noise_std,
                temperature,
                params,
                arena,
                weights,
                dt_ctrl,
                substeps,
                s,
            )
        } else {
            MppiOptimizer::new(
                num_samples,
                horizon,
                noise_std,
                temperature,
                params,
                arena,
                weights,
                dt_ctrl,
                substeps,
            )
        };

        // Enable risk-aware mode if wind_sigma > 0
        if risk_wind_sigma > 0.0 {
            optimizer.set_risk_config(RiskConfig {
                wind: WindModel::new(risk_wind_theta, Vector3::zeros(), risk_wind_sigma),
                cvar_alpha: risk_cvar_alpha,
                cvar_penalty: risk_cvar_penalty,
            });
        }

        // Enable chance constraint if cc_delta is provided
        if let Some(delta) = cc_delta {
            optimizer.set_chance_constraint(ChanceConstraintConfig {
                delta,
                lambda_lr: cc_lambda_lr,
                lambda_init: cc_lambda_init,
                lambda_min: 0.0,
                lambda_max: 1e6,
            });
        }

        Self { optimizer }
    }

    /// Compute optimal motor thrusts [f1,f2,f3,f4] given 13-element state arrays.
    fn compute_action(
        &mut self,
        self_state: [f64; 13],
        enemy_state: [f64; 13],
        pursuit: bool,
    ) -> [f64; 4] {
        let s = DroneState::from_array(&self_state);
        let e = DroneState::from_array(&enemy_state);
        let a = self.optimizer.compute_action(&s, &e, pursuit);
        [a[0], a[1], a[2], a[3]]
    }

    /// Compute optimal motor thrusts with belief-weighted costs (Level 3).
    ///
    /// `belief_var` is the position variance from the particle filter.
    /// When high, the controller plans conservatively.
    fn compute_action_with_belief(
        &mut self,
        self_state: [f64; 13],
        enemy_state: [f64; 13],
        pursuit: bool,
        belief_var: f64,
    ) -> [f64; 4] {
        let s = DroneState::from_array(&self_state);
        let e = DroneState::from_array(&enemy_state);
        let a = self
            .optimizer
            .compute_action_with_belief(&s, &e, pursuit, belief_var);
        [a[0], a[1], a[2], a[3]]
    }

    /// Reset warm-start state.
    fn reset(&mut self) {
        self.optimizer.reset();
    }
}

// ---------------------------------------------------------------------------
// GPU batch MPPI — only compiled with the `gpu` feature.
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu")]
mod gpu_binding {
    use aces_batch_sim::battle::{BatchConfig, StepResult};
    use aces_batch_sim::f32_dynamics::DroneParamsF32;
    use aces_batch_sim::f32_sdf::{ArenaF32, ObstacleF32};
    use aces_batch_sim::gpu::orchestrator::GpuBatchOrchestrator;
    use aces_batch_sim::gpu::pipeline::{CostWeightsGpu, GpuBatchMppi};
    use aces_batch_sim::reward::RewardConfig;
    use aces_sim_core::lockon::LockOnParams;
    use nalgebra::Vector3;
    use numpy::{
        IntoPyArray, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2,
        PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods,
    };
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyList};

    /// Pack a `Vec<StepResult>` into the `(obs, rewards, dones, infos)` tuple
    /// returned by both `PyGpuVecEnv::step` and
    /// `PyGpuVecEnv::step_with_agent_a`. Factored out so the two step methods
    /// don't drift.
    #[allow(clippy::type_complexity)]
    fn pack_step_results<'py>(
        py: Python<'py>,
        step_results: Vec<StepResult>,
        n_envs: usize,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyList>,
    )> {
        debug_assert_eq!(step_results.len(), n_envs);

        let mut obs = Vec::with_capacity(n_envs * 21);
        let mut rewards = Vec::with_capacity(n_envs);
        let mut dones = Vec::with_capacity(n_envs);

        for r in &step_results {
            for j in 0..21 {
                obs.push(r.obs_a[j] as f32);
            }
            rewards.push(r.reward_a as f32);
            dones.push(r.done);
        }

        let py_obs_1d = obs.into_pyarray(py);
        let py_obs = py_obs_1d
            .reshape([n_envs, 21])
            .map_err(|e| PyRuntimeError::new_err(format!("reshape failed: {e}")))?;
        let py_rewards = rewards.into_pyarray(py);
        let py_dones = dones.into_pyarray(py);

        let infos = PyList::empty(py);
        for r in &step_results {
            let d = PyDict::new(py);
            d.set_item("distance", r.info.distance)?;
            d.set_item("lock_a", r.info.lock_progress_a)?;
            d.set_item("lock_b", r.info.lock_progress_b)?;
            d.set_item("visible", r.info.visible_ab)?;
            d.set_item("kill_a", r.info.kill_a)?;
            d.set_item("kill_b", r.info.kill_b)?;
            d.set_item("collision_a", r.info.collision_a)?;
            d.set_item("collision_b", r.info.collision_b)?;
            d.set_item("timeout", r.info.timeout)?;
            // CPU env compat aliases — see docs/architecture.md §11.25
            d.set_item("collision", r.info.collision_a)?;
            d.set_item("truncated", r.info.timeout)?;
            d.set_item("lock_a_progress", r.info.lock_progress_a)?;
            infos.append(d)?;
        }

        Ok((py_obs, py_rewards, py_dones, infos))
    }

    /// GPU-accelerated batched MPPI optimizer.
    ///
    /// Constructs a single `wgpu` device + compiled compute pipeline and runs
    /// `n_drones` independent MPPI rollouts in parallel on the GPU. Useful
    /// when training code wants to batch opponent planning across many
    /// parallel environments without incurring CPU sampling cost.
    ///
    /// Defaults:
    ///   * physics: Crazyflie 2.1
    ///   * arena:   10x10x3 m warehouse with 5 box pillars
    ///   * weights: w_dist=1.0, w_face=5.0, w_ctrl=0.01, w_obs=1000.0, d_safe=0.3
    ///
    /// GPU init will fail with a `RuntimeError` if no adapter is available
    /// (e.g. headless server with no GPU drivers).
    #[pyclass(name = "GpuBatchMppi")]
    pub struct PyGpuBatchMppi {
        inner: GpuBatchMppi,
    }

    #[pymethods]
    impl PyGpuBatchMppi {
        #[new]
        #[pyo3(signature = (n_drones, n_samples, horizon))]
        fn new(n_drones: usize, n_samples: usize, horizon: usize) -> PyResult<Self> {
            let params = DroneParamsF32::crazyflie();
            let arena = default_warehouse_arena();
            let weights = CostWeightsGpu::new(
                1.0,
                5.0,
                0.01,
                1000.0,
                0.3,
                params.hover_thrust(),
                [10.0, 10.0, 3.0],
            );
            let inner = GpuBatchMppi::new(n_drones, n_samples, horizon, &params, weights, &arena)
                .map_err(|e| {
                PyRuntimeError::new_err(format!("GpuBatchMppi init failed: {e}"))
            })?;
            Ok(PyGpuBatchMppi { inner })
        }

        #[getter]
        fn n_drones(&self) -> usize {
            self.inner.n_drones
        }

        #[getter]
        fn n_samples(&self) -> usize {
            self.inner.n_samples
        }

        #[getter]
        fn horizon(&self) -> usize {
            self.inner.horizon
        }

        /// Run one full MPPI iteration on the GPU.
        ///
        /// Arguments:
        ///   states:     float32 array shape (n_drones, 13)
        ///   enemies:    float32 array shape (n_drones, 13)
        ///   mean_ctrls: float32 array shape (n_drones, horizon, 4)
        ///   noise:      float32 array shape (n_drones, n_samples, horizon, 4)
        ///
        /// Returns:
        ///   new_mean_ctrls: float32 array shape (n_drones, horizon, 4)
        ///
        /// # Thread safety
        /// A single `GpuBatchMppi` instance must NOT be called concurrently from
        /// multiple Python threads. The method releases the GIL during GPU dispatch,
        /// but all calls on the same instance share the same GPU buffers — concurrent
        /// calls will corrupt the staging state and produce wrong results silently.
        /// Use one instance per thread, or serialize calls via a Python lock.
        fn compute_batch_actions<'py>(
            &self,
            py: Python<'py>,
            states: PyReadonlyArray2<'py, f32>,
            enemies: PyReadonlyArray2<'py, f32>,
            mean_ctrls: PyReadonlyArray3<'py, f32>,
            noise: PyReadonlyArray4<'py, f32>,
        ) -> PyResult<Bound<'py, PyArray3<f32>>> {
            let n_drones = self.inner.n_drones;
            let n_samples = self.inner.n_samples;
            let horizon = self.inner.horizon;

            // Validate shapes.
            let states_shape = states.shape();
            if states_shape != [n_drones, 13] {
                return Err(PyValueError::new_err(format!(
                    "states shape {:?} != ({}, 13)",
                    states_shape, n_drones
                )));
            }
            if enemies.shape() != [n_drones, 13] {
                return Err(PyValueError::new_err(format!(
                    "enemies shape {:?} != ({}, 13)",
                    enemies.shape(),
                    n_drones
                )));
            }
            if mean_ctrls.shape() != [n_drones, horizon, 4] {
                return Err(PyValueError::new_err(format!(
                    "mean_ctrls shape {:?} != ({}, {}, 4)",
                    mean_ctrls.shape(),
                    n_drones,
                    horizon
                )));
            }
            if noise.shape() != [n_drones, n_samples, horizon, 4] {
                return Err(PyValueError::new_err(format!(
                    "noise shape {:?} != ({}, {}, {}, 4)",
                    noise.shape(),
                    n_drones,
                    n_samples,
                    horizon
                )));
            }

            // Materialize into contiguous f32 vecs; `.iter().copied()` on an
            // ndarray view walks in row-major regardless of memory layout,
            // which matches the GPU shader's indexing convention.
            let states_vec: Vec<f32> = states.as_array().iter().copied().collect();
            let enemies_vec: Vec<f32> = enemies.as_array().iter().copied().collect();
            let mean_ctrls_vec: Vec<f32> = mean_ctrls.as_array().iter().copied().collect();
            let noise_vec: Vec<f32> = noise.as_array().iter().copied().collect();

            // Release the GIL during GPU work so other Python threads can run.
            let result = py.allow_threads(|| {
                self.inner.compute_batch_actions(
                    &states_vec,
                    &enemies_vec,
                    &mean_ctrls_vec,
                    &noise_vec,
                )
            });

            // Build the output as a 1-D numpy array and reshape to
            // (n_drones, horizon, 4). `into_pyarray` consumes `result` with
            // no extra copy.
            let arr1 = result.into_pyarray(py);
            arr1.reshape([n_drones, horizon, 4])
                .map_err(|e| PyRuntimeError::new_err(format!("reshape failed: {e}")))
        }
    }

    // -----------------------------------------------------------------
    // GpuVecEnv — VecEnv-style wrapper around GpuBatchOrchestrator.
    // Runs N parallel battles with GPU MPPI driving both sides.
    // -----------------------------------------------------------------

    /// Vectorized MPPI-vs-MPPI battle environment backed by a single
    /// `GpuBatchOrchestrator`.
    ///
    /// Each `step()` runs one GPU MPPI dispatch for every drone in every
    /// battle and advances physics in parallel. Observations are the
    /// agent-A side only (21-dim per battle), suitable for dropping into
    /// a Gymnasium / Stable-Baselines3 VecEnv pipeline.
    ///
    /// External agent actions are not supported yet — this is the
    /// pure MPPI-vs-MPPI slice.
    #[pyclass(name = "GpuVecEnv")]
    pub struct PyGpuVecEnv {
        inner: GpuBatchOrchestrator,
        n_envs: usize,
    }

    #[pymethods]
    impl PyGpuVecEnv {
        #[new]
        #[pyo3(signature = (
            n_envs,
            mppi_samples = 128,
            mppi_horizon = 15,
            noise_std = 0.03,
            max_steps = 1000,
            dt_ctrl = 0.01,
            substeps = 10,
            wind_sigma = 0.0,
            wind_theta = 2.0,
            seed = 42,
            // Reward config (defaults match `RewardConfig::default()` /
            // configs/rules.toml [reward]). Overridable from Python so
            // GPU training reads the same tuned weights as the CPU env.
            kill_reward = 100.0,
            killed_penalty = -100.0,
            collision_penalty = -50.0,
            opponent_crash_reward = 5.0,
            lock_progress_reward = 5.0,
            approach_reward = 3.0,
            survival_bonus = 0.01,
            control_penalty = 0.01,
            // MPPI cost weights (configs/rules.toml [mppi.weights]).
            // When None, the hardcoded defaults (matching rules.toml) are used.
            w_dist = None,
            w_face = None,
            w_ctrl = None,
            w_obs = None,
            d_safe = None,
            // Lock-on params (configs/rules.toml [lockon]).
            // When None, crate defaults are used.
            fov_degrees = None,
            lock_distance = None,
            lock_duration = None,
        ))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            n_envs: usize,
            mppi_samples: usize,
            mppi_horizon: usize,
            noise_std: f32,
            max_steps: u32,
            dt_ctrl: f64,
            substeps: usize,
            wind_sigma: f64,
            wind_theta: f64,
            seed: u64,
            kill_reward: f64,
            killed_penalty: f64,
            collision_penalty: f64,
            opponent_crash_reward: f64,
            lock_progress_reward: f64,
            approach_reward: f64,
            survival_bonus: f64,
            control_penalty: f64,
            w_dist: Option<f32>,
            w_face: Option<f32>,
            w_ctrl: Option<f32>,
            w_obs: Option<f32>,
            d_safe: Option<f32>,
            fov_degrees: Option<f32>,
            lock_distance: Option<f32>,
            lock_duration: Option<f32>,
        ) -> PyResult<Self> {
            if n_envs == 0 {
                return Err(PyValueError::new_err("n_envs must be > 0"));
            }
            let batch_config = BatchConfig {
                max_steps,
                dt_ctrl,
                substeps,
                wind_sigma,
                wind_theta,
            };
            let reward_config = RewardConfig {
                kill_reward,
                killed_penalty,
                collision_penalty,
                opponent_crash_reward,
                lock_progress_reward,
                approach_reward,
                survival_bonus,
                control_penalty,
            };
            // Build cost weights if any scalar override was provided.
            // We also check `fov_degrees` here: fov_half is stored in
            // CostWeightsGpu (binding 8) for the WGSL evasion cost, so a
            // fov_degrees override must update cost weights as well as lockon params.
            let cost_weights_opt: Option<CostWeightsGpu> = if w_dist.is_some()
                || w_face.is_some()
                || w_ctrl.is_some()
                || w_obs.is_some()
                || d_safe.is_some()
                || fov_degrees.is_some()
            {
                // Use defaults matching rules.toml for any field left as None.
                let hover = DroneParamsF32::crazyflie().hover_thrust();
                // fov_half = fov_degrees / 2 converted to radians;
                // default PI/4 (45°) matches [lockon] fov_degrees = 90.
                let fov_half = fov_degrees
                    .map(|d| (d / 2.0_f32).to_radians())
                    .unwrap_or(std::f32::consts::FRAC_PI_4);
                Some(CostWeightsGpu::with_fov_half(
                    w_dist.unwrap_or(1.0),
                    w_face.unwrap_or(5.0),
                    w_ctrl.unwrap_or(0.01),
                    w_obs.unwrap_or(1000.0),
                    d_safe.unwrap_or(0.3),
                    hover,
                    // arena bounds must match the orchestrator's default warehouse arena.
                    [10.0_f32, 10.0_f32, 3.0_f32],
                    fov_half,
                ))
            } else {
                None
            };
            // Build LockOnParams if any override was provided.
            // `fov_degrees` is the full cone angle; `LockOnParams::fov` stores
            // the full cone angle in radians (same convention as rules.toml).
            let lockon_opt: Option<LockOnParams> =
                if fov_degrees.is_some() || lock_distance.is_some() || lock_duration.is_some() {
                    let def = LockOnParams::default();
                    Some(LockOnParams {
                        fov: fov_degrees
                            .map(|d| (d as f64).to_radians())
                            .unwrap_or(def.fov),
                        lock_distance: lock_distance.map(|v| v as f64).unwrap_or(def.lock_distance),
                        lock_duration: lock_duration.map(|v| v as f64).unwrap_or(def.lock_duration),
                    })
                } else {
                    None
                };
            let inner = GpuBatchOrchestrator::new_with_weights(
                n_envs,
                batch_config,
                reward_config,
                mppi_samples,
                mppi_horizon,
                noise_std,
                seed,
                cost_weights_opt,
                lockon_opt,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("GpuVecEnv init failed: {e}")))?;
            Ok(PyGpuVecEnv { inner, n_envs })
        }

        #[getter]
        fn n_envs(&self) -> usize {
            self.n_envs
        }

        #[getter]
        fn horizon(&self) -> usize {
            self.inner.horizon()
        }

        #[getter]
        fn n_samples(&self) -> usize {
            self.inner.n_samples()
        }

        /// Reset all battles to fresh random spawns. Returns agent-A
        /// observations as a float32 ndarray of shape `(n_envs, 21)`.
        fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
            let obs_rows = py.allow_threads(|| self.inner.reset());
            let mut flat = Vec::with_capacity(self.n_envs * 21);
            for row in &obs_rows {
                flat.extend_from_slice(row);
            }
            let arr1 = flat.into_pyarray(py);
            arr1.reshape([self.n_envs, 21])
                .map_err(|e| PyRuntimeError::new_err(format!("reshape failed: {e}")))
        }

        /// Step every battle once using GPU MPPI for both sides.
        ///
        /// Returns `(obs, rewards, dones, infos)` where:
        ///   * obs:     float32 ndarray shape `(n_envs, 21)`
        ///   * rewards: float32 ndarray shape `(n_envs,)` — agent A reward
        ///   * dones:   bool ndarray shape `(n_envs,)`
        ///   * infos:   Python list of dicts with
        ///              `{distance, lock_a, lock_b, visible, step}`
        ///
        /// Note: the underlying `GpuBatchOrchestrator` resets terminated
        /// battles in place — the returned observation for a done env is
        /// the *pre-reset* observation (Gymnasium's "terminal observation"
        /// convention). Downstream VecEnv wrappers that need the next
        /// reset obs can call `reset()` themselves when `dones[i]`.
        #[allow(clippy::type_complexity)]
        fn step<'py>(
            &mut self,
            py: Python<'py>,
        ) -> PyResult<(
            Bound<'py, PyArray2<f32>>,
            Bound<'py, PyArray1<f32>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyList>,
        )> {
            let step_results = py.allow_threads(|| self.inner.step_all());
            pack_step_results(py, step_results, self.n_envs)
        }

        /// Step every battle once using external actions for agent A
        /// (learning agent) and GPU MPPI for agent B (opponent).
        ///
        /// Arguments:
        ///   * `actions_a`: float32 ndarray shape `(n_envs, 4)`. Each row is a
        ///     per-motor thrust command in `[0, max_thrust]` for the agent-A
        ///     drone in that battle. Values outside the range are clamped.
        ///
        /// Returns `(obs, rewards, dones, infos)` with the same layout as
        /// [`Self::step`]. The full GPU MPPI dispatch still runs for both
        /// drones (so agent B's rollouts condition on agent A's true state),
        /// but the GPU's proposed action for agent A is discarded in favor
        /// of `actions_a`.
        #[allow(clippy::type_complexity)]
        fn step_with_agent_a<'py>(
            &mut self,
            py: Python<'py>,
            actions_a: PyReadonlyArray2<'py, f32>,
        ) -> PyResult<(
            Bound<'py, PyArray2<f32>>,
            Bound<'py, PyArray1<f32>>,
            Bound<'py, PyArray1<bool>>,
            Bound<'py, PyList>,
        )> {
            let shape = actions_a.shape();
            if shape != [self.n_envs, 4] {
                return Err(PyValueError::new_err(format!(
                    "actions_a shape {:?} != ({}, 4)",
                    shape, self.n_envs
                )));
            }

            // Materialize into a Vec<[f32; 4]> ahead of `allow_threads` so
            // that the numpy array borrow doesn't escape the GIL scope.
            let arr = actions_a.as_array();
            let actions_vec: Vec<[f32; 4]> = (0..self.n_envs)
                .map(|i| [arr[(i, 0)], arr[(i, 1)], arr[(i, 2)], arr[(i, 3)]])
                .collect();

            let step_results =
                py.allow_threads(|| self.inner.step_with_agent_a_actions(&actions_vec));
            pack_step_results(py, step_results, self.n_envs)
        }
    }

    /// 10x10x3 warehouse with 5 box pillars — matches the default arena
    /// used by the Python dogfight env so out-of-the-box numerics line up.
    fn default_warehouse_arena() -> ArenaF32 {
        let mut arena = ArenaF32::new(Vector3::new(10.0, 10.0, 3.0));
        for (x, y) in [
            (2.0f32, 2.0f32),
            (2.0, 8.0),
            (5.0, 5.0),
            (8.0, 2.0),
            (8.0, 8.0),
        ] {
            arena.obstacles.push(ObstacleF32::Box {
                center: Vector3::new(x, y, 1.5),
                half_extents: Vector3::new(0.5, 0.5, 1.5),
            });
        }
        arena
    }
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", "0.1.0")?;
    m.add_class::<StepResult>()?;
    m.add_class::<Simulation>()?;
    m.add_class::<MppiController>()?;

    #[cfg(feature = "gpu")]
    m.add_class::<gpu_binding::PyGpuBatchMppi>()?;

    #[cfg(feature = "gpu")]
    m.add_class::<gpu_binding::PyGpuVecEnv>()?;

    Ok(())
}
