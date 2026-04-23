use nalgebra::Vector3;
use rand::distributions::Distribution;
use rand::Rng;
use rand_distr::Normal;

/// A single particle representing a hypothesis about opponent state.
#[derive(Debug, Clone)]
pub struct Particle {
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub weight: f64,
}

/// Bounds for constraining particles within the arena.
#[derive(Debug, Clone)]
pub struct ParticleBounds {
    /// Arena min corner (usually [0,0,0])
    pub min: Vector3<f64>,
    /// Arena max corner (e.g. [10,10,3])
    pub max: Vector3<f64>,
}

/// Particle filter for tracking opponent under occlusion.
#[derive(Debug, Clone)]
pub struct ParticleFilter {
    pub particles: Vec<Particle>,
    /// Process noise standard deviation for acceleration
    pub process_noise_accel: f64,
    /// Measurement noise standard deviation
    pub measurement_noise: f64,
    /// Arena bounds for particle clamping
    pub bounds: Option<ParticleBounds>,
}

impl ParticleFilter {
    pub fn new(
        initial_position: Vector3<f64>,
        num_particles: usize,
        process_noise_accel: f64,
        measurement_noise: f64,
        rng: &mut impl Rng,
    ) -> Self {
        let normal = Normal::new(0.0, 0.1).unwrap();

        let mut particles = Vec::with_capacity(num_particles);
        for _ in 0..num_particles {
            particles.push(Particle {
                position: initial_position
                    + Vector3::new(
                        normal.sample(&mut *rng),
                        normal.sample(&mut *rng),
                        normal.sample(&mut *rng),
                    ),
                velocity: Vector3::zeros(),
                weight: 1.0 / num_particles as f64,
            });
        }

        Self {
            particles,
            process_noise_accel,
            measurement_noise,
            bounds: None,
        }
    }

    /// Set arena bounds for constraining particles.
    pub fn set_bounds(&mut self, min: Vector3<f64>, max: Vector3<f64>) {
        self.bounds = Some(ParticleBounds { min, max });
    }

    /// Prediction step: propagate particles with constant-velocity + noise.
    /// Optionally rejects particles that land inside obstacles using a SDF function.
    pub fn predict_with_sdf<F>(&mut self, dt: f64, sdf: F, rng: &mut impl Rng)
    where
        F: Fn(&Vector3<f64>) -> f64,
    {
        let normal = Normal::new(0.0, self.process_noise_accel * dt.sqrt()).unwrap();

        for p in &mut self.particles {
            // Save previous state for rejection
            let prev_pos = p.position;
            let prev_vel = p.velocity;

            // Propose new state
            p.velocity += Vector3::new(
                normal.sample(&mut *rng),
                normal.sample(&mut *rng),
                normal.sample(&mut *rng),
            );
            p.position += p.velocity * dt;

            // Clamp to bounds
            if let Some(ref bounds) = self.bounds {
                for i in 0..3 {
                    p.position[i] = p.position[i].clamp(bounds.min[i] + 0.05, bounds.max[i] - 0.05);
                }
            }

            // Reject if inside obstacle (SDF < 0)
            if sdf(&p.position) < 0.05 {
                // Revert to previous position with dampened velocity
                p.position = prev_pos;
                p.velocity = prev_vel * 0.5;
            }
        }
    }

    /// Prediction step: propagate particles (no obstacle checking).
    pub fn predict(&mut self, dt: f64, rng: &mut impl Rng) {
        let normal = Normal::new(0.0, self.process_noise_accel * dt.sqrt()).unwrap();

        for p in &mut self.particles {
            p.velocity += Vector3::new(
                normal.sample(&mut *rng),
                normal.sample(&mut *rng),
                normal.sample(&mut *rng),
            );
            p.position += p.velocity * dt;

            // Clamp to bounds
            if let Some(ref bounds) = self.bounds {
                for i in 0..3 {
                    p.position[i] = p.position[i].clamp(bounds.min[i] + 0.05, bounds.max[i] - 0.05);
                }
            }
        }
    }

    /// Update step: reweight particles based on position measurement.
    pub fn update(&mut self, measured_position: &Vector3<f64>, rng: &mut impl Rng) {
        let inv_2sigma2 = 1.0 / (2.0 * self.measurement_noise * self.measurement_noise);

        for p in &mut self.particles {
            let diff = p.position - measured_position;
            let dist_sq = diff.norm_squared();
            p.weight = (-dist_sq * inv_2sigma2).exp();
        }

        // Normalize weights
        let total: f64 = self.particles.iter().map(|p| p.weight).sum();
        if total > 1e-12 {
            for p in &mut self.particles {
                p.weight /= total;
            }
        }

        self.resample(rng);
    }

    /// Systematic resampling.
    fn resample(&mut self, rng: &mut impl Rng) {
        let n = self.particles.len();
        let step = 1.0 / n as f64;
        let mut u: f64 = rng.gen::<f64>() * step;

        let mut cumulative = vec![0.0; n];
        cumulative[0] = self.particles[0].weight;
        for i in 1..n {
            cumulative[i] = cumulative[i - 1] + self.particles[i].weight;
        }

        let mut new_particles = Vec::with_capacity(n);
        let mut idx = 0;
        for _ in 0..n {
            while idx < n - 1 && u > cumulative[idx] {
                idx += 1;
            }
            let mut p = self.particles[idx].clone();
            p.weight = step;
            new_particles.push(p);
            u += step;
        }

        self.particles = new_particles;
    }

    /// Estimate mean position from particles.
    pub fn mean_position(&self) -> Vector3<f64> {
        let mut mean = Vector3::zeros();
        for p in &self.particles {
            mean += p.position * p.weight;
        }
        mean
    }

    /// Estimate position variance from particles.
    pub fn position_variance(&self) -> f64 {
        let mean = self.mean_position();
        self.particles
            .iter()
            .map(|p| (p.position - mean).norm_squared() * p.weight)
            .sum()
    }

    /// Get all particle positions (for visualization).
    pub fn particle_positions(&self) -> Vec<[f64; 3]> {
        self.particles
            .iter()
            .map(|p| [p.position.x, p.position.y, p.position.z])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_particles_stay_in_bounds() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut pf = ParticleFilter::new(Vector3::new(5.0, 5.0, 1.5), 100, 5.0, 0.1, &mut rng);
        pf.set_bounds(Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 10.0, 3.0));

        // Run many predict steps with high noise
        for _ in 0..200 {
            pf.predict(0.01, &mut rng);
        }

        // All particles should be within bounds
        for p in &pf.particles {
            assert!(
                p.position.x >= 0.0 && p.position.x <= 10.0,
                "particle x={} out of bounds",
                p.position.x
            );
            assert!(
                p.position.y >= 0.0 && p.position.y <= 10.0,
                "particle y={} out of bounds",
                p.position.y
            );
            assert!(
                p.position.z >= 0.0 && p.position.z <= 3.0,
                "particle z={} out of bounds",
                p.position.z
            );
        }
    }

    #[test]
    fn test_particles_avoid_obstacles() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut pf = ParticleFilter::new(Vector3::new(4.0, 5.0, 1.5), 100, 2.0, 0.1, &mut rng);
        pf.set_bounds(Vector3::new(0.0, 0.0, 0.0), Vector3::new(10.0, 10.0, 3.0));

        // SDF: box obstacle at (5,5,1.5) with half-extents (0.5, 0.5, 1.5)
        let sdf = |p: &Vector3<f64>| {
            let center = Vector3::new(5.0, 5.0, 1.5);
            let half = Vector3::new(0.5, 0.5, 1.5);
            let d = (p - center).abs() - half;
            let outside = Vector3::new(d.x.max(0.0), d.y.max(0.0), d.z.max(0.0)).norm();
            let inside = d.x.max(d.y.max(d.z)).min(0.0);
            outside + inside
        };

        for _ in 0..200 {
            pf.predict_with_sdf(0.01, sdf, &mut rng);
        }

        // No particles should be deep inside the obstacle
        for p in &pf.particles {
            let d = sdf(&p.position);
            assert!(
                d >= -0.1,
                "particle at ({:.2},{:.2},{:.2}) inside obstacle, sdf={:.3}",
                p.position.x,
                p.position.y,
                p.position.z,
                d
            );
        }
    }

    #[test]
    fn test_reconvergence_after_occlusion() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let target = Vector3::new(5.0, 5.0, 1.5);
        let mut pf = ParticleFilter::new(target, 200, 2.0, 0.1, &mut rng);

        // Phase 1: Let particles diffuse (simulating occlusion)
        for _ in 0..100 {
            pf.predict(0.01, &mut rng);
        }
        let var_after_diffusion = pf.position_variance();
        assert!(
            var_after_diffusion > 0.01,
            "particles should have diffused, var={}",
            var_after_diffusion
        );

        // Phase 2: Provide measurements (simulating re-acquisition)
        for _ in 0..50 {
            pf.predict(0.01, &mut rng);
            pf.update(&target, &mut rng);
        }
        let var_after_reconvergence = pf.position_variance();

        assert!(
            var_after_reconvergence < var_after_diffusion * 0.5,
            "particles should reconverge, var_before={:.4}, var_after={:.4}",
            var_after_diffusion,
            var_after_reconvergence
        );
    }
}
