use bevy::prelude::*;

use crate::config::GameConfig;
use crate::simulation::SimState;

pub struct DronePlugin;

impl Plugin for DronePlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_drones)
            .add_systems(Update, (sync_drone_transforms, update_trails, draw_trails));
    }
}

/// Marker component identifying which drone this entity represents.
#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
pub enum DroneMarker {
    A,
    B,
}

/// Trail buffer for a drone.
#[derive(Component)]
pub struct DroneTrail {
    pub positions: Vec<Vec3>,
    pub max_len: usize,
}

impl DroneTrail {
    fn new() -> Self {
        Self {
            positions: Vec::with_capacity(200),
            max_len: 200,
        }
    }

    fn push(&mut self, pos: Vec3) {
        if self.positions.len() >= self.max_len {
            self.positions.remove(0);
        }
        self.positions.push(pos);
    }
}

/// Heading arrow marker (child of drone).
#[derive(Component)]
pub struct HeadingArrow;

fn spawn_drones(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<GameConfig>,
) {
    let drone_size = 0.12;
    let arm_length = 0.08;
    let arm_thickness = 0.02;

    let cyan = materials.add(StandardMaterial {
        base_color: Color::srgb(0.0, 0.85, 1.0),
        emissive: LinearRgba::new(0.0, 0.3, 0.4, 1.0),
        ..default()
    });
    let orange = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 0.6, 0.0),
        emissive: LinearRgba::new(0.4, 0.2, 0.0, 1.0),
        ..default()
    });

    let arm_mesh = meshes.add(Cuboid::new(arm_length * 2.0, arm_thickness, arm_thickness));
    let arrow_mesh = meshes.add(Cuboid::new(0.15, 0.015, 0.015));
    let arrow_material = materials.add(StandardMaterial {
        base_color: Color::srgb(1.0, 1.0, 0.2),
        emissive: LinearRgba::new(0.5, 0.5, 0.0, 1.0),
        unlit: true,
        ..default()
    });

    let spawn_data: [(
        DroneMarker,
        &nalgebra::Vector3<f64>,
        Handle<StandardMaterial>,
    ); 2] = [
        (DroneMarker::A, &config.spawn_a, cyan.clone()),
        (DroneMarker::B, &config.spawn_b, orange.clone()),
    ];

    for (marker, spawn_pos, mat) in spawn_data {
        // sim (x,y,z) z-up → bevy (x,z,y) y-up
        let bevy_pos = Vec3::new(spawn_pos.x as f32, spawn_pos.z as f32, spawn_pos.y as f32);

        commands
            .spawn((
                marker,
                DroneTrail::new(),
                Mesh3d(meshes.add(Cuboid::new(drone_size, drone_size * 0.3, drone_size))),
                MeshMaterial3d(mat.clone()),
                Transform::from_translation(bevy_pos),
            ))
            .with_children(|parent| {
                // Cross arms
                parent.spawn((
                    Mesh3d(arm_mesh.clone()),
                    MeshMaterial3d(mat.clone()),
                    Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::FRAC_PI_4)),
                ));
                parent.spawn((
                    Mesh3d(arm_mesh.clone()),
                    MeshMaterial3d(mat.clone()),
                    Transform::from_rotation(Quat::from_rotation_y(-std::f32::consts::FRAC_PI_4)),
                ));
                // Heading arrow (forward = +X in sim → +X in bevy for the local frame)
                parent.spawn((
                    HeadingArrow,
                    Mesh3d(arrow_mesh.clone()),
                    MeshMaterial3d(arrow_material.clone()),
                    Transform::from_xyz(0.12, 0.0, 0.0),
                ));
            });
    }
}

/// Sync Bevy Transform from sim-core DroneState.
fn sync_drone_transforms(sim: Res<SimState>, mut query: Query<(&DroneMarker, &mut Transform)>) {
    for (marker, mut transform) in &mut query {
        let state = match marker {
            DroneMarker::A => &sim.state_a,
            DroneMarker::B => &sim.state_b,
        };

        // sim (x,y,z) z-up → bevy (x,z,y) y-up
        transform.translation = Vec3::new(
            state.position.x as f32,
            state.position.z as f32,
            state.position.y as f32,
        );

        // Convert quaternion: sim q rotates in z-up frame, bevy in y-up
        // sim attitude: body-to-world in z-up
        let q = state.attitude.quaternion();
        // Swap y and z components to convert from z-up to y-up
        transform.rotation = Quat::from_xyzw(
            q.i as f32, q.k as f32, // sim z → bevy y
            q.j as f32, // sim y → bevy z
            q.w as f32,
        );
    }
}

fn update_trails(sim: Res<SimState>, mut query: Query<(&DroneMarker, &mut DroneTrail)>) {
    for (marker, mut trail) in &mut query {
        let state = match marker {
            DroneMarker::A => &sim.state_a,
            DroneMarker::B => &sim.state_b,
        };
        let pos = Vec3::new(
            state.position.x as f32,
            state.position.z as f32,
            state.position.y as f32,
        );
        trail.push(pos);
    }
}

fn draw_trails(query: Query<(&DroneMarker, &DroneTrail)>, mut gizmos: Gizmos) {
    for (marker, trail) in &query {
        if trail.positions.len() < 2 {
            continue;
        }
        let color = match marker {
            DroneMarker::A => Color::srgba(0.0, 0.85, 1.0, 0.4),
            DroneMarker::B => Color::srgba(1.0, 0.6, 0.0, 0.4),
        };
        for window in trail.positions.windows(2) {
            gizmos.line(window[0], window[1], color);
        }
    }
}
