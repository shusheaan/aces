use bevy::prelude::*;

use crate::config::GameConfig;

/// Spawn the arena geometry: floor, walls, obstacles, lighting.
pub fn spawn_arena(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<GameConfig>,
) {
    let [bx, by, bz] = config.bounds;

    // ── Floor ──
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::new(
            Vec3::Y,
            Vec2::new(bx as f32 / 2.0, by as f32 / 2.0),
        ))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.3, 0.35),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::from_xyz(bx as f32 / 2.0, 0.0, by as f32 / 2.0),
    ));

    // ── Walls (semi-transparent) ──
    let wall_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.5, 0.6, 0.7, 0.15),
        alpha_mode: AlphaMode::Blend,
        double_sided: true,
        cull_mode: None,
        ..default()
    });

    let wall_thickness = 0.02_f32;
    // sim coords (x,y,z) z-up → Bevy (x,z,y) y-up
    // Wall along X at y=0 (sim) → wall along X at z=0 (bevy)
    let walls = [
        // (position, size) — bevy coords
        // South wall (sim y=0)
        (
            Vec3::new(bx as f32 / 2.0, bz as f32 / 2.0, 0.0),
            Vec3::new(bx as f32, bz as f32, wall_thickness),
        ),
        // North wall (sim y=by)
        (
            Vec3::new(bx as f32 / 2.0, bz as f32 / 2.0, by as f32),
            Vec3::new(bx as f32, bz as f32, wall_thickness),
        ),
        // West wall (sim x=0)
        (
            Vec3::new(0.0, bz as f32 / 2.0, by as f32 / 2.0),
            Vec3::new(wall_thickness, bz as f32, by as f32),
        ),
        // East wall (sim x=bx)
        (
            Vec3::new(bx as f32, bz as f32 / 2.0, by as f32 / 2.0),
            Vec3::new(wall_thickness, bz as f32, by as f32),
        ),
        // Ceiling (sim z=bz)
        (
            Vec3::new(bx as f32 / 2.0, bz as f32, by as f32 / 2.0),
            Vec3::new(bx as f32, wall_thickness, by as f32),
        ),
    ];

    for (pos, size) in walls {
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(size.x, size.y, size.z))),
            MeshMaterial3d(wall_material.clone()),
            Transform::from_translation(pos),
        ));
    }

    // ── Obstacles (pillars) ──
    let obstacle_material = materials.add(StandardMaterial {
        base_color: Color::srgb(0.7, 0.2, 0.2),
        perceptual_roughness: 0.8,
        ..default()
    });

    for (center, half_ext) in &config.obstacle_data {
        // sim (cx, cy, cz) → bevy (cx, cz, cy)
        let bevy_pos = Vec3::new(center[0] as f32, center[2] as f32, center[1] as f32);
        let bevy_size = Vec3::new(
            half_ext[0] as f32 * 2.0,
            half_ext[2] as f32 * 2.0,
            half_ext[1] as f32 * 2.0,
        );
        commands.spawn((
            Mesh3d(meshes.add(Cuboid::new(bevy_size.x, bevy_size.y, bevy_size.z))),
            MeshMaterial3d(obstacle_material.clone()),
            Transform::from_translation(bevy_pos),
        ));
    }

    // ── Lighting ──
    commands.spawn((
        DirectionalLight {
            illuminance: 8000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_rotation(Quat::from_euler(EulerRot::XYZ, -0.8, 0.4, 0.0)),
    ));

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 300.0,
    });
}
