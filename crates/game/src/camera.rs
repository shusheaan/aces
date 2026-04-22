use bevy::input::mouse::{MouseMotion, MouseWheel};
use bevy::prelude::*;
use bevy::render::camera::Viewport;

use crate::config::GameConfig;
use crate::simulation::SimState;
use crate::ActiveDrone;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<CameraMode>()
            .init_resource::<OrbitState>()
            .init_resource::<MainCameraEntity>()
            .add_systems(Startup, spawn_cameras)
            .add_systems(
                Update,
                (
                    orbit_input,
                    update_main_camera.after(orbit_input),
                    update_fpv_cameras,
                    resize_viewports,
                ),
            );
    }
}

/// Resource storing the main camera entity so HUD can target it.
#[derive(Resource, Default)]
pub struct MainCameraEntity(pub Option<Entity>);

#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum CameraMode {
    #[default]
    GodView,
    Follow,
}

/// Tag for the main 3D camera (top portion).
#[derive(Component)]
pub struct MainCamera;

/// Tag for FPV cameras (bottom split).
#[derive(Component)]
pub struct FpvCamera {
    pub drone: FpvDrone,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FpvDrone {
    A,
    B,
}

/// Orbit camera state for god-view mode.
#[derive(Resource)]
pub struct OrbitState {
    pub yaw: f32,
    pub pitch: f32,
    pub distance: f32,
    pub center: Vec3,
}

impl Default for OrbitState {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 1.55, // ~89°, nearly top-down (matches original god-view)
            distance: 14.0,
            center: Vec3::new(5.0, 0.0, 5.0), // arena center
        }
    }
}

/// Layout: top 65% = main camera, bottom 35% = two FPV side by side.
const MAIN_HEIGHT_RATIO: f32 = 0.65;

fn spawn_cameras(
    mut commands: Commands,
    windows: Query<&Window>,
    config: Res<GameConfig>,
    mut main_cam_entity: ResMut<MainCameraEntity>,
) {
    let window = windows.single();
    let w = window.physical_width();
    let h = window.physical_height();

    let main_h = (h as f32 * MAIN_HEIGHT_RATIO) as u32;
    let fpv_h = h - main_h;
    let fpv_w = w / 2;

    // Main camera — top portion
    let main_entity = commands
        .spawn((
            MainCamera,
            Camera3d::default(),
            Camera {
                viewport: Some(Viewport {
                    physical_position: UVec2::new(0, 0),
                    physical_size: UVec2::new(w, main_h),
                    ..default()
                }),
                order: 0,
                ..default()
            },
            Transform::from_xyz(5.0, 12.0, 5.0).looking_at(Vec3::new(5.0, 0.0, 5.0), Vec3::Y),
        ))
        .id();

    // Store main camera entity in resource for HUD targeting
    main_cam_entity.0 = Some(main_entity);

    // FPV camera A — bottom-left, initial position from spawn config
    let spawn_a = config.spawn_a;
    let fpv_a_pos = Vec3::new(spawn_a.x as f32, spawn_a.z as f32, spawn_a.y as f32);
    commands.spawn((
        FpvCamera { drone: FpvDrone::A },
        Camera3d::default(),
        Camera {
            viewport: Some(Viewport {
                physical_position: UVec2::new(0, main_h),
                physical_size: UVec2::new(fpv_w, fpv_h),
                ..default()
            }),
            order: 1,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.05, 0.05, 0.1)),
            ..default()
        },
        Transform::from_translation(fpv_a_pos),
    ));

    // FPV camera B — bottom-right, initial position from spawn config
    let spawn_b = config.spawn_b;
    let fpv_b_pos = Vec3::new(spawn_b.x as f32, spawn_b.z as f32, spawn_b.y as f32);
    commands.spawn((
        FpvCamera { drone: FpvDrone::B },
        Camera3d::default(),
        Camera {
            viewport: Some(Viewport {
                physical_position: UVec2::new(fpv_w, main_h),
                physical_size: UVec2::new(w - fpv_w, fpv_h),
                ..default()
            }),
            order: 2,
            clear_color: ClearColorConfig::Custom(Color::srgb(0.1, 0.05, 0.05)),
            ..default()
        },
        Transform::from_translation(fpv_b_pos),
    ));
}

/// Read mouse input for orbit camera (god-view only).
fn orbit_input(
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut motion_events: EventReader<MouseMotion>,
    mut scroll_events: EventReader<MouseWheel>,
    mut orbit: ResMut<OrbitState>,
    mode: Res<CameraMode>,
) {
    // Always drain events to prevent buildup
    let mut motion_delta = Vec2::ZERO;
    for ev in motion_events.read() {
        motion_delta += ev.delta;
    }
    let mut scroll_delta = 0.0_f32;
    for ev in scroll_events.read() {
        scroll_delta += ev.y;
    }

    if *mode != CameraMode::GodView {
        return;
    }

    // Right-click or left-click drag: orbit
    if mouse_buttons.pressed(MouseButton::Right) || mouse_buttons.pressed(MouseButton::Left) {
        orbit.yaw -= motion_delta.x * 0.005;
        orbit.pitch = (orbit.pitch - motion_delta.y * 0.005).clamp(0.1, 1.55);
    } else if mouse_buttons.pressed(MouseButton::Middle) {
        // Middle-click drag: pan
        let pan_speed = orbit.distance * 0.001;
        let right = Vec3::new(orbit.yaw.cos(), 0.0, -orbit.yaw.sin());
        let fwd = Vec3::new(orbit.yaw.sin(), 0.0, orbit.yaw.cos());
        orbit.center += right * motion_delta.x * pan_speed;
        orbit.center -= fwd * motion_delta.y * pan_speed;
    }

    // Scroll: zoom
    if scroll_delta.abs() > 0.0 {
        orbit.distance = (orbit.distance - scroll_delta * 0.5).clamp(2.0, 30.0);
    }
}

fn update_main_camera(
    mode: Res<CameraMode>,
    orbit: Res<OrbitState>,
    sim: Res<SimState>,
    active: Res<ActiveDrone>,
    mut camera_q: Query<&mut Transform, With<MainCamera>>,
    time: Res<Time>,
) {
    let Ok(mut cam_transform) = camera_q.get_single_mut() else {
        return;
    };

    let lerp_speed = 3.0 * time.delta_secs();

    match *mode {
        CameraMode::GodView => {
            let cos_p = orbit.pitch.cos();
            let sin_p = orbit.pitch.sin();
            let target_pos = orbit.center
                + Vec3::new(
                    cos_p * orbit.yaw.sin() * orbit.distance,
                    sin_p * orbit.distance,
                    cos_p * orbit.yaw.cos() * orbit.distance,
                );

            cam_transform.translation = cam_transform.translation.lerp(target_pos, lerp_speed);
            let target_rot = Transform::from_translation(cam_transform.translation)
                .looking_at(orbit.center, Vec3::Y)
                .rotation;
            cam_transform.rotation = cam_transform.rotation.slerp(target_rot, lerp_speed);
        }
        CameraMode::Follow => {
            let state = match *active {
                ActiveDrone::A => &sim.state_a,
                ActiveDrone::B => &sim.state_b,
            };

            let drone_pos = sim_to_bevy_pos(state);
            let bevy_fwd = sim_to_bevy_fwd(state);

            let target_pos = drone_pos - bevy_fwd * 2.0 + Vec3::Y * 1.0;
            let target_look = drone_pos + bevy_fwd * 2.0;

            cam_transform.translation = cam_transform.translation.lerp(target_pos, lerp_speed);
            let target_rot = Transform::from_translation(cam_transform.translation)
                .looking_at(target_look, Vec3::Y)
                .rotation;
            cam_transform.rotation = cam_transform.rotation.slerp(target_rot, lerp_speed);
        }
    }
}

/// Update FPV cameras to sit at each drone's position, looking forward.
fn update_fpv_cameras(
    sim: Res<SimState>,
    mut fpv_q: Query<(&FpvCamera, &mut Transform)>,
    time: Res<Time>,
) {
    let lerp_speed = 8.0 * time.delta_secs();

    for (fpv, mut transform) in &mut fpv_q {
        let state = match fpv.drone {
            FpvDrone::A => &sim.state_a,
            FpvDrone::B => &sim.state_b,
        };

        let drone_pos = sim_to_bevy_pos(state);
        let bevy_fwd = sim_to_bevy_fwd(state);

        // FPV: slightly above drone center, looking forward
        let target_pos = drone_pos + Vec3::Y * 0.03;
        let look_at = target_pos + bevy_fwd * 5.0;

        transform.translation = transform.translation.lerp(target_pos, lerp_speed);
        let target_rot = Transform::from_translation(transform.translation)
            .looking_at(look_at, Vec3::Y)
            .rotation;
        transform.rotation = transform.rotation.slerp(target_rot, lerp_speed);
    }
}

/// Resize viewports when window resizes.
fn resize_viewports(
    windows: Query<&Window>,
    mut main_q: Query<&mut Camera, With<MainCamera>>,
    mut fpv_q: Query<(&FpvCamera, &mut Camera), Without<MainCamera>>,
) {
    let window = windows.single();
    let w = window.physical_width();
    let h = window.physical_height();
    if w == 0 || h == 0 {
        return;
    }

    let main_h = (h as f32 * MAIN_HEIGHT_RATIO) as u32;
    let fpv_h = h - main_h;
    let fpv_w = w / 2;

    if let Ok(mut cam) = main_q.get_single_mut() {
        cam.viewport = Some(Viewport {
            physical_position: UVec2::new(0, 0),
            physical_size: UVec2::new(w, main_h),
            ..default()
        });
    }

    for (fpv, mut cam) in &mut fpv_q {
        let x = match fpv.drone {
            FpvDrone::A => 0,
            FpvDrone::B => fpv_w,
        };
        let size_w = match fpv.drone {
            FpvDrone::A => fpv_w,
            FpvDrone::B => w - fpv_w,
        };
        cam.viewport = Some(Viewport {
            physical_position: UVec2::new(x, main_h),
            physical_size: UVec2::new(size_w, fpv_h),
            ..default()
        });
    }
}

// ── Coordinate helpers ──

fn sim_to_bevy_pos(state: &aces_sim_core::state::DroneState) -> Vec3 {
    Vec3::new(
        state.position.x as f32,
        state.position.z as f32, // sim z-up → bevy y-up
        state.position.y as f32,
    )
}

fn sim_to_bevy_fwd(state: &aces_sim_core::state::DroneState) -> Vec3 {
    let fwd = state.forward();
    Vec3::new(fwd.x as f32, fwd.z as f32, fwd.y as f32).normalize_or(Vec3::X)
}
