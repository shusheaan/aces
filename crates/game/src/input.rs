use bevy::prelude::*;

use crate::camera::CameraMode;
use crate::config::GameConfig;
use crate::simulation::{reset_sim, SimState};
use crate::{ActiveDrone, GameState};

pub struct InputPlugin;

impl Plugin for InputPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<DroneCommand>()
            .add_systems(Update, (read_input, handle_actions));
    }
}

/// High-level drone control command, produced by input system.
#[derive(Resource, Default, Debug)]
pub struct DroneCommand {
    pub roll: f32,     // [-1, 1]
    pub pitch: f32,    // [-1, 1]
    pub yaw: f32,      // [-1, 1]
    pub throttle: f32, // [-1, 1], 0 = hover
}

const DEADZONE: f32 = 0.15;

fn apply_deadzone(val: f32) -> f32 {
    if val.abs() < DEADZONE {
        0.0
    } else {
        (val - val.signum() * DEADZONE) / (1.0 - DEADZONE)
    }
}

/// Read keyboard and gamepad axes → DroneCommand.
fn read_input(
    keyboard: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut cmd: ResMut<DroneCommand>,
) {
    // Try gamepad first
    let mut from_gamepad = false;
    for gamepad in &gamepads {
        let lx = apply_deadzone(gamepad.get(GamepadAxis::LeftStickX).unwrap_or(0.0));
        let ly = apply_deadzone(gamepad.get(GamepadAxis::LeftStickY).unwrap_or(0.0));
        let rx = apply_deadzone(gamepad.get(GamepadAxis::RightStickX).unwrap_or(0.0));
        let rt = gamepad.get(GamepadAxis::RightZ).unwrap_or(0.0); // RT
        let lt = gamepad.get(GamepadAxis::LeftZ).unwrap_or(0.0); // LT

        if lx.abs() > 0.0 || ly.abs() > 0.0 || rx.abs() > 0.0 || rt.abs() > 0.01 || lt.abs() > 0.01
        {
            cmd.roll = lx;
            cmd.pitch = ly;
            cmd.yaw = rx;
            cmd.throttle = (rt - lt).clamp(-1.0, 1.0);
            from_gamepad = true;
            break;
        }
    }

    if !from_gamepad {
        // Keyboard fallback
        let mut roll = 0.0_f32;
        let mut pitch = 0.0_f32;
        let mut yaw = 0.0_f32;
        let mut throttle = 0.0_f32;

        if keyboard.pressed(KeyCode::KeyA) {
            roll -= 1.0;
        }
        if keyboard.pressed(KeyCode::KeyD) {
            roll += 1.0;
        }
        if keyboard.pressed(KeyCode::KeyW) {
            pitch += 1.0;
        }
        if keyboard.pressed(KeyCode::KeyS) {
            pitch -= 1.0;
        }
        if keyboard.pressed(KeyCode::KeyQ) {
            yaw -= 1.0;
        }
        if keyboard.pressed(KeyCode::KeyE) {
            yaw += 1.0;
        }
        if keyboard.pressed(KeyCode::ShiftLeft) {
            throttle += 1.0;
        }
        if keyboard.pressed(KeyCode::ControlLeft) {
            throttle -= 1.0;
        }

        cmd.roll = roll;
        cmd.pitch = pitch;
        cmd.yaw = yaw;
        cmd.throttle = throttle;
    }
}

/// Handle discrete actions: switch drone, camera, pause, reset.
#[allow(clippy::too_many_arguments)]
fn handle_actions(
    keyboard: Res<ButtonInput<KeyCode>>,
    gamepads: Query<&Gamepad>,
    mut active: ResMut<ActiveDrone>,
    mut camera_mode: ResMut<CameraMode>,
    current_state: Res<State<GameState>>,
    mut next_state: ResMut<NextState<GameState>>,
    mut sim: ResMut<SimState>,
    config: Res<GameConfig>,
) {
    // Switch drone: Space / Y button
    let switch = keyboard.just_pressed(KeyCode::Space)
        || gamepads
            .iter()
            .any(|g| g.just_pressed(GamepadButton::North));
    if switch {
        *active = match *active {
            ActiveDrone::A => ActiveDrone::B,
            ActiveDrone::B => ActiveDrone::A,
        };
    }

    // Switch camera: Tab / B button
    let cam_switch = keyboard.just_pressed(KeyCode::Tab)
        || gamepads.iter().any(|g| g.just_pressed(GamepadButton::East));
    if cam_switch {
        *camera_mode = match *camera_mode {
            CameraMode::GodView => CameraMode::Follow,
            CameraMode::Follow => CameraMode::GodView,
        };
    }

    // Pause: P / Start
    let pause = keyboard.just_pressed(KeyCode::KeyP)
        || gamepads
            .iter()
            .any(|g| g.just_pressed(GamepadButton::Start));
    if pause {
        match current_state.get() {
            GameState::Running => next_state.set(GameState::Paused),
            GameState::Paused => next_state.set(GameState::Running),
        }
    }

    // Reset: R / Back
    let reset = keyboard.just_pressed(KeyCode::KeyR)
        || gamepads
            .iter()
            .any(|g| g.just_pressed(GamepadButton::Select));
    if reset {
        reset_sim(&mut sim, &config);
    }
}
