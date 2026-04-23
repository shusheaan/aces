mod arena;
mod camera;
mod config;
mod drone;
mod fsm;
mod hud;
mod input;
mod marker;
mod perception;
mod policy;
mod simulation;
mod weights;

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;

use camera::CameraPlugin;
use config::GameConfig;
use drone::DronePlugin;
use hud::HudPlugin;
use input::InputPlugin;
use marker::MarkerPlugin;
use simulation::SimulationPlugin;

/// Top-level game state.
#[derive(States, Default, Clone, Eq, PartialEq, Debug, Hash)]
pub enum GameState {
    #[default]
    Running,
    Paused,
}

/// Which drone the player currently controls.
#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActiveDrone {
    #[default]
    A,
    B,
}

fn main() {
    let config = GameConfig::load();

    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "ACES — Air Combat Engagement Simulation".into(),
                resolution: (1280., 720.).into(),
                ..default()
            }),
            ..default()
        }))
        .init_state::<GameState>()
        .init_resource::<ActiveDrone>()
        .insert_resource(Time::<Fixed>::from_hz(100.0))
        .insert_resource(config)
        .add_systems(Startup, arena::spawn_arena)
        .add_plugins((
            FrameTimeDiagnosticsPlugin,
            LogDiagnosticsPlugin::default(),
            SimulationPlugin,
            InputPlugin,
            DronePlugin,
            CameraPlugin,
            HudPlugin,
            MarkerPlugin,
        ))
        .run();
}
