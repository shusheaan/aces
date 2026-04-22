use bevy::prelude::*;

use crate::simulation::SimState;
use crate::{ActiveDrone, GameState};

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HelpVisible>()
            .add_systems(Startup, spawn_hud)
            .add_systems(
                Update,
                (
                    update_hud,
                    update_pause_overlay,
                    toggle_help,
                    update_help_overlay,
                ),
            );
    }
}

// ── HUD marker components ──

#[derive(Component)]
struct HudActiveDrone;

#[derive(Component)]
struct HudLockOn;

#[derive(Component)]
struct HudTelemetry;

#[derive(Component)]
struct HudEnemyInfo;

#[derive(Component)]
struct HudWallWarning;

#[derive(Component)]
struct PauseOverlay;

#[derive(Component)]
struct HelpOverlay;

#[derive(Component)]
struct HelpHint;

#[derive(Resource, Default)]
struct HelpVisible(bool);

const HELP_TEXT: &str = "\
── Drone Control ──
W / S         Pitch
A / D         Roll
Q / E         Yaw
L-Shift       Throttle up
L-Ctrl        Throttle down
Space         Switch drone

── View ──
Tab           Camera mode
P             Pause / Resume
R             Reset
H             Toggle help

── God-View Mouse ──
Right-drag    Orbit
Scroll        Zoom
Mid-drag      Pan

── Gamepad ──
L-Stick       Roll / Pitch
R-Stick X     Yaw
RT / LT       Throttle
Y             Switch drone
B             Camera
Start         Pause
Back          Reset";

fn spawn_hud(mut commands: Commands) {
    // Root UI container
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(100.0),
            flex_direction: FlexDirection::Column,
            justify_content: JustifyContent::SpaceBetween,
            ..default()
        })
        .with_children(|root| {
            // ── Top row ──
            root.spawn(Node {
                width: Val::Percent(100.0),
                justify_content: JustifyContent::SpaceBetween,
                padding: UiRect::all(Val::Px(12.0)),
                ..default()
            })
            .with_children(|top| {
                // Active drone label (top-left)
                top.spawn((
                    HudActiveDrone,
                    Text::new("[A] Drone A"),
                    TextFont {
                        font_size: 22.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.0, 0.85, 1.0)),
                ));

                // Lock-on info (top-right)
                top.spawn((
                    HudLockOn,
                    Text::new("Lock: 0%"),
                    TextFont {
                        font_size: 20.0,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.3, 0.3)),
                ));
            });

            // ── Center: wall warning ──
            root.spawn(Node {
                width: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                ..default()
            })
            .with_children(|center| {
                center.spawn((
                    HudWallWarning,
                    Text::new(""),
                    TextFont {
                        font_size: 28.0,
                        ..default()
                    },
                    TextColor(Color::srgb(1.0, 0.2, 0.2)),
                ));
            });

            // ── Bottom row ──
            root.spawn(Node {
                width: Val::Percent(100.0),
                justify_content: JustifyContent::SpaceBetween,
                padding: UiRect::all(Val::Px(12.0)),
                ..default()
            })
            .with_children(|bottom| {
                // Telemetry (bottom-left)
                bottom.spawn((
                    HudTelemetry,
                    Text::new("ALT 0.0m  SPD 0.0m/s"),
                    TextFont {
                        font_size: 18.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.8, 0.8, 0.8)),
                ));

                // Enemy info (bottom-right)
                bottom.spawn((
                    HudEnemyInfo,
                    Text::new("DIST --  HIDDEN"),
                    TextFont {
                        font_size: 18.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.8, 0.8, 0.8)),
                ));
            });
        });

    // Help hint — small "[H] Help" label, hidden when overlay is open
    commands
        .spawn((
            HelpHint,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(12.0),
                top: Val::Px(40.0),
                ..default()
            },
        ))
        .with_children(|hint| {
            hint.spawn((
                Text::new("[H] Help"),
                TextFont {
                    font_size: 14.0,
                    ..default()
                },
                TextColor(Color::srgba(0.6, 0.6, 0.6, 0.7)),
            ));
        });

    // Help overlay (hidden by default)
    commands
        .spawn((
            HelpOverlay,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Px(12.0),
                top: Val::Px(40.0),
                padding: UiRect::all(Val::Px(12.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.75)),
            Visibility::Hidden,
        ))
        .with_children(|overlay| {
            overlay.spawn((
                Text::new(HELP_TEXT),
                TextFont {
                    font_size: 13.0,
                    ..default()
                },
                TextColor(Color::srgb(0.9, 0.9, 0.9)),
            ));
        });

    // Pause overlay (hidden by default)
    commands
        .spawn((
            PauseOverlay,
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                position_type: PositionType::Absolute,
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
            Visibility::Hidden,
        ))
        .with_children(|overlay| {
            overlay.spawn((
                Text::new("PAUSED"),
                TextFont {
                    font_size: 64.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
        });

    // ── FPV split labels (bottom portion) ──
    // Container pinned to bottom 35% of screen
    commands
        .spawn(Node {
            width: Val::Percent(100.0),
            height: Val::Percent(35.0),
            position_type: PositionType::Absolute,
            bottom: Val::Px(0.0),
            left: Val::Px(0.0),
            flex_direction: FlexDirection::Row,
            ..default()
        })
        .with_children(|fpv_row| {
            // Left FPV label (Drone A)
            fpv_row
                .spawn(Node {
                    width: Val::Percent(50.0),
                    padding: UiRect::all(Val::Px(6.0)),
                    ..default()
                })
                .with_children(|left| {
                    left.spawn((
                        Text::new("FPV — Drone A"),
                        TextFont {
                            font_size: 16.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.0, 0.85, 1.0)),
                    ));
                });

            // Right FPV label (Drone B)
            fpv_row
                .spawn(Node {
                    width: Val::Percent(50.0),
                    padding: UiRect::all(Val::Px(6.0)),
                    ..default()
                })
                .with_children(|right| {
                    right.spawn((
                        Text::new("FPV — Drone B"),
                        TextFont {
                            font_size: 16.0,
                            ..default()
                        },
                        TextColor(Color::srgb(1.0, 0.6, 0.0)),
                    ));
                });
        });
}

#[allow(clippy::type_complexity)]
fn update_hud(
    sim: Res<SimState>,
    active: Res<ActiveDrone>,
    mut active_text: Query<(&mut Text, &mut TextColor), With<HudActiveDrone>>,
    mut lock_text: Query<
        &mut Text,
        (
            With<HudLockOn>,
            Without<HudActiveDrone>,
            Without<HudTelemetry>,
            Without<HudEnemyInfo>,
            Without<HudWallWarning>,
        ),
    >,
    mut telem_text: Query<
        &mut Text,
        (
            With<HudTelemetry>,
            Without<HudActiveDrone>,
            Without<HudLockOn>,
            Without<HudEnemyInfo>,
            Without<HudWallWarning>,
        ),
    >,
    mut enemy_text: Query<
        &mut Text,
        (
            With<HudEnemyInfo>,
            Without<HudActiveDrone>,
            Without<HudLockOn>,
            Without<HudTelemetry>,
            Without<HudWallWarning>,
        ),
    >,
    mut wall_text: Query<
        &mut Text,
        (
            With<HudWallWarning>,
            Without<HudActiveDrone>,
            Without<HudLockOn>,
            Without<HudTelemetry>,
            Without<HudEnemyInfo>,
        ),
    >,
) {
    let (player_state, lock_progress, sdf, sees_enemy) = match *active {
        ActiveDrone::A => (&sim.state_a, sim.lock_a.progress(), sim.sdf_a, sim.a_sees_b),
        ActiveDrone::B => (&sim.state_b, sim.lock_b.progress(), sim.sdf_b, sim.b_sees_a),
    };

    let being_locked = match *active {
        ActiveDrone::A => sim.lock_b.progress(),
        ActiveDrone::B => sim.lock_a.progress(),
    };

    // Active drone label
    if let Ok((mut text, mut color)) = active_text.get_single_mut() {
        let (label, c) = match *active {
            ActiveDrone::A => ("[A] Drone A", Color::srgb(0.0, 0.85, 1.0)),
            ActiveDrone::B => ("[B] Drone B", Color::srgb(1.0, 0.6, 0.0)),
        };
        **text = label.to_string();
        *color = TextColor(c);
    }

    // Lock-on
    if let Ok(mut text) = lock_text.get_single_mut() {
        let pct = (lock_progress * 100.0) as i32;
        let enemy_pct = (being_locked * 100.0) as i32;
        **text = format!("Lock: {}%  Locked: {}%", pct, enemy_pct);
    }

    // Telemetry
    if let Ok(mut text) = telem_text.get_single_mut() {
        let alt = player_state.position.z;
        let spd = player_state.velocity.norm();
        **text = format!("ALT {:.1}m  SPD {:.1}m/s", alt, spd);
    }

    // Enemy info
    if let Ok(mut text) = enemy_text.get_single_mut() {
        let vis = if sees_enemy { "VISIBLE" } else { "HIDDEN" };
        **text = format!("DIST {:.1}m  {}", sim.distance, vis);
    }

    // Wall warning
    if let Ok(mut text) = wall_text.get_single_mut() {
        if sdf < 0.5 {
            **text = format!("!! WALL {:.2}m !!", sdf);
        } else {
            **text = String::new();
        }
    }
}

fn update_pause_overlay(
    state: Res<State<GameState>>,
    mut query: Query<&mut Visibility, With<PauseOverlay>>,
) {
    if let Ok(mut vis) = query.get_single_mut() {
        *vis = match state.get() {
            GameState::Paused => Visibility::Visible,
            GameState::Running => Visibility::Hidden,
        };
    }
}

fn toggle_help(keyboard: Res<ButtonInput<KeyCode>>, mut visible: ResMut<HelpVisible>) {
    if keyboard.just_pressed(KeyCode::KeyH) {
        visible.0 = !visible.0;
    }
}

#[allow(clippy::type_complexity)]
fn update_help_overlay(
    visible: Res<HelpVisible>,
    mut overlay_q: Query<&mut Visibility, With<HelpOverlay>>,
    mut hint_q: Query<
        &mut Visibility,
        (With<HelpHint>, Without<HelpOverlay>, Without<PauseOverlay>),
    >,
) {
    if let Ok(mut vis) = overlay_q.get_single_mut() {
        *vis = if visible.0 {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
    }
    if let Ok(mut vis) = hint_q.get_single_mut() {
        *vis = if visible.0 {
            Visibility::Hidden
        } else {
            Visibility::Visible
        };
    }
}
