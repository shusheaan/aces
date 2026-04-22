use bevy::prelude::*;

use crate::camera::MainCamera;
use crate::simulation::SimState;
use crate::ActiveDrone;

pub struct MarkerPlugin;

impl Plugin for MarkerPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_marker)
            .add_systems(Update, update_marker);
    }
}

#[derive(Component)]
struct EnemyMarker;

#[derive(Component)]
struct EnemyMarkerText;

fn spawn_marker(mut commands: Commands) {
    // Enemy marker container (absolute positioned)
    commands
        .spawn((
            EnemyMarker,
            Node {
                position_type: PositionType::Absolute,
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                ..default()
            },
            Visibility::Hidden,
        ))
        .with_children(|parent| {
            // Diamond symbol
            parent.spawn((
                Text::new("◇"),
                TextFont {
                    font_size: 32.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.3, 0.3)),
            ));
            // Distance text
            parent.spawn((
                EnemyMarkerText,
                Text::new("0.0m"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.8, 0.8)),
            ));
        });
}

#[allow(clippy::type_complexity)]
fn update_marker(
    sim: Res<SimState>,
    active: Res<ActiveDrone>,
    camera_q: Query<(&Camera, &GlobalTransform), With<MainCamera>>,
    mut marker_q: Query<
        (&mut Node, &mut Visibility),
        (With<EnemyMarker>, Without<EnemyMarkerText>),
    >,
    mut text_q: Query<&mut Text, With<EnemyMarkerText>>,
) {
    let Ok((camera, cam_gt)) = camera_q.get_single() else {
        return;
    };
    let Ok((mut node, mut vis)) = marker_q.get_single_mut() else {
        return;
    };

    let (sees_enemy, enemy_state) = match *active {
        ActiveDrone::A => (sim.a_sees_b, &sim.state_b),
        ActiveDrone::B => (sim.b_sees_a, &sim.state_a),
    };

    if !sees_enemy {
        *vis = Visibility::Hidden;
        return;
    }

    // Enemy position in Bevy coords
    let enemy_bevy = Vec3::new(
        enemy_state.position.x as f32,
        enemy_state.position.z as f32,
        enemy_state.position.y as f32,
    );

    // Project to screen
    if let Ok(viewport_pos) = camera.world_to_viewport(cam_gt, enemy_bevy) {
        *vis = Visibility::Visible;
        node.left = Val::Px(viewport_pos.x - 16.0);
        node.top = Val::Px(viewport_pos.y - 20.0);

        if let Ok(mut text) = text_q.get_single_mut() {
            **text = format!("{:.1}m", sim.distance);
        }
    } else {
        *vis = Visibility::Hidden;
    }
}
