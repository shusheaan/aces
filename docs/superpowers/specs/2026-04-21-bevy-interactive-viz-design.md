# Bevy Interactive Visualization — Design Spec

## Goal

Add an interactive 3D visualization to ACES: a Bevy game crate (`crates/game/`) that renders the arena with two drones, supports Xbox controller and keyboard input for manual piloting, and displays HUD information (wall warnings, enemy markers, lock-on progress). The screen is split into a main view (top 65%) and two FPV first-person cameras (bottom 35%, side by side) for each drone.

The sim-core physics engine remains unchanged. Bevy is purely a rendering/interaction frontend. The non-player drone is controlled by MPPI (existing controller) as AI opponent.

## Project Structure

```
crates/game/
├── Cargo.toml        # deps: sim-core, mppi, bevy (0.15), toml, serde, nalgebra, rand
└── src/
    ├── main.rs       # App setup, plugin registration, GameState / ActiveDrone enums
    ├── config.rs     # Serde deserialization of arena.toml / drone.toml / rules.toml
    ├── arena.rs      # Spawn 3D meshes: floor, walls (semi-transparent), pillars, lighting
    ├── drone.rs      # Drone entities: cross-shaped mesh, heading arrow, trail (gizmos)
    ├── input.rs      # Keyboard + Xbox gamepad → DroneCommand resource
    ├── simulation.rs # Drive sim-core at 100Hz FixedUpdate, MPPI AI, lock-on, sync transforms
    ├── hud.rs        # UI overlay: active drone, lock-on bar, telemetry, wall warning, FPV labels
    ├── camera.rs     # 3 cameras: main (god-view/follow) + 2 FPV, viewport management
    └── marker.rs     # Enemy diamond marker projected from 3D to screen space
```

Added as `"crates/game"` to workspace members in root `Cargo.toml`. Workspace dependencies include `bevy = "0.15"`, `toml = "0.8"`, `serde`.

## Architecture

```
┌───────────────────────────────────────────────┐
│                  Bevy App                      │
│                                                │
│  FixedUpdate (100Hz)          Render (vsync)   │
│  ┌─────────────┐              ┌────────────┐   │
│  │ input_system │─→ DroneCmd  │ 3 cameras  │   │
│  └─────────────┘      │       │ HUD nodes  │   │
│  ┌─────────────┐      ▼       │ markers    │   │
│  │  sim_step    │─→ SimState  └────────────┘   │
│  │ (sim-core)   │      │                       │
│  └─────────────┘      ▼                        │
│  ┌─────────────┐  ┌────────┐                   │
│  │ hud_update   │  │sync_sys│→ Transform       │
│  └─────────────┘  └────────┘                   │
└───────────────────────────────────────────────┘
         │
         ▼
   sim-core (unchanged)         mppi (unchanged)
   - Dynamics (1000Hz RK4)      - MPPI optimizer
   - SDF arena, collision       - Pursuit/evasion costs
   - Lock-on logic
   - Wind (Ornstein-Uhlenbeck)
```

## Screen Layout

```
┌─────────────────────────────────────────────────┐
│                                                 │
│            Main Camera (65%)                    │
│      God-view / Follow-cam, Tab to switch       │
│      HUD overlaid: lock-on, telemetry,          │
│      wall warnings, enemy marker (◇)            │
│                                                 │
├────────────────────┬────────────────────────────┤
│                    │                            │
│  FPV — Drone A     │  FPV — Drone B             │
│  (cyan label)      │  (orange label)            │
│  First-person      │  First-person              │
│  from drone A      │  from drone B              │
│                    │                            │
│       35% height, 50% width each                │
└────────────────────┴────────────────────────────┘
```

### Camera Details

**3 Camera entities**, each with a `Viewport` restricting its render area:

| Camera | Viewport | Position | Behavior |
|--------|----------|----------|----------|
| Main (`MainCamera`) | Top 65% of window, full width | God-view: `(5, 14, 5)` looking at center; Follow: behind active drone | Smooth lerp transitions, `lerp_speed = 3.0 * dt` |
| FPV A (`FpvCamera { drone: A }`) | Bottom-left 35% height, 50% width | At drone A position + 0.03m up, looking forward | Fast lerp tracking, `lerp_speed = 8.0 * dt` |
| FPV B (`FpvCamera { drone: B }`) | Bottom-right 35% height, 50% width | At drone B position + 0.03m up, looking forward | Fast lerp tracking, `lerp_speed = 8.0 * dt` |

- Camera `order`: Main=0, FPV A=1, FPV B=2
- FPV cameras have distinct `clear_color` for visual separation (dark blue for A, dark red for B)
- All viewports dynamically resize via `resize_viewports` system on window resize
- `MAIN_HEIGHT_RATIO = 0.65` constant controls the split

### Main Camera Modes

**God-View (default):**
- Position: `(5.0, 14.0, 5.0)` in Bevy coords, looking at arena center `(5.0, 0.0, 5.0)`
- Provides overview of entire 10x10x3 arena

**Follow-Cam:**
- Position: 2.0m behind + 1.0m above active drone
- Look-at: drone position + forward * 2.0
- Smooth follow with lerp

**Switching:** Tab (keyboard) or B (gamepad) toggles. Camera lerps to new target smoothly.

### FPV Camera Behavior

- Positioned at drone's world position + 0.03m upward offset (above body center)
- Look direction: drone's forward vector * 5.0 (derived from `DroneState::forward()`)
- Faster lerp (8.0x) than main camera for responsive first-person feel
- Both FPV cameras always active — you see both drones' perspectives simultaneously

## Simulation Loop

### FixedUpdate schedule (100Hz, dt = 0.01s)

Systems run in order within `sim_step`:

1. **Read DroneCommand** from input system
2. **command_to_motors**: Convert DroneCommand to `Vector4<f64>` motor thrusts via inverse X-config mixing matrix:
   ```
   collective = hover_thrust * 4 + throttle * hover_thrust * 4
   tau_x = roll * max_thrust * arm_length * 2
   tau_y = pitch * max_thrust * arm_length * 2
   tau_z = yaw * max_thrust * arm_length (half scale for yaw)

   s = 1/√2 (X-config factor)
   m0 = collective/4 + tau_x/(4·d·s) + tau_y/(4·d·s) + tau_z/(4·c)
   m1 = collective/4 - tau_x/(4·d·s) + tau_y/(4·d·s) - tau_z/(4·c)
   m2 = collective/4 - tau_x/(4·d·s) - tau_y/(4·d·s) + tau_z/(4·c)
   m3 = collective/4 + tau_x/(4·d·s) - tau_y/(4·d·s) - tau_z/(4·c)

   each clamped to [0, max_thrust_per_motor]
   ```
3. **MPPI AI**: `mppi.compute_action(&ai_state, &player_state, pursuit=true)` for non-player drone
4. **Physics substeps**: Loop `substeps` (10) times, calling `step_rk4()` at dt_sim = 0.001s per drone
5. **Wind**: `WindModel::step(dt_ctrl, &mut rng)` → external force vector applied to both drones (Ornstein-Uhlenbeck process)
6. **Lock-on**: `LockOnTracker::update()` for both drones, checks FOV cone + range + line-of-sight
7. **Collision/SDF**: `arena.is_collision()`, `arena.sdf()` for wall proximity
8. **Visibility**: `check_line_of_sight()` between both drone positions
9. **sync_drone_transforms**: Copy `DroneState.position` and `DroneState.attitude` → Bevy `Transform` with coordinate swap

### Coordinate Transform

sim-core uses z-up: `(x, y, z)`. Bevy uses y-up. Mapping:
```
sim (x, y, z) → Bevy (x, z, y)
```

Applied in `sync_drone_transforms`, `update_fpv_cameras`, `update_main_camera`, and `update_marker`.

Quaternion mapping: `(qw, qi, qj, qk)` → Bevy `Quat::from_xyzw(qi, qk, qj, qw)` (swap j↔k for y↔z).

## Input

### Keyboard

| Key | Function |
|-----|----------|
| W / S | Pitch forward / backward |
| A / D | Roll left / right |
| Q / E | Yaw left / right |
| Left Shift | Throttle up |
| Left Ctrl | Throttle down |
| **Space** | **Switch controlled drone (A ↔ B)** |
| Tab | Switch main camera mode (god-view ↔ follow) |
| P | Pause / resume simulation |
| R | Reset simulation |

### Xbox Gamepad

| Input | Function |
|-------|----------|
| Left Stick X/Y | Roll / Pitch |
| Right Stick X | Yaw |
| Right Trigger (RT) | Throttle up |
| Left Trigger (LT) | Throttle down |
| Y Button | Switch controlled drone |
| B Button | Switch camera mode |
| Start | Pause / resume |
| Back/Select | Reset |

### Input Processing

- All axes normalized to `[-1.0, 1.0]`
- Gamepad dead zone: 0.15 with linear rescaling beyond dead zone
- Keyboard inputs are binary (±1.0)
- Both input sources merged: if gamepad has any non-zero axis, gamepad takes priority; otherwise keyboard

### DroneCommand Resource

```rust
#[derive(Resource, Default)]
pub struct DroneCommand {
    pub roll: f32,      // [-1, 1]
    pub pitch: f32,     // [-1, 1]
    pub yaw: f32,       // [-1, 1]
    pub throttle: f32,  // [-1, 1], 0 = hover
}
```

## HUD

Overlaid on the main camera viewport (top 65% of screen).

```
┌───────────────────────────────────────────────┐
│ [A] Drone A                    Lock: 47%       │
│                                Locked: 12%     │
│                                                │
│            ◇ Enemy (3.2m)                      │
│                                                │
│  !! WALL 0.30m !!                              │
│                                                │
│ ALT 1.5m  SPD 2.1m/s      DIST 3.2m   VISIBLE │
└───────────────────────────────────────────────┘
```

### HUD Elements

| Element | Position | Data Source | Update Condition |
|---------|----------|-------------|------------------|
| Active drone label | Top-left | `ActiveDrone` resource | On Space press |
| Lock-on progress | Top-right | `LockOnTracker::progress()` | Every frame |
| Being-locked progress | Top-right | Opponent's `LockOnTracker::progress()` | Every frame |
| Wall warning | Center | `arena.sdf(player_pos)` | When SDF < 0.5m |
| Telemetry (ALT, SPD) | Bottom-left | `DroneState.position.z`, `velocity.norm()` | Every frame |
| Enemy distance + visibility | Bottom-right | `distance`, `a_sees_b` / `b_sees_a` | Every frame |
| Enemy marker (◇) | Projected screen pos | `Camera::world_to_viewport()` | When enemy visible |
| FPV labels | Bottom split area | Static | Always |
| Pause overlay | Full screen | `GameState` | When paused |

### Implementation

- Bevy UI (`Node`, `Text`, `TextFont`, `TextColor`, `BackgroundColor`)
- Enemy marker: absolute-positioned `Node` with `◇` text, repositioned via `world_to_viewport()` on main camera
- Wall warning text turns red, appears only when SDF < 0.5m
- Pause overlay: semi-transparent black background with "PAUSED" text, visibility toggled by `GameState`
- FPV labels: fixed row at bottom 35%, two 50%-width containers with colored text ("FPV — Drone A" cyan, "FPV — Drone B" orange)

## Arena Rendering

Source: `configs/arena.toml`

- **Floor**: `Plane3d` (10m × 10m), gray `StandardMaterial`, at Bevy `(5, 0, 5)`
- **Walls**: 4 vertical + 1 ceiling. Semi-transparent (`alpha = 0.15`, `AlphaMode::Blend`, `double_sided: true`, `cull_mode: None`). 0.02m thick.
- **Obstacles**: `Cuboid` meshes from `[[obstacles]]` in arena.toml. Dark red (`Color::srgb(0.7, 0.2, 0.2)`). 5 floor-to-ceiling pillars (1m × 1m × 3m).
- **Lighting**: `DirectionalLight` (illuminance 8000, shadows enabled, angled -0.8 rad X, 0.4 rad Y) + `AmbientLight` (brightness 300).

### Coordinate Mapping for Arena

```
Floor:    sim (bx/2, by/2, 0) → Bevy (bx/2, 0, by/2)
Walls:    sim (cx, cy, cz) → Bevy (cx, cz, cy), size (sx, sy, sz) → (sx, sz, sy)
Pillars:  sim center (2,2,1.5) → Bevy (2, 1.5, 2), size (1, 1, 3) → (1, 3, 1)
```

## Drone Rendering

Each drone is a parent entity with child meshes:

- **Body**: `Cuboid(0.12, 0.036, 0.12)` — flat rectangular body
- **Arms**: Two `Cuboid(0.16, 0.02, 0.02)` rotated ±45° around Y — X-config cross shape
- **Heading arrow**: `Cuboid(0.15, 0.015, 0.015)` offset 0.12m forward, yellow emissive unlit material
- **Colors**: Drone A = cyan `(0.0, 0.85, 1.0)` with cyan emissive; Drone B = orange `(1.0, 0.6, 0.0)` with orange emissive
- **Trail**: `DroneTrail` component with `Vec<Vec3>` ring buffer (200 points), rendered as line gizmos each frame. Semi-transparent drone color.

### Transform Sync

```rust
// sim (x,y,z) z-up → bevy (x,z,y) y-up
transform.translation = Vec3::new(pos.x, pos.z, pos.y);
transform.rotation = Quat::from_xyzw(q.i, q.k, q.j, q.w);
```

## State Management

```rust
#[derive(States, Default, Clone, Eq, PartialEq, Debug, Hash)]
pub enum GameState {
    #[default]
    Running,
    Paused,
}

#[derive(Resource, Default, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ActiveDrone {
    #[default]
    A,
    B,
}
```

### SimState Resource

```rust
#[derive(Resource)]
pub struct SimState {
    pub state_a: DroneState,           // 13-DOF state
    pub state_b: DroneState,
    pub params: DroneParams,           // Crazyflie physical params
    pub arena: Arena,                  // SDF environment
    pub lock_a: LockOnTracker,         // A locking onto B
    pub lock_b: LockOnTracker,         // B locking onto A
    pub wind_a: WindModel,             // OU wind for drone A
    pub wind_b: WindModel,             // OU wind for drone B
    pub mppi: MppiOptimizer,           // AI opponent controller
    pub dt_ctrl: f64,                  // 0.01s
    pub substeps: usize,              // 10
    pub rng: StdRng,                   // seeded RNG for reproducibility
    // Cached per-frame data for HUD
    pub distance: f64,
    pub a_sees_b: bool,
    pub b_sees_a: bool,
    pub sdf_a: f64,
    pub sdf_b: f64,
    pub collision_a: bool,
    pub collision_b: bool,
    pub kill_a: bool,
    pub kill_b: bool,
}
```

## Configuration

All parameters loaded from existing TOML configs at startup via `GameConfig::load()`:

| Config File | Parameters Used |
|-------------|----------------|
| `configs/arena.toml` | bounds (x,y,z), spawn positions, obstacle list (center + half_extents), collision_radius |
| `configs/drone.toml` | mass, arm_length, max_motor_thrust, torque_coefficient, drag_coefficient, gravity, inertia (Ixx,Iyy,Izz), dt_ctrl, substeps |
| `configs/rules.toml` | lockon (fov_degrees→radians, lock_distance, lock_duration), mppi (num_samples, horizon, temperature, noise_std, weights), noise (wind_theta, wind_mu, wind_sigma) |

Config directory resolution: tries `./configs/`, `../../configs/` (from crate dir), then `CARGO_MANIFEST_DIR/../../configs/`.

No new config files created. All parameters come from existing configs.

## Dependencies

```toml
[dependencies]
aces-sim-core = { path = "../sim-core" }
aces-mppi = { path = "../mppi" }
bevy = { workspace = true }        # 0.15
toml = { workspace = true }        # 0.8
serde = { workspace = true }       # 1.x with derive
nalgebra = { workspace = true }    # 0.33
rand = { workspace = true }        # 0.8
```

## Borrow Checker Strategy

`SimState` contains many interrelated fields (states, arena, lock trackers, wind, rng). To avoid Rust borrow conflicts:

1. Get `&mut SimState` via `let s: &mut SimState = &mut sim;` on `ResMut<SimState>` — this enables field-level borrow splitting (unlike trait-based `DerefMut`)
2. Clone `state_a` and `state_b` after physics stepping, before passing to `lock_a.update()` and collision checks
3. Extract scalar values (`dt_ctrl`, `substeps`) into locals before mutable operations

## What This Does NOT Include (Future Work)

- Sim-core depth camera rendering in FPV (current FPV uses Bevy's 3D camera, not the sphere-traced depth image)
- Neural network policy loading (future: ONNX runtime integration)
- Multiplayer / network play
- Sound effects
- Particle effects (explosions, thrust plumes)
- Terrain / complex environments beyond box arena
- Recording / replay system
- Mouse orbit control for god-view camera

## Success Criteria

1. `cargo run -p aces-game` opens a window with the arena, 2 drones, and split-screen layout
2. Top 65%: main camera (god-view or follow-cam)
3. Bottom 35%: two side-by-side FPV cameras from each drone's perspective
4. Keyboard WASD+QE+Shift/Ctrl flies the active drone
5. Xbox controller flies the active drone (if connected)
6. Space switches which drone player controls (A ↔ B)
7. Tab switches main camera between god-view and follow-cam
8. HUD shows: active drone label, lock-on %, wall warnings < 0.5m, enemy marker (◇) with distance, telemetry
9. Physics runs at 1000Hz (10 RK4 substeps per 100Hz control), display at vsync
10. Non-player drone runs MPPI pursuit behavior
11. P pauses (overlay shown), R resets to spawn positions
12. FPV labels show "FPV — Drone A" (cyan) and "FPV — Drone B" (orange)
13. `cargo clippy -p aces-game` clean, `cargo test --workspace` passes (29 existing tests unaffected)
