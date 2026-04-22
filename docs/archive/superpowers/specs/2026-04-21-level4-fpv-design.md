# Level 4 Design: First-Person Vision

## Overview

Add forward-facing cameras to each drone, replacing omniscient opponent knowledge with vision-derived perception. The agent must locate the opponent and map obstacles through a 320x240 depth image rendered via Rust-side SDF sphere tracing. Proprioceptive state (IMU) remains available; spatial awareness comes exclusively from the camera.

This phase uses geometric rendering and analytical detection. The architecture is designed so that MuJoCo photorealistic rendering + YOLOv8-nano CNN detection can be swapped in later without structural changes.

## Camera System

### Location

New Rust module: `crates/sim-core/src/camera.rs`

### Pinhole Camera Model

Each drone has a single forward-facing camera fixed to the body frame (rotates with attitude).

```
Resolution:       320 x 240 pixels
FOV:              90 degrees (horizontal)
Focal length:     fx = fy = width / (2 * tan(fov/2)) = 160 px
Principal point:  (cx, cy) = (160, 120)
Max depth:        15.0 m (rays beyond this return max_depth)
```

The camera looks along the drone's nose direction: `+X` in body frame.

### Ray Generation

For each pixel `(u, v)`:

```
direction_body = normalize([1.0, -(u - cx) / fx, -(v - cy) / fy])
direction_world = R(q_drone) * direction_body
origin = pos_drone
```

Convention: body-frame `+X` forward, `+Y` left, `+Z` up (matching existing dynamics).

### Sphere-Traced Rendering

Reuse the existing SDF infrastructure (`environment.rs`). For each ray:

1. Start at `origin`
2. Evaluate `SDF(point)` against arena geometry (obstacles, boundaries) and opponent drone (modeled as sphere, radius 0.05 m)
3. Advance by `max(SDF, epsilon)` along the ray
4. Terminate when `SDF < hit_threshold` (hit) or `distance > max_depth` (miss)
5. Record hit distance as depth value

The opponent drone is included in the SDF as a sphere primitive during tracing, enabling natural occlusion handling — if a pillar is closer, the opponent is occluded.

### Parallelization

Parallelize over image rows using `rayon`. Each row is independent. Expected cost: ~77K rays per frame at 30 Hz = ~2.3M rays/second per drone.

### Render Rate

Camera renders at **30 Hz**. Physics runs at 1000 Hz, control at 100 Hz. A new camera frame is produced approximately every 3.3 control steps. Between camera frames, the last depth image and detection result are reused.

Implementation: track `time_since_last_render` in the simulation. Trigger render when `time_since_last_render >= 1.0 / render_hz`.

### Output

```rust
pub struct CameraFrame {
    pub depth: Vec<f32>,       // 320*240 = 76800 values, row-major
    pub width: usize,          // 320
    pub height: usize,         // 240
    pub timestamp: f64,        // simulation time when rendered
}
```

## Geometric Detection

### Location

New Rust module: `crates/sim-core/src/detection.rs`

### Approach

Analytical projection of the opponent drone into the camera frame. No CNN — this produces the same output interface a CNN would (bounding box + confidence), computed from geometry.

### Algorithm

1. Compute relative position: `d = pos_opponent - pos_self`
2. Transform to camera frame: `d_cam = R(q_self)^T * d`
3. Check if in front of camera: `d_cam.x > 0`
4. Project to pixel coordinates: `u = fx * (-d_cam.y / d_cam.x) + cx`, `v = fy * (-d_cam.z / d_cam.x) + cy`
5. Check if within image bounds: `0 <= u < 320, 0 <= v < 240`
6. Verify visibility: single sphere-trace ray from self to opponent (reuse existing `collision::line_of_sight`)
7. Compute bounding box from projected angular extent of drone sphere at measured depth

### Bounding Box Computation

The opponent drone (radius `r = 0.05 m`) at distance `d` subtends:

```
half_angle = atan(r / d)
pixel_radius = fx * tan(half_angle)
bbox = [u - pixel_radius, v - pixel_radius, 2 * pixel_radius, 2 * pixel_radius]
```

Clamp to image bounds.

### Confidence Model

Confidence degrades with distance:

```
confidence = clamp(1.0 - distance / min_confidence_distance, 0.0, 1.0)
```

Where `min_confidence_distance = 5.0 m` (configurable). At 5+ meters, confidence drops to 0 (too small to reliably detect). This models real detection degradation.

### Output

```rust
pub struct Detection {
    pub detected: bool,
    pub bbox: [f32; 4],        // [x, y, width, height] in pixels
    pub confidence: f32,       // 0.0 - 1.0
    pub depth: f32,            // distance to opponent in meters
    pub pixel_center: [f32; 2], // [u, v] projected center
}
```

## Observation Space Redesign

### Principle

A real drone has an IMU (proprioception) and a camera (exteroception). The observation space reflects this split:

- **Proprioceptive vector** (IMU-like): things the drone knows about itself
- **Depth image** (camera): spatial awareness of the environment and opponent

The agent must learn to extract opponent location and obstacle layout from the depth image. No opponent-relative state is provided directly.

### Image Input (CNN path)

The 320x240 depth image is **downsampled to 80x60** for the policy network:

- 4x downsampling via average pooling (each output pixel = mean of 4x4 block)
- Normalized to [0, 1] range: `pixel = depth / max_depth`
- Shape: `(1, 60, 80)` — single channel, height x width

Downsampling happens Python-side (numpy) after receiving the full-res image from Rust. Full resolution is kept for visualization.

### Vector Input (MLP path — 12 dimensions)

| Index | Field | Dims | Source |
|-------|-------|------|--------|
| 0-2 | own_velocity | 3 | IMU/state |
| 3-5 | own_angular_velocity | 3 | IMU/gyro |
| 6-8 | own_attitude (roll, pitch, yaw) | 3 | IMU/AHRS |
| 9 | lock_progress | 1 | lock-on system |
| 10 | being_locked_progress | 1 | lock-on system |
| 11 | nearest_obstacle_distance | 1 | SDF query |

**Removed from current 21-dim observation** (must be inferred from camera):
- opponent_relative_position (3)
- opponent_relative_velocity (3)
- opponent_visible (1)
- belief_uncertainty (1)
- time_since_last_seen (1)

### Gymnasium Space Definition

```python
observation_space = spaces.Dict({
    "image": spaces.Box(low=0.0, high=1.0, shape=(1, 60, 80), dtype=np.float32),
    "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32),
})
```

## Policy Network

### Architecture: CNN + MLP Hybrid

```
Depth image (1x60x80)
  -> Conv2d(1, 32, 8, stride=4) -> ReLU    # -> 32x14x19
  -> Conv2d(32, 64, 4, stride=2) -> ReLU   # -> 64x6x9
  -> Conv2d(64, 64, 3, stride=1) -> ReLU   # -> 64x4x7
  -> Flatten -> Linear(1792, 128) -> ReLU   # -> 128-dim image features

IMU vector (12)
  -> Linear(12, 64) -> ReLU                 # -> 64-dim

Concat(128, 64) = 192
  -> Linear(192, 128) -> ReLU
  -> Linear(128, 128) -> ReLU
  -> Policy head: Linear(128, 4)            # action (motor deltas)
  -> Value head: Linear(128, 1)             # state value
```

### Implementation

Custom `stable-baselines3` feature extractor (`CnnImuExtractor`) registered with PPO. Placed in `python/aces/policy.py`.

### Training Considerations

- Larger observation space means slower training convergence
- Frame stacking (2-4 consecutive frames) may help with motion estimation — start without it, add if needed
- Learning rate may need reduction vs. current MLP training

## Render Pipeline

```
Timestep hierarchy:

  1000 Hz  Physics step (dynamics + wind + collision)
           |
   100 Hz  Control step (policy inference -> motor commands)
           |  reuses last camera frame between renders
    30 Hz  Camera render (sphere-trace depth, run detection)
```

Per control step:
1. Check if camera render is due (`time_since_last_render >= 1/30`)
2. If yes: ray-trace depth map, run geometric detection, store as current frame
3. If no: reuse previous frame
4. Build observation: downsample depth + IMU vector
5. Run policy inference
6. Apply motor commands

## Py-Bridge Extensions

### StepResult Changes

Add to the existing `StepResult` struct:

```rust
// Camera data (only populated on render frames, None otherwise)
pub depth_image: Option<Vec<f32>>,      // 320*240 flattened depth values
pub detection: Option<DetectionResult>, // geometric detection output

// Camera metadata
pub camera_rendered: bool,              // true if this step rendered a new frame
```

When `camera_rendered` is false, Python-side code uses the previously cached frame.

### New Python-Accessible Methods

```rust
#[pymethods]
impl Simulation {
    /// Get camera intrinsics for visualization
    fn camera_intrinsics(&self) -> (f32, f32, f32, f32); // fx, fy, cx, cy
}
```

### Detection Exposed as Python Dict

```python
{
    "detected": bool,
    "bbox": [x, y, w, h],
    "confidence": float,
    "depth": float,
    "pixel_center": [u, v],
}
```

## Configuration

### Additions to `configs/rules.toml`

```toml
[camera]
width = 320
height = 240
fov_deg = 90.0
render_hz = 30
max_depth = 15.0
policy_width = 80      # downsampled resolution for policy
policy_height = 60

[detection]
drone_radius = 0.05
min_confidence_distance = 5.0
```

## Visualization Updates

### Rerun Extensions (`viz.py`)

- **Depth image panel**: Log 80x60 (or full 320x240) depth images as grayscale 2D images per drone
- **Detection overlay**: Draw bounding box on depth image when opponent detected
- **Camera frustum**: Render FOV cone in 3D view (replace or augment existing FOV cone with actual camera frustum geometry)
- **Detection markers**: Highlight detected opponent position in 3D view

## Testing Strategy

### Unit Tests (Rust)

- `camera.rs`: ray generation correctness (center pixel -> nose direction, corner pixels -> FOV edges)
- `camera.rs`: known scene depth values (ray at obstacle at known distance)
- `camera.rs`: opponent appears in depth map at correct distance
- `camera.rs`: occluded opponent does not appear
- `detection.rs`: opponent in front -> detected with correct bbox
- `detection.rs`: opponent behind -> not detected
- `detection.rs`: opponent occluded by pillar -> not detected
- `detection.rs`: confidence degrades with distance
- `detection.rs`: bbox size shrinks with distance

### Integration Tests (Python)

- `test_env.py`: new observation space shape is correct (Dict with image + vector)
- `test_env.py`: depth image values are in valid range [0, max_depth]
- `test_env.py`: detection matches known scenarios
- `test_env.py`: camera renders at ~30 Hz (check render count over N steps)
- Policy network: forward pass with dummy observation produces valid action shape

### Performance Tests

- Measure ray-tracing time per frame (target: < 10 ms for 320x240)
- Measure full control loop time with camera rendering
- Verify 30 Hz render doesn't bottleneck 100 Hz control

## Future: MuJoCo Upgrade Path

When adding photorealistic rendering:

1. Add `mujoco` Python dependency, mirror arena geometry in MJCF XML
2. Replace Rust ray tracer with MuJoCo `mj_render` calls (RGB + depth)
3. Add RGB channels: observation image becomes `(4, 60, 80)` RGBD
4. Replace geometric detection with YOLOv8-nano CNN trained on MuJoCo renders
5. CNN encoder input channels change from 1 to 4; rest of architecture unchanged
6. Rust ray tracer remains available as fallback / fast mode

The `Detection` output interface stays the same regardless of backend.
