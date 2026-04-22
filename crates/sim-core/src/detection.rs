use nalgebra::{Rotation3, Vector3};

use crate::camera::CameraParams;
use crate::collision::{check_line_of_sight, Visibility};
use crate::environment::Arena;

/// Result of geometric opponent detection.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Whether the opponent was detected in the camera frame.
    pub detected: bool,
    /// Bounding box [x, y, width, height] in pixels.
    pub bbox: [f32; 4],
    /// Detection confidence (degrades with distance).
    pub confidence: f32,
    /// Distance to opponent in meters.
    pub depth: f32,
    /// Projected center [u, v] in pixels.
    pub pixel_center: [f32; 2],
}

impl Detection {
    /// A negative detection result.
    pub fn none() -> Self {
        Self {
            detected: false,
            bbox: [0.0; 4],
            confidence: 0.0,
            depth: 0.0,
            pixel_center: [0.0; 2],
        }
    }
}

/// Detect opponent via analytical projection into the camera frame.
///
/// Steps:
/// 1. Transform opponent position to camera frame
/// 2. Check if in front of camera
/// 3. Project to pixel coordinates
/// 4. Check image bounds
/// 5. Verify line-of-sight (not occluded by obstacle)
/// 6. Compute bounding box from angular extent
/// 7. Compute distance-based confidence
pub fn detect_opponent(
    cam: &CameraParams,
    arena: &Arena,
    self_pos: &Vector3<f64>,
    rotation: &Rotation3<f64>,
    opponent_pos: &Vector3<f64>,
    opponent_radius: f64,
    min_confidence_distance: f64,
) -> Detection {
    // 1. Relative position in world frame
    let d_world = opponent_pos - self_pos;

    // 2. Transform to camera frame (body frame)
    let d_cam = rotation.inverse() * d_world;

    // 3. Must be in front of camera (+X is forward)
    if d_cam.x <= 0.0 {
        return Detection::none();
    }

    // 4. Project to pixel coordinates
    let u = cam.fx * (-d_cam.y / d_cam.x) + cam.cx;
    let v = cam.fy * (-d_cam.z / d_cam.x) + cam.cy;

    // 5. Check if within image bounds (with some margin for bbox)
    if u < 0.0 || u >= cam.width as f64 || v < 0.0 || v >= cam.height as f64 {
        return Detection::none();
    }

    // 6. Verify line-of-sight
    if check_line_of_sight(arena, self_pos, opponent_pos) == Visibility::Occluded {
        return Detection::none();
    }

    // 7. Compute bounding box from angular extent
    let distance = d_world.norm();
    let half_angle = (opponent_radius / distance).atan();
    let pixel_radius = cam.fx * half_angle.tan();

    let bbox_x = (u - pixel_radius).max(0.0) as f32;
    let bbox_y = (v - pixel_radius).max(0.0) as f32;
    let bbox_w = (2.0 * pixel_radius).min((cam.width as f64 - bbox_x as f64).max(0.0)) as f32;
    let bbox_h = (2.0 * pixel_radius).min((cam.height as f64 - bbox_y as f64).max(0.0)) as f32;

    // 8. Confidence degrades with distance
    let confidence = (1.0 - distance / min_confidence_distance).clamp(0.0, 1.0) as f32;

    Detection {
        detected: true,
        bbox: [bbox_x, bbox_y, bbox_w, bbox_h],
        confidence,
        depth: distance as f32,
        pixel_center: [u as f32, v as f32],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Obstacle;
    use nalgebra::{UnitQuaternion, Vector3};

    fn identity_rotation() -> Rotation3<f64> {
        UnitQuaternion::identity().to_rotation_matrix()
    }

    fn test_arena() -> Arena {
        let mut arena = Arena::new(Vector3::new(10.0, 10.0, 3.0));
        for (x, y) in [(2.0, 2.0), (2.0, 8.0), (5.0, 5.0), (8.0, 2.0), (8.0, 8.0)] {
            arena.obstacles.push(Obstacle::Box {
                center: Vector3::new(x, y, 1.5),
                half_extents: Vector3::new(0.5, 0.5, 1.5),
            });
        }
        arena
    }

    #[test]
    fn test_opponent_in_front_detected() {
        let cam = CameraParams::default_fpv();
        let arena = Arena::new(Vector3::new(20.0, 20.0, 10.0));
        let self_pos = Vector3::new(5.0, 10.0, 5.0);
        let rot = identity_rotation();
        let opp_pos = Vector3::new(8.0, 10.0, 5.0); // 3m ahead

        let det = detect_opponent(&cam, &arena, &self_pos, &rot, &opp_pos, 0.05, 5.0);

        assert!(det.detected);
        assert!((det.depth - 3.0).abs() < 0.01);
        // Center should be near image center
        assert!((det.pixel_center[0] - 160.0).abs() < 1.0);
        assert!((det.pixel_center[1] - 120.0).abs() < 1.0);
        assert!(det.confidence > 0.0);
    }

    #[test]
    fn test_opponent_behind_not_detected() {
        let cam = CameraParams::default_fpv();
        let arena = Arena::new(Vector3::new(20.0, 20.0, 10.0));
        let self_pos = Vector3::new(5.0, 10.0, 5.0);
        let rot = identity_rotation();
        let opp_pos = Vector3::new(2.0, 10.0, 5.0); // behind

        let det = detect_opponent(&cam, &arena, &self_pos, &rot, &opp_pos, 0.05, 5.0);

        assert!(!det.detected);
    }

    #[test]
    fn test_opponent_occluded_by_pillar() {
        let cam = CameraParams::default_fpv();
        let arena = test_arena();
        let self_pos = Vector3::new(3.0, 5.0, 1.5);
        let rot = identity_rotation();
        let opp_pos = Vector3::new(7.0, 5.0, 1.5); // behind center pillar

        let det = detect_opponent(&cam, &arena, &self_pos, &rot, &opp_pos, 0.05, 5.0);

        assert!(!det.detected);
    }

    #[test]
    fn test_confidence_degrades_with_distance() {
        let cam = CameraParams::default_fpv();
        let arena = Arena::new(Vector3::new(20.0, 20.0, 10.0));
        let self_pos = Vector3::new(0.0, 10.0, 5.0);
        let rot = identity_rotation();

        let near = detect_opponent(
            &cam,
            &arena,
            &self_pos,
            &rot,
            &Vector3::new(2.0, 10.0, 5.0),
            0.05,
            5.0,
        );
        let far = detect_opponent(
            &cam,
            &arena,
            &self_pos,
            &rot,
            &Vector3::new(4.0, 10.0, 5.0),
            0.05,
            5.0,
        );

        assert!(near.confidence > far.confidence);
    }

    #[test]
    fn test_camera_body_frame_consistency() {
        // Object directly to the LEFT in body frame (+Y body) should appear
        // on the RIGHT side of the image (pixel u > cx).
        // Object directly to the RIGHT in body frame (-Y body) should appear
        // on the LEFT side of the image (pixel u < cx).
        let cam = CameraParams::default_fpv();
        let arena = Arena::new(Vector3::new(20.0, 20.0, 10.0));
        let self_pos = Vector3::new(5.0, 10.0, 5.0);
        let rot = identity_rotation();

        // Object to the left (+Y in world = +Y in body at identity)
        // At identity: body +Y = world +Y
        let opp_left = Vector3::new(7.0, 11.0, 5.0); // ahead and +Y
        let det_left = detect_opponent(&cam, &arena, &self_pos, &rot, &opp_left, 0.05, 10.0);
        assert!(det_left.detected);
        // Camera convention: -d_cam.y / d_cam.x mapped to pixel u
        // +Y body → negative pixel offset → u < cx (right-hand image convention)
        assert!(
            det_left.pixel_center[0] < cam.cx as f32,
            "left-side object should project to u < cx, got u={}",
            det_left.pixel_center[0]
        );

        // Object to the right (-Y in body)
        let opp_right = Vector3::new(7.0, 9.0, 5.0); // ahead and -Y
        let det_right = detect_opponent(&cam, &arena, &self_pos, &rot, &opp_right, 0.05, 10.0);
        assert!(det_right.detected);
        assert!(
            det_right.pixel_center[0] > cam.cx as f32,
            "right-side object should project to u > cx, got u={}",
            det_right.pixel_center[0]
        );
    }

    #[test]
    fn test_bbox_shrinks_with_distance() {
        let cam = CameraParams::default_fpv();
        let arena = Arena::new(Vector3::new(20.0, 20.0, 10.0));
        let self_pos = Vector3::new(0.0, 10.0, 5.0);
        let rot = identity_rotation();

        let near = detect_opponent(
            &cam,
            &arena,
            &self_pos,
            &rot,
            &Vector3::new(1.0, 10.0, 5.0),
            0.05,
            5.0,
        );
        let far = detect_opponent(
            &cam,
            &arena,
            &self_pos,
            &rot,
            &Vector3::new(3.0, 10.0, 5.0),
            0.05,
            5.0,
        );

        assert!(near.bbox[2] > far.bbox[2]); // width
        assert!(near.bbox[3] > far.bbox[3]); // height
    }
}
