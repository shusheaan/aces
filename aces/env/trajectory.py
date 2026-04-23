"""Trajectory generators for curriculum training Stage 2.

Provides parametric flight paths for opponent drones in ACES dogfight simulation.
The arena is 10m x 10m x 3m; trajectories respect a 1.5m wall margin.
"""

from __future__ import annotations

import math

import numpy as np


class Trajectory:
    @staticmethod
    def circle(center, radius, altitude, speed, t) -> np.ndarray:
        """Horizontal circle at fixed altitude. center=[x,y], speed=rad/s."""
        angle = speed * t
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        return np.array([x, y, altitude], dtype=np.float64)

    @staticmethod
    def lemniscate(center, scale, altitude, speed, t) -> np.ndarray:
        """Figure-8 at fixed altitude. Parametric lemniscate of Bernoulli."""
        angle = speed * t
        denom = 1.0 + math.sin(angle) ** 2
        x = center[0] + scale * math.cos(angle) / denom
        y = center[1] + scale * math.sin(angle) * math.cos(angle) / denom
        return np.array([x, y, altitude], dtype=np.float64)

    @staticmethod
    def patrol(waypoints, speed, t) -> np.ndarray:
        """Linear interpolation between waypoints at constant speed, looping.

        waypoints: list/array of 3D points [[x,y,z], ...]
        speed: m/s
        t: time in seconds
        """
        pts = [np.array(w, dtype=np.float64) for w in waypoints]
        n = len(pts)

        # Compute segment lengths
        seg_lengths = [
            float(np.linalg.norm(pts[(i + 1) % n] - pts[i])) for i in range(n)
        ]
        total_length = sum(seg_lengths)

        if total_length == 0.0:
            return pts[0].copy()

        # Distance traveled (mod total path length for looping)
        dist = (speed * t) % total_length

        # Find which segment we are on
        accumulated = 0.0
        for i in range(n):
            seg_len = seg_lengths[i]
            if dist <= accumulated + seg_len:
                frac = (dist - accumulated) / seg_len if seg_len > 0.0 else 0.0
                return pts[i] + frac * (pts[(i + 1) % n] - pts[i])
            accumulated += seg_len

        # Floating-point edge case: return last waypoint start
        return pts[-1].copy()

    @staticmethod
    def random_trajectory(bounds, rng) -> tuple[str, dict]:
        """Pick random trajectory type with arena-safe parameters.

        bounds=[bx, by, bz]. margin=1.5m from walls.
        Returns (type_name, kwargs) so caller can do:
            Trajectory.<type_name>(t=t, **kwargs)
        """
        bx, by, bz = float(bounds[0]), float(bounds[1]), float(bounds[2])
        margin = 1.5

        traj_type = rng.choice(["circle", "lemniscate", "patrol"])

        # Safe center range after margin
        cx = rng.uniform(margin, bx - margin)
        cy = rng.uniform(margin, by - margin)
        altitude = rng.uniform(margin, bz - margin)

        if traj_type == "circle":
            # Radius must keep the circle inside bounds with margin on all sides
            max_radius = min(
                cx - margin, bx - margin - cx, cy - margin, by - margin - cy
            )
            max_radius = max(max_radius, 0.5)  # enforce a minimum sensible radius
            radius = rng.uniform(0.5, max_radius)
            speed = rng.uniform(0.3, 1.0)  # rad/s
            return "circle", {
                "center": [cx, cy],
                "radius": radius,
                "altitude": altitude,
                "speed": speed,
            }

        if traj_type == "lemniscate":
            # scale: half-width of figure-8; must fit within margin on both axes
            max_scale = min(
                cx - margin, bx - margin - cx, cy - margin, by - margin - cy
            )
            max_scale = max(max_scale, 0.5)
            scale = rng.uniform(0.5, max_scale)
            speed = rng.uniform(0.3, 1.0)  # rad/s
            return "lemniscate", {
                "center": [cx, cy],
                "scale": scale,
                "altitude": altitude,
                "speed": speed,
            }

        # patrol: random waypoints within safe region
        n_waypoints = int(rng.integers(3, 6))
        waypoints = [
            [
                rng.uniform(margin, bx - margin),
                rng.uniform(margin, by - margin),
                altitude,
            ]
            for _ in range(n_waypoints)
        ]
        speed = rng.uniform(1.0, 3.0)  # m/s
        return "patrol", {"waypoints": waypoints, "speed": speed}
