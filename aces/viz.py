"""Real-time 3D visualization for ACES using Rerun."""

from __future__ import annotations

import numpy as np
import rerun as rr

from aces.config import ArenaConfig, load_configs


class AcesVisualizer:
    """Logs simulation state to Rerun for 3D viewing."""

    def __init__(
        self,
        config_dir: str | None = None,
        recording_id: str = "aces",
        max_trail: int = 500,
        spawn: bool = True,
    ):
        self.max_trail = max_trail
        self.trail_a: list[list[float]] = []
        self.trail_b: list[list[float]] = []

        rr.init(recording_id, spawn=spawn)

        # Load arena config and log static geometry
        cfg = load_configs(config_dir)
        self._log_arena_config(cfg.arena)

    def _log_arena_config(self, arena: ArenaConfig):
        """Log static arena geometry."""
        bx, by, bz = arena.bounds
        half = [bx / 2, by / 2, bz / 2]
        center = half

        rr.log(
            "world/arena",
            rr.Boxes3D(centers=[center], half_sizes=[half], colors=[[80, 80, 80, 60]]),
            static=True,
        )

        rr.log(
            "world/ground",
            rr.Boxes3D(
                centers=[[half[0], half[1], -0.01]],
                half_sizes=[[half[0], half[1], 0.01]],
                colors=[[40, 100, 40, 100]],
            ),
            static=True,
        )

        for i, (obs_center, obs_half) in enumerate(arena.obstacles):
            rr.log(
                f"world/obstacles/{i}",
                rr.Boxes3D(
                    centers=[obs_center],
                    half_sizes=[obs_half],
                    colors=[[200, 60, 60, 180]],
                ),
                static=True,
            )

    def log_step(self, step: int, result, sim=None):
        """Log one simulation step to Rerun.

        Args:
            step: Simulation step number
            result: StepResult from Simulation.step()
            sim: Optional Simulation reference for particle visualization
        """
        rr.set_time("step", sequence=step)

        pos_a = list(result.drone_a_state[:3])
        pos_b = list(result.drone_b_state[:3])
        fwd_a = list(result.drone_a_forward)
        fwd_b = list(result.drone_b_forward)

        rr.log(
            "world/drone_a", rr.Points3D([pos_a], radii=[0.05], colors=[[0, 255, 255]])
        )
        rr.log(
            "world/drone_b", rr.Points3D([pos_b], radii=[0.05], colors=[[255, 165, 0]])
        )

        arrow_scale = 0.3
        rr.log(
            "world/drone_a/heading",
            rr.Arrows3D(
                origins=[pos_a],
                vectors=[[f * arrow_scale for f in fwd_a]],
                colors=[[0, 255, 255]],
            ),
        )
        rr.log(
            "world/drone_b/heading",
            rr.Arrows3D(
                origins=[pos_b],
                vectors=[[f * arrow_scale for f in fwd_b]],
                colors=[[255, 165, 0]],
            ),
        )

        self.trail_a.append(pos_a)
        self.trail_b.append(pos_b)
        if len(self.trail_a) > self.max_trail:
            self.trail_a = self.trail_a[-self.max_trail :]
        if len(self.trail_b) > self.max_trail:
            self.trail_b = self.trail_b[-self.max_trail :]

        if len(self.trail_a) >= 2:
            rr.log(
                "world/drone_a/trail",
                rr.LineStrips3D([self.trail_a], colors=[[0, 200, 200, 120]]),
            )
        if len(self.trail_b) >= 2:
            rr.log(
                "world/drone_b/trail",
                rr.LineStrips3D([self.trail_b], colors=[[200, 130, 0, 120]]),
            )

        rr.log("metrics/lock_a_progress", rr.Scalars(result.lock_a_progress))
        rr.log("metrics/lock_b_progress", rr.Scalars(result.lock_b_progress))
        rr.log("metrics/distance", rr.Scalars(result.distance))

        # Visibility and belief state
        if hasattr(result, "a_sees_b"):
            rr.log("metrics/a_sees_b", rr.Scalars(1.0 if result.a_sees_b else 0.0))
            rr.log("metrics/belief_b_var", rr.Scalars(result.belief_b_var_from_a))
            rr.log("metrics/time_since_a_saw_b", rr.Scalars(result.time_since_a_saw_b))

            # Belief position marker
            belief_pos = list(result.belief_b_pos_from_a)
            color = [0, 255, 0, 150] if result.a_sees_b else [255, 0, 0, 150]
            rr.log(
                "world/belief_b",
                rr.Points3D([belief_pos], radii=[0.04], colors=[color]),
            )

        # Particle cloud visualization
        if sim is not None and hasattr(sim, "belief_particles_a"):
            particles = sim.belief_particles_a()
            if particles:
                rr.log(
                    "world/particles_a",
                    rr.Points3D(particles, radii=[0.015], colors=[[255, 255, 0, 80]]),
                )

    def log_camera(self, step: int, result, drone: str = "a"):
        """Log depth image and detection overlay for a drone's camera.

        Args:
            step: Simulation step number.
            result: StepResult from Simulation.step().
            drone: "a" or "b".
        """
        rr.set_time("step", sequence=step)

        rendered = getattr(result, f"camera_rendered_{drone}", False)
        depth_data = getattr(result, f"depth_image_{drone}", None)

        if not rendered or depth_data is None:
            return

        depth_arr = np.array(depth_data, dtype=np.float32)
        # Determine resolution from data length (assume 320x240 default)
        total = len(depth_arr)
        if total == 320 * 240:
            h, w = 240, 320
        elif total == 80 * 60:
            h, w = 60, 80
        else:
            return

        depth_img = depth_arr.reshape(h, w)
        rr.log(
            f"camera_{drone}/depth",
            rr.DepthImage(depth_img, meter=1.0),
        )

        # Detection overlay
        detected = getattr(result, f"det_{drone}_detected", False)
        if detected:
            bbox = getattr(result, f"det_{drone}_bbox")
            conf = getattr(result, f"det_{drone}_confidence")
            rr.log(
                f"camera_{drone}/detection",
                rr.Boxes2D(
                    array=[bbox],
                    array_format=rr.Box2DFormat.XYWH,
                    labels=[f"opp {conf:.2f}"],
                    colors=[[255, 0, 0]],
                ),
            )
        else:
            rr.log(f"camera_{drone}/detection", rr.Clear(recursive=False))

    def reset(self):
        """Clear trails for a new episode."""
        self.trail_a.clear()
        self.trail_b.clear()
