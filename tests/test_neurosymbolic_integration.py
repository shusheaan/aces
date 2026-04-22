"""Integration test: God Oracle → PerceptionNet → FSM → verify outputs."""

import numpy as np
import torch

from aces.fsm import DroneMode, SymbolicFSM
from aces.god_oracle import GodOracle
from aces.perception import PerceptionNet


def test_full_pipeline_oracle_to_fsm() -> None:
    """Run oracle → FSM on a high-threat scenario."""
    oracle = GodOracle()
    fsm = SymbolicFSM(hysteresis_ticks=0)

    labels = oracle.compute(
        lock_b_progress=0.8,
        distance=1.5,
        opponent_facing_me=0.9,
        lock_a_progress=0.0,
        a_sees_b=True,
        i_face_opponent=0.0,
        nearest_obs_dist=3.0,
        speed=1.0,
        belief_var=0.0,
        opponent_closing_speed=1.0,
    )
    assert labels["threat"] > 0.7

    out = fsm.step(labels)
    assert out.mode == DroneMode.EVADE


def test_perception_net_processes_observation() -> None:
    """PerceptionNet forward pass produces valid features for FSM."""
    net = PerceptionNet()
    net.eval()

    obs = np.zeros(21, dtype=np.float32)
    obs_t = torch.as_tensor(obs).unsqueeze(0)
    with torch.no_grad():
        continuous, intent_logits = net(obs_t)

    features = {
        "threat": continuous[0, 0].item(),
        "opportunity": continuous[0, 1].item(),
        "collision_risk": continuous[0, 2].item(),
        "uncertainty": continuous[0, 3].item(),
        "opponent_distance": continuous[0, 4].item(),
        "opponent_intent": int(intent_logits[0].argmax().item()),
    }

    assert 0 <= features["threat"] <= 1
    assert 0 <= features["opportunity"] <= 1
    assert 0 <= features["collision_risk"] <= 1
    assert 0 <= features["uncertainty"] <= 1
    assert features["opponent_distance"] >= 0
    assert features["opponent_intent"] in (0, 1, 2)

    fsm = SymbolicFSM()
    out = fsm.step(features)
    assert out.mode in list(DroneMode)
