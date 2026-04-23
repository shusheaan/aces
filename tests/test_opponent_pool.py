"""Tests for the Elo-rated opponent pool."""

from aces.env import DroneDogfightEnv
from aces.training import OpponentPool
from stable_baselines3 import PPO


def _make_model():
    """Create a tiny PPO model suitable for pool tests."""
    env = DroneDogfightEnv(max_episode_steps=50, opponent="random")
    return PPO("MlpPolicy", env, n_steps=128, batch_size=64, verbose=0)


def test_pool_creation(tmp_path):
    """A freshly created pool has size 0."""
    pool = OpponentPool(pool_dir=tmp_path / "pool")
    assert pool.size == 0
    assert pool.entries == []


def test_pool_add_and_sample(tmp_path):
    """Adding a model and sampling should return a policy and metadata."""
    pool = OpponentPool(pool_dir=tmp_path / "pool")
    model = _make_model()
    entry_id = pool.add(model, metadata={"step": 100})

    assert pool.size == 1
    assert isinstance(entry_id, str)

    policy, meta = pool.sample()
    assert meta == {"step": 100}
    # The policy should have a predict method (SB3 BasePolicy).
    assert hasattr(policy, "predict")


def test_pool_max_size(tmp_path):
    """Adding beyond max_size should evict the lowest-Elo entry."""
    pool = OpponentPool(pool_dir=tmp_path / "pool", max_size=3)
    model = _make_model()

    ids = []
    for i in range(4):
        eid = pool.add(model, metadata={"idx": i})
        ids.append(eid)

    assert pool.size == 3

    # The first entry (idx=0) has default Elo 1000, same as all others.
    # With equal Elo the first added one (earliest in the list at the time
    # of eviction) will be evicted by min() since it appears first.
    remaining_ids = {e.id for e in pool.entries}
    # The evicted one should not be in the pool any more.
    assert len(remaining_ids) == 3


def test_pool_max_size_evicts_lowest_elo(tmp_path):
    """When Elos differ, the lowest-Elo entry is evicted."""
    pool = OpponentPool(pool_dir=tmp_path / "pool", max_size=3)
    model = _make_model()

    pool.add(model)
    id1 = pool.add(model)
    pool.add(model)

    # Lower id1's Elo so it becomes the weakest.
    pool._get_entry(id1).elo = 800.0

    # Adding a 4th should evict id1.
    pool.add(model)
    remaining_ids = {e.id for e in pool.entries}
    assert id1 not in remaining_ids
    assert pool.size == 3


def test_pool_elo_update(tmp_path):
    """Winning agent makes opponent's Elo decrease."""
    pool = OpponentPool(pool_dir=tmp_path / "pool")
    model = _make_model()
    entry_id = pool.add(model)

    initial_elo = pool.entries[0].elo
    assert initial_elo == 1000.0

    pool.update_elo(agent_won=True, opponent_id=entry_id)
    assert pool.entries[0].elo < initial_elo

    # Conversely, if the agent loses, the opponent's Elo increases.
    elo_after_loss = pool.entries[0].elo
    pool.update_elo(agent_won=False, opponent_id=entry_id)
    assert pool.entries[0].elo > elo_after_loss


def test_pool_state_dict_roundtrip(tmp_path):
    """state_dict -> load_state_dict preserves entries and Elo."""
    pool = OpponentPool(pool_dir=tmp_path / "pool", max_size=10)
    model = _make_model()

    id0 = pool.add(model, metadata={"tag": "alpha"})
    pool.add(model, metadata={"tag": "beta"})

    # Modify Elo so we can verify it round-trips.
    pool.update_elo(agent_won=True, opponent_id=id0)

    state = pool.state_dict()

    # Reconstruct into a fresh pool instance.
    pool2 = OpponentPool(pool_dir=tmp_path / "pool2")
    pool2.load_state_dict(state)

    assert pool2.size == 2
    assert pool2._max_size == 10

    ids_orig = {e.id: e for e in pool.entries}
    ids_restored = {e.id: e for e in pool2.entries}

    for eid in ids_orig:
        assert eid in ids_restored
        assert ids_orig[eid].elo == ids_restored[eid].elo
        assert ids_orig[eid].metadata == ids_restored[eid].metadata
        assert ids_orig[eid].path == ids_restored[eid].path
