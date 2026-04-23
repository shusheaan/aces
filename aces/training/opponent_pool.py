"""Elo-rated opponent checkpoint pool for self-play training."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


@dataclass
class PoolEntry:
    """A single opponent checkpoint in the pool."""

    id: str
    path: str
    elo: float = 1000.0
    metadata: dict = field(default_factory=dict)


class OpponentPool:
    """Maintains a pool of opponent checkpoints rated by Elo.

    Used during self-play training to sample diverse historical opponents
    instead of always playing against the current agent's copy.
    """

    def __init__(self, pool_dir: Path | str, max_size: int = 20):
        self._pool_dir = Path(pool_dir)
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size
        self._entries: list[PoolEntry] = []

    @property
    def size(self) -> int:
        """Return number of entries in the pool."""
        return len(self._entries)

    @property
    def entries(self) -> list[PoolEntry]:
        """Return list of pool entries."""
        return list(self._entries)

    def add(self, model: PPO, metadata: dict | None = None) -> str:
        """Save a model checkpoint and add it to the pool.

        If the pool exceeds *max_size*, the entry with the lowest Elo is
        evicted.  Returns the new entry's ID.
        """
        entry_id = uuid.uuid4().hex[:12]
        save_path = str(self._pool_dir / entry_id)
        model.save(save_path)

        entry = PoolEntry(
            id=entry_id,
            path=save_path,
            elo=1000.0,
            metadata=metadata if metadata is not None else {},
        )
        self._entries.append(entry)

        # Evict lowest-Elo entry when over capacity.
        if len(self._entries) > self._max_size:
            worst = min(self._entries, key=lambda e: e.elo)
            self._entries.remove(worst)
            # Clean up checkpoint file on disk.
            p = Path(worst.path + ".zip")
            if p.exists():
                p.unlink()

        return entry_id

    def sample(self, env=None) -> tuple:
        """Sample an opponent weighted by Elo (softmax over elo/400).

        Returns ``(policy, metadata)`` where *policy* is the loaded SB3
        policy object ready for ``policy.predict()``.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty pool")

        elos = np.array([e.elo for e in self._entries], dtype=np.float64)
        logits = elos / 400.0
        # Numerically stable softmax.
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum()

        idx = np.random.choice(len(self._entries), p=weights)
        entry = self._entries[idx]

        loaded_model = PPO.load(entry.path, env=env)
        return loaded_model.policy, entry.metadata

    def update_elo(self, agent_won: bool, opponent_id: str) -> None:
        """Standard Elo update with K=32.

        When the agent wins the opponent's Elo decreases (and vice-versa).
        We treat the *agent* as a fixed 1000-Elo reference so only the
        opponent's rating moves.
        """
        entry = self._get_entry(opponent_id)
        K = 32
        agent_elo = 1000.0

        expected_opponent = 1.0 / (1.0 + 10.0 ** ((agent_elo - entry.elo) / 400.0))

        # Opponent's actual score: 0 if agent won, 1 if agent lost.
        actual_opponent = 0.0 if agent_won else 1.0
        entry.elo += K * (actual_opponent - expected_opponent)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Serialize pool state for checkpointing."""
        return {
            "pool_dir": str(self._pool_dir),
            "max_size": self._max_size,
            "entries": [
                {
                    "id": e.id,
                    "path": e.path,
                    "elo": e.elo,
                    "metadata": e.metadata,
                }
                for e in self._entries
            ],
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore pool from a previously saved *state_dict*."""
        self._pool_dir = Path(state["pool_dir"])
        self._max_size = state["max_size"]
        self._entries = [
            PoolEntry(
                id=e["id"],
                path=e["path"],
                elo=e["elo"],
                metadata=e["metadata"],
            )
            for e in state["entries"]
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_entry(self, entry_id: str) -> PoolEntry:
        for e in self._entries:
            if e.id == entry_id:
                return e
        raise KeyError(f"No pool entry with id={entry_id!r}")
