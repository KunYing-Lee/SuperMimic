"""Exact float32 transition replay, with independently selectable storage and learner devices."""

from __future__ import annotations

import torch

from .mdp import STATE_DIM, XI_DIM

FIELDS = {
    "s": (STATE_DIM, torch.float32),
    "s1": (STATE_DIM, torch.float32),
    "a": (29, torch.float32),
    "a_prev": (29, torch.float32),
    "xi": (XI_DIM, torch.float32),
    "r": ((), torch.float32),
    "f": ((), torch.long),
    "ep_step": ((), torch.int32),
    "done": ((), torch.bool),
    "terminated": ((), torch.bool),
}


class ReplayBuffer:
    def __init__(
        self,
        T: int,
        N: int,
        device: str,
        storage_device: str | None = None,
        num_joints: int = 29,
        depth_shape: tuple | None = None,
    ):
        if T < 1 or N < 1:
            raise ValueError("Replay dimensions must be positive")
        self.T, self.N = T, N
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device or device)
        self.fields = dict(FIELDS)
        for key in ("s", "s1"):
            self.fields[key] = (13 + 2 * num_joints, torch.float32)
        for key in ("a", "a_prev"):
            self.fields[key] = (num_joints, torch.float32)
        self.fields["xi"] = (num_joints + 4, torch.float32)
        if depth_shape is not None:
            for key in ("depth", "depth1"):
                self.fields[key] = (depth_shape, torch.float32)
        self.data = {}
        for k, (shape, dt) in self.fields.items():
            shp = (T, N) + ((shape,) if isinstance(shape, int) else shape)
            self.data[k] = torch.zeros(shp, dtype=dt, device=self.storage_device)
        self.ptr, self.size = 0, 0

    @property
    def nbytes(self) -> int:
        return sum(v.numel() * v.element_size() for v in self.data.values())

    def add(self, tr: dict):
        for k in self.fields:
            self.data[k][self.ptr].copy_(tr[k])
        self.ptr = (self.ptr + 1) % self.T
        self.size = min(self.size + 1, self.T)

    def state_dict(self) -> dict:
        return {
            "T": self.T,
            "N": self.N,
            "ptr": self.ptr,
            "size": self.size,
            "data": self.data,
        }

    def load_state_dict(self, state: dict):
        if (state["T"], state["N"]) != (self.T, self.N):
            raise ValueError(
                "Resume requires the original replay capacity and environment count"
            )
        if set(state["data"]) != set(self.fields):
            raise ValueError("Checkpoint replay schema does not match this version")
        for k in self.fields:
            self.data[k].copy_(state["data"][k])
        self.ptr, self.size = state["ptr"], state["size"]

    def _row(self, age: torch.Tensor) -> torch.Tensor:
        """Map chronological index (0 = oldest) to its storage row."""
        return (self.ptr - self.size + age) % self.T

    def _to_learner(self, out: dict) -> dict:
        return {k: v.to(self.device) for k, v in out.items()}

    def sample_windows(self, B: int, k: int):
        """Uniform k-step windows; later transitions stop at episode boundaries."""
        if not 1 <= k <= self.size:
            raise ValueError(f"Need at least {k} replay rows, have {self.size}")
        dev = self.storage_device
        t0 = torch.randint(0, self.size - k + 1, (B,), device=dev)
        env = torch.randint(0, self.N, (B,), device=dev)
        rows = self._row(t0.unsqueeze(-1) + torch.arange(k, device=dev))
        e = env.unsqueeze(-1).expand_as(rows)
        keys = ["s", "s1", "a", "xi", "done", "terminated", "f"]
        if "depth" in self.fields:
            keys.extend(("depth", "depth1"))
        out = self._to_learner({key: self.data[key][rows, e] for key in keys})
        d = out["done"].float()
        out["mask"] = torch.cumprod(
            torch.cat((torch.ones(B, 1, device=self.device), 1 - d[:, :-1]), -1), -1
        )
        return out

    def sample_starts(
        self,
        B: int,
        hist: int,
        clip_end_by_frame: torch.Tensor | None = None,
        horizon: int = 0,
    ):
        """Uniform valid start states; history is repeated only at episode starts."""
        if self.size == 0:
            raise ValueError("Cannot sample empty replay")
        dev = self.storage_device
        ages = torch.arange(self.size, device=dev)
        chronological_rows = self._row(ages)
        episode_steps = self.data["ep_step"][chronological_rows].long()
        valid = ages[:, None] >= torch.minimum(
            episode_steps, torch.full_like(episode_steps, hist - 1)
        )
        if clip_end_by_frame is not None:
            frames = self.data["f"][chronological_rows]
            valid &= frames + horizon <= clip_end_by_frame[frames]
        eligible = valid.flatten().nonzero().flatten()
        if eligible.numel() == 0:
            raise ValueError(
                "Replay contains no start state with complete observation history and imagination horizon"
            )
        flat = eligible[torch.randint(eligible.numel(), (B,), device=dev)]
        age, env = flat // self.N, flat % self.N
        row = self._row(age)
        ep_step = self.data["ep_step"][row, env].long()
        back = torch.arange(hist - 1, -1, -1, device=dev)
        back = torch.minimum(back.unsqueeze(0), ep_step.unsqueeze(-1))
        hrows = self._row(age.unsqueeze(-1) - back)
        henv = env.unsqueeze(-1).expand_as(hrows)
        keys = ("s", "a_prev", "xi", "f")
        start_keys = keys + (("depth",) if "depth" in self.fields else ())
        start = self._to_learner({k: self.data[k][row, env] for k in start_keys})
        history = self._to_learner({k: self.data[k][hrows, henv] for k in keys})
        return start, history
