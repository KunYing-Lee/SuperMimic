"""World model, policy (motion encoder + shared decoder), and critic (paper Table A.4)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .mdp import NUM_J, XI_DIM, invariant_features
from .rotations import matmul3, matvec3, so3_exp


def mlp(
    inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False
) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self.count >= self.max_count:
            return
        n = x.shape[0]
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (
            self.var * self.count + v * n + delta**2 * self.count * n / tot
        ) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """FP32 tensors with TF32 CUDA matmuls, matching the paper's precision setting."""
    return net(x)


class WorldModel(nn.Module):
    """Deterministic dynamics s_{t+1} = f(s_t, a_t, xi) predicting frame-local state increments."""

    def __init__(
        self,
        hidden=(2048, 2048, 2048),
        goal_dim: int = 730,
        num_joints: int = NUM_J,
        xi_dim: int = XI_DIM,
        terrain_dim: int = 0,
    ):
        super().__init__()
        self.num_joints, self.terrain_dim = num_joints, terrain_dim
        inv_dim, out_dim = 10 + 2 * num_joints, 12 + 2 * num_joints
        self.in_dim = inv_dim + num_joints + xi_dim + terrain_dim
        self.net = mlp(self.in_dim, list(hidden), out_dim, act=nn.SiLU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(out_dim, eps=1e-4)
        # The paper permits learned heads but does not specify their architecture.
        # Use complete invariant state and reference features rather than a hand-tuned
        # approximation to the simulator's hard termination rule.
        self.termination_net = mlp(2 * inv_dim + goal_dim, [256, 128], 1)

    def termination_logits(self, errors: torch.Tensor) -> torch.Tensor:
        return self.termination_net(errors).squeeze(-1)

    def termination_probability(self, errors: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.termination_logits(errors))

    def inputs(self, S, a, xi_n, terrain=None):
        features = (invariant_features(S), a, xi_n)
        if self.terrain_dim:
            if terrain is None or terrain.shape[-1] != self.terrain_dim:
                raise ValueError(
                    "terrain-conditioned dynamics require a matching height scan"
                )
            features += (terrain,)
        elif terrain is not None:
            raise ValueError(
                "this dynamics model was constructed without terrain inputs"
            )
        return torch.cat(features, -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                matvec3(Rt, p1 - p),
                so3_log(matmul3(Rt, R1)),
                matvec3(Rt, v1 - v),
                matvec3(Rt, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n, terrain=None):
        p, R, v, w, q, qd = S
        y = run_mlp(self.net, self.in_norm(self.inputs(S, a, xi_n, terrain)))
        d = y * self.out_norm.std() + self.out_norm.mean
        dp, dth, dv, dw, dq, dqd = d.split(
            (3, 3, 3, 3, self.num_joints, self.num_joints), -1
        )
        p1 = p + matvec3(R, dp)
        R1 = matmul3(R, so3_exp(dth))
        v1 = v + matvec3(R, dv)
        w1 = w + matvec3(R, dw)
        return (p1, R1, v1, w1, q + dq, qd + dqd)

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        from .rotations import so3_log

        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                matvec3(Rt, ph - p),
                so3_log(matmul3(Rt, Rh)),
                matvec3(Rt, vh - v),
                matvec3(Rt, wh - w),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class Policy(nn.Module):
    """a = dec(enc(g), o_{t-9:t}); deterministic, latent z in R^32 bounded by tanh."""

    def __init__(
        self,
        obs_dim: int,
        goal_dim: int,
        hist: int = 10,
        z_dim: int = 32,
        num_joints: int = NUM_J,
        depth_shape: tuple[int, int] | None = None,
    ):
        super().__init__()
        self.hist = hist
        self.enc = mlp(goal_dim, [2048, 1024, 512, 256], z_dim)
        self.depth_shape = depth_shape
        if depth_shape is not None:
            if tuple(depth_shape) != (18, 32):
                raise ValueError("the depth encoder expects 18 x 32 images")
            self.depth_encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, 2, 1),
                nn.ELU(),
                nn.Conv2d(16, 32, 3, 2, 1),
                nn.ELU(),
                nn.Flatten(),
                nn.Linear(32 * 5 * 8, 128),
                nn.ELU(),
            )
        self.dec = mlp(
            z_dim + obs_dim * hist + (128 if depth_shape else 0),
            [2048, 1024, 512, 256],
            num_joints,
        )
        with torch.no_grad():
            self.dec[-1].weight.mul_(0.01)
            self.dec[-1].bias.zero_()
        self.obs_norm = RunningNorm(obs_dim)
        self.goal_norm = RunningNorm(goal_dim)

    def forward(
        self, obs_hist: torch.Tensor, goal: torch.Tensor, depth=None
    ) -> torch.Tensor:
        """obs_hist: (B, hist, obs_dim) noisy observations, newest last."""
        z = torch.tanh(run_mlp(self.enc, self.goal_norm(goal)))
        o = self.obs_norm(obs_hist).flatten(-2)
        features = (z, o)
        if self.depth_shape is not None:
            if depth is None or tuple(depth.shape[-3:]) != (1, *self.depth_shape):
                raise ValueError(
                    "the visual policy requires B x 1 x 18 x 32 depth images"
                )
            features += (self.depth_encoder(depth),)
        elif depth is not None:
            raise ValueError("this policy was constructed without a depth encoder")
        return run_mlp(self.dec, torch.cat(features, -1))


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


class DepthHead(nn.Module):
    """Anchor-conditioned depth observation model, following InstinctMJ.

    Predicts a clean normalized image, not dynamics. Train against real sensor
    targets, with detached predicted states for autoregressive supervision.
    Architecture reference: project-instinct/InstinctMJ, supermimic branch at
    c9d3e00c8f4c16e394ddb3c2d545bac8b3e8819b (CC BY-NC 4.0).
    """

    def __init__(self, num_joints: int, terrain_dim: int = 187):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 8, 256),
            nn.ELU(),
        )
        self.features = mlp(9 + num_joints + 6 + terrain_dim, [256, 256], 256)
        self.fc = nn.Sequential(nn.Linear(512, 64 * 5 * 8), nn.ELU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=(0, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=(1, 1)),
            nn.ELU(),
        )
        self.output = nn.Conv2d(33, 1, 3, 1, 1)
        self.register_buffer(
            "context_scale", torch.tensor([50.0] * 3 + [5.0] * 3), persistent=False
        )

    def forward(self, anchor, pose, q, context, scan):
        """Inputs Bx1x18x32, Bx9, BxJ, Bx6, Bx187; output Bx1x18x32."""
        conditioning = torch.cat(
            (pose, q, context * self.context_scale, scan / 5.0), -1
        )
        latent = torch.cat((self.encoder(anchor), self.features(conditioning)), -1)
        decoded = self.decoder(self.fc(latent).reshape(-1, 64, 5, 8))
        return torch.sigmoid(self.output(torch.cat((decoded, anchor), 1)))


class DepthEnsemble(nn.Module):
    """Bootstrap depth heads, with one fixed member per imagined trajectory."""

    def __init__(
        self, num_joints: int = NUM_J, terrain_dim: int = 187, ensemble_size: int = 3
    ):
        super().__init__()
        if ensemble_size < 1:
            raise ValueError("ensemble_size must be positive")
        self.heads = nn.ModuleList(
            [DepthHead(num_joints, terrain_dim) for _ in range(ensemble_size)]
        )

    def forward_all(self, *inputs):
        return torch.stack([head(*inputs) for head in self.heads])

    def groups(self, members: torch.Tensor) -> list[torch.Tensor]:
        """Group a fixed member assignment once, outside the rollout loop."""
        if members.ndim != 1 or members.dtype != torch.long:
            raise ValueError("member assignments must be a one-dimensional long tensor")
        if (
            members.numel() == 0
            or (members < 0).any()
            or (members >= len(self.heads)).any()
        ):
            raise ValueError("member assignments must index this ensemble")
        counts = torch.bincount(members, minlength=len(self.heads)).tolist()
        return list(members.argsort().split(counts))

    def forward(self, members, *inputs):
        groups = self.groups(members) if isinstance(members, torch.Tensor) else members
        images, indices = [], []
        for head, ids in zip(self.heads, groups, strict=True):
            if ids.numel():
                images.append(head(*(value[ids] for value in inputs)))
                indices.append(ids)
        order = torch.cat(indices)
        return torch.cat(images)[order.argsort()]
