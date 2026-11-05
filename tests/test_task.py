"""Task boundaries and strict provenance without a CUDA simulator."""

import argparse
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from supermimic.buffer import ReplayBuffer
from supermimic.mdp import MDP
from supermimic.reproducibility import SCHEMA_VERSION, manifest, sha256, validate_resume
from supermimic.task import LafanTask
from supermimic.train import _imag_step, main, make_parser


def test_task_arguments_and_default_sampling():
    class ExternalTask(LafanTask):
        name = "external"
        fine_tune_args = frozenset({"threshold"})

        def add_arguments(self, parser):
            parser.add_argument("--threshold", type=float, default=0.2)
            parser.set_defaults(activation_checkpoint=False)

    args = make_parser(LafanTask()).parse_args([])
    assert (args.task, args.num_envs, args.utd, args.horizon, args.activation_checkpoint) == (
        "lafan", 4096, 0.4, 32, True)
    assert Path(args.motion_dir).name == "lafan1_g1"
    args = make_parser(ExternalTask()).parse_args(["--threshold", "0.3", "--activation_checkpoint"])
    assert (args.task, args.threshold, args.activation_checkpoint) == ("external", 0.3, True)
    with pytest.raises(SystemExit) as result:
        main(["--help"], task=ExternalTask())
    assert result.value.code == 0

    replay = ReplayBuffer(4, 1, "cpu")
    replay.size = 4
    replay.data["f"][:, 0] = torch.arange(4)
    replay.data["ep_step"][:, 0] = torch.arange(4)
    starts, history = LafanTask().sample_starts(replay, 10, 2, torch.full((4,), 3), 2)
    assert (starts["f"] <= 1).all()
    torch.testing.assert_close(history["f"][:, -1], starts["f"])


@pytest.fixture
def provenance(tmp_path):
    core = tmp_path / "core"
    (core / "supermimic").mkdir(parents=True)
    (core / "supermimic" / "train.py").write_text("core = 1\n")
    external = tmp_path / "external"
    external.mkdir()
    source = external / "task.py"
    source.write_text("task = 1\n")
    asset = external / "scene.json"
    asset.write_text('{"scene": 1}')
    robot = core / "robot" / "g1.xml"
    robot.parent.mkdir()
    robot.write_text("<mujoco/>")
    motion = SimpleNamespace(files=[], names=["fixture"], num_frames=10)
    args = argparse.Namespace(device="cuda:0", task="external", threshold=0.2, lr=0.001)

    def record():
        with patch("torch.cuda.get_device_name", return_value="fixture"), patch(
            "torch.cuda.get_device_properties", return_value=SimpleNamespace(total_memory=1)
        ):
            return manifest(str(core), args, motion, robot, [source], [asset])

    return record, core / "supermimic" / "train.py", source, asset


def test_manifest_hashes_core_and_task_without_git(provenance):
    record, core, source, asset = provenance
    current = record()
    assert current["schema_version"] == SCHEMA_VERSION == 2
    assert current["git_commit"] is None and current["git_dirty"] is None
    assert current["source_sha256"] == {"supermimic/train.py": sha256(core), str(source): sha256(source)}
    assert current["asset_sha256"][str(asset)] == sha256(asset)
    validate_resume({"schema_version": 2, "manifest": current}, record())


@pytest.mark.parametrize("index, error", [(1, "source_sha256"), (2, "source_sha256"), (3, "asset hashes")])
def test_resume_rejects_changed_core_or_task_files(provenance, index, error):
    record = provenance[0]
    saved = {"schema_version": 2, "manifest": record()}
    provenance[index].write_text("changed")
    with pytest.raises(ValueError, match=error):
        validate_resume(saved, record())
    if index == 3:
        with pytest.raises(ValueError, match=error):
            validate_resume(saved, record(), allow_source=True)


def test_fine_tune_only_allows_declared_arguments_and_source(provenance):
    current = provenance[0]()
    saved = {"schema_version": 2, "manifest": copy.deepcopy(current)}
    current["args"]["threshold"] = 0.3
    current["source_sha256"]["supermimic/train.py"] = "changed"
    with pytest.raises(ValueError, match="configuration"):
        validate_resume(saved, current, allow_source=True)
    validate_resume(saved, current, allow_args={"threshold"}, allow_source=True)
    current["args"]["lr"] = 0.002
    with pytest.raises(ValueError, match="lr"):
        validate_resume(saved, current, allow_args={"threshold"}, allow_source=True)
    with pytest.raises(ValueError, match="complete restart state"):
        validate_resume({"schema_version": 1}, current)


@pytest.mark.parametrize("field", ["packages", "cuda"])
def test_fine_tune_still_checks_runtime(provenance, field):
    current = provenance[0]()
    saved = {"schema_version": 2, "manifest": copy.deepcopy(current)}
    current[field] = "changed"
    with pytest.raises(ValueError, match=field):
        validate_resume(saved, current, allow_source=True)


def test_imagination_uses_task_derivatives_and_carries_aux():
    class AnalyticMDP(MDP):
        analytic_termination = True

        def __init__(self):
            pass

        def obs(self, S, a_prev, xi, f):
            return S[0]

        def add_noise(self, o):
            return o

        def goal(self, f, K, S):
            return S[0]

        def reward(self, S, f, a, a_prev, F=()):
            return S[0].sum(-1)

        def terminate(self, S, f, aux, F=()):
            return aux + 1, S[0][:, 0] > 0

        def imagine_termination(self, S, f, aux, F=()):
            next_aux, hard = self.terminate(S, f, aux, F)
            probability = S[0][:, 0].sigmoid()
            return next_aux, hard.float() + probability - probability.detach()

    class Model:
        def step(self, S, a, xi, extra):
            return (S[0] + a,)

    action = torch.tensor([[0.2]], requires_grad=True)
    S = (torch.zeros(1, 1),)
    mdp = AnalyticMDP()
    aux = torch.zeros(1, 1)
    frame = torch.zeros(1, dtype=torch.long)
    result = _imag_step(S, torch.zeros(1, 2, 1), torch.zeros_like(action), frame,
                        torch.zeros(1, 33), torch.zeros(1, 33), S[0], aux, (),
                        pi=lambda history, goal: action, wm=Model(), mdp=mdp, K=1)
    torch.testing.assert_close(result[6], torch.ones(1))
    torch.testing.assert_close(result[7], torch.ones(1, 1))
    result[6].sum().backward()
    probability = action.detach().sigmoid()
    torch.testing.assert_close(action.grad, probability * (1 - probability))
    assert mdp.terminate(result[0], result[3], aux)[1].dtype == torch.bool
    assert MDP.imagine_termination(mdp, result[0], result[3], aux)[1].dtype == torch.bool
