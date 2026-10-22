# Paper alignment and implementation audit

Audited on 2026-10-22 against [SuperMimic, arXiv:2610.28480v1](https://arxiv.org/abs/2610.28480v1), submitted 2026-10-20, and its [primary PDF](https://arxiv.org/pdf/2610.28480). The identifier is valid. The [authors' project page](https://supermimic.github.io/) is a supplementary source for benchmark and compute-accounting interpretation.

The imported implementation is `supermimic_vibed` at `a1508d42c9d1699a52e8233ab16d08c488441ab9`. Its historical reported return and timing describe its Isaac Lab run; they are not results of this mjlab port.

## Scope and evidentiary limits

This repository targets the LAFAN1 controlled experiment: 40 G1-retargeted clips, approximately 2.45 hours (Figure 4). It does not reproduce the paper's headline BONES-SEED experiment or the velocity-command interface. A complete migration of this repository's tracker to mjlab is a different claim from reproduction of every experiment in the paper.

The supplied [shared discussion](https://chatgpt.com/share/6ad9fe39-cac8-83ec-a71e-6de6f877a91c) was reachable by title, but its conversation body was not retrievable during this audit. Its contents have not been treated as evidence.

The paper does not fully specify the dynamics state coordinates, per-frame goal features, activations, normalization, termination-head architecture, reset perturbations, or running-return estimator. These require disclosed implementation choices. No available official implementation was identified on the authors' project page. Hyperparameter agreement alone cannot establish numerical equivalence.

## Numerical target contract

The following numerical settings are transcribed as factual configuration targets from Appendix A, Tables A.4/A.5, and Figure 5. They are not a claim that all checks have passed.

| Component | Target |
| --- | --- |
| Physics | MuJoCo Warp; 0.02 s control; four 0.005 s physics steps |
| Observation | 93 values: joint position 29, joint velocity 29, base angular velocity 3, gravity 3, previous action 29; history 10 |
| Uniform observation-noise half-widths | Position 0.01; velocity 0.5; angular velocity 0.2; gravity 0.05; previous action 0 |
| Goal | 10 future reference frames |
| World-model hidden widths | 2048, 2048, 2048 |
| Encoder and decoder hidden widths | 2048, 1024, 512, 256 each |
| Latent | 32; tanh bounded |
| Critic hidden widths | 256, 128 |
| Replay | 4096 environments; 1000 transitions/environment |
| Dynamics update | 10 batches of 2048 windows; window 8 |
| Policy update | 4096 imagined rollouts; horizon 32; discount 0.99 |
| Critic update | Lambda 0.95; 5 epochs; 4 shuffled minibatches/epoch |
| Optimizer | Adam; policy betas 0.9/0.95 |
| One-GPU learning rate | 0.0002 for all optimizers |
| Clipping | Global gradient norm 1; policy additionally elementwise 0.1 |
| Exploration | Independent Gaussian action noise, standard deviation 0.1 |
| Schedule | Update/data ratio 0.4; one policy update/world-model update; 320000 iterations |
| Learning-rate schedule | Cosine to 0.03 times initial rate over 250000 iterations; constant thereafter |
| Matmul arithmetic | TF32 neural-network products; float32 multiply/add for 3D geometry; no bf16 autocast |
| Reward standard deviations | Body position 0.3, orientation 0.4, linear velocity 1, angular velocity 3.14; root position 0.3, orientation 0.4 |
| Reward weights | Four body terms 1 each; two root terms 0.5 each; action-rate penalty -0.1; joint-limit penalty -2; multiply all by 0.02 |
| Hard failure thresholds | Root height error >0.25 m or absolute projected-gravity z difference >0.8 |
| Randomized context | Friction [0.3,1.2]; joint calibration +/-0.01 rad; torso CoM x +/-0.025 m, y/z +/-0.05 m |
| Reset sampler | 1 s bins; weights 0.5 uniform, 0.4 current softmax, 0.1 historical softmax; temperature multiplier -25 |

## Findings in the imported implementation

1. `mdp.py` used 99 observation values by appending reference-relative root orientation and additional noise. Removing these six values changes the network input contract; old checkpoints are incompatible with the aligned actor.
2. `train.py` enabled bf16 by default. TF32 requires float32 tensors and appropriate matrix-multiply settings; calling the hardware "Blackwell" does not justify claiming equivalent numerical behavior.
3. `MDP.soft_termination` introduced two fixed sigmoid temperatures. These are tuning choices absent from the paper, and they directly change policy gradients. A learned termination head supervised by binary failure labels is an explicitly supported route in Equation 1. Such a head must receive reference information: the goal-free dynamics input alone cannot distinguish a valid low pose from a height-tracking failure.
4. The dynamics loss divided component errors by increment standard deviations with extra hard-coded floors. This changes relative loss weighting. Equation 1 supports a direct state error; any choice of state coordinates must still be documented. Output normalization can be a parameterization without becoming a loss weight.
5. Collection clipped actions to [-20,20], skipped policy updates for nonfinite gradient norms, and delayed them for 500 dynamics-only iterations. These are additional behavioral changes. The aligned path should fail visibly on nonfinite updates and use the documented joint update loop after the replay windows can be sampled.
6. The original simulator disabled self-collisions. Removing a self-collision reward penalty does not imply disabling self-collision physics. Collision configuration needs independent readback.
7. Friction and torso CoM were sampled only at startup; the published context is described per episode. Reset scheduling must be stated and verified.
8. The original reset-return update used indexed assignment with duplicate bin indices, making simultaneous updates ambiguous. A grouped reduction is required for a deterministic running estimate. Its timeout bootstrap used extrapolated mean reward; that extrapolation is an implementation heuristic, not a published rule.
9. A ten-second training timeout, reset perturbations, running-normalizer epsilon/count limits, and EMA rates were implementation choices. The port defaults to exact reference initialization and natural clip boundaries; reset perturbations and episode time limits are opt-in. Eight initial collection steps populate the minimum world-model window. Running normalization remains an explicit parameterization choice.

## Choices that remain implementation-specific

The imported representation uses local state increments, an SO(3) rotation error, and 73 features per future goal frame. Its differentiable forward kinematics and analytic reward/observation heads are reasonable constructions, but their exact equality to the authors' implementation is unknown. Learned termination-network dimensions also cannot be attributed to Table A.4, which lists only the dynamics network. The paper's reported eight-step error uses an unspecified nominal-error normalization; this repository's raw state error must not be compared numerically with Figure 3's error axis.

The port interprets reset returns over the planned finite episode horizon. If a reset permits `L` control steps before clip end or the configured timeout, its maximum discounted return is `r_max * (1 - gamma**L) / (1 - gamma)`. The observed discounted sum is normalized by this bound fixed at reset, including when failure occurs early. Per-bin estimates use grouped arithmetic sample means; historical softmax weights use arithmetic running means. This avoids an extrapolated reward tail and duplicate-index overwrites, but these estimator details are not specified by Appendix D.1.

Removing the extra orientation observation required changing the goal's information content. A reference-only goal expressed in reference-heading coordinates cannot distinguish two robot states that differ by a constant global yaw rotation. The 93 proprioceptive values do not supply this information either, although the root-orientation and world-velocity rewards penalize the difference. The port therefore expresses future target position relative to the current robot root, and target orientation/velocities in the current full robot body frame. It keeps 73 values per frame and applies the same transform in collection, imagination, and evaluation, including its state derivatives in BPTT. This is a disclosed coordinate choice, not a claim about the authors' unpublished encoding. Deployment would require estimates of current root position and orientation to construct this goal; simulator success alone does not establish sensor compatibility on a physical robot.

The learned termination head is an MLP with hidden widths 256/128. It receives invariant predicted robot state, invariant current reference state, and the future goal, and is trained on actual failure labels with binary cross entropy. Timeout labels only delimit replay windows. The policy objective retains gradients through predicted survival probabilities, while the world-model weights are frozen during its update. The state-regression term uses an unweighted squared tangent-space state error, summed over valid autoregressive steps; this explicitly differs from the imported increment-variance-weighted objective.

Using mjlab supplies the requested simulator integration; it does not itself prove identical contacts, actuator semantics, collision masks, friction mixing, root-velocity conventions, or reset state. Compare forward kinematics, observations, rewards, hard termination, physical context, and torque limits against simulator readback before treating a run as valid.

## Evaluation and acceptance

Score complete clips from their first frames with friction 1, no randomization or observation/action noise, and deterministic policy/physics. Use hard training failure thresholds, report failed clip names, and divide return by the maximum for each entire clip so early termination remains penalized. Preserve both clip-mean and frame-weighted returns; do not silently interchange them. The Figure 4 target is evaluation return 0.82, not training return from an evolving reset distribution.

Store source commit, dependency versions, asset/data hashes, launch arguments, GPU identity, and checkpoints with the run. Separate a passing smoke test, an alive long-running process, measured evaluation, and rendered motion acceptance. A 5090 run's wall time is not an H100 reproduction result. Evaluation time and training time should be recorded separately, including startup/compilation time where measured.

## Rotation representation

The dynamics model uses body-frame gravity (three values), body-frame velocities and body-frame translation increments. Future goals also use the full body frame. These preserve global yaw/translation invariance without a yaw chart at vertical pelvis poses. The imported epsilon-normalized projected-forward matrix was not orthogonal near vertical and could suppress horizontal movement and loss; it is removed from model coordinates. The inherited body-reward heading alignment retains the exact Euler `atan2(R10,R00)` convention and a proper SO(2) matrix. Its derivative is intrinsically undefined when the relative forward axis is exactly vertical; no epsilon-shaped surrogate is introduced.
