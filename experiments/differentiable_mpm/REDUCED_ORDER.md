# Learned reduced-order MPM model

This first implementation learns **forward dough dynamics**, not tool actions.
It compresses a fixed ordered particle set and evolves the reduced state under
supplied tool controls. The existing Taichi solver and policy adapters are unchanged.

## Representation

A full state has `25 * N` values: `x`, `v`, `C`, `F`, `Jp`, in that order.
Each field gets its own centered POD basis, obtained by SVD of training states.
Each field has one RMS normalization scale; latent coordinates are standardized
using training statistics. With rank 8 per field, the dynamics state has 40 values.
The actual rank is limited by the number of training snapshots and field width.
Zero-variance POD directions are disabled, so constant training fields remain at
their training mean. The model cannot infer unseen variation in those fields.

The residual MLP receives:

- the standardized latent state;
- two tool positions, rotation matrices, and linear/angular velocities (36 values);
- the nine material/contact parameters in `state.PARAMETER_NAMES` order.

It predicts a latent increment for **one fixed physics timestep**. Rotation matrices
make quaternion signs irrelevant. `control_features` converts PyTorch xyzw poses
and velocities without detaching gradients. The timestep, numerical configuration,
ordered reference particles and collider SDF belong to the model's scene identity.
A different scene is rejected during training/evaluation.

Training uses complete-trajectory train/validation/test splits. Means, scales and
POD bases use only training states. The loss combines short recursively predicted
sequences with individual one-step predictions. Model selection uses full recursive
validation trajectories. Test trajectories are never used for model selection.

## Commands

Use `/usr/bin/python3` on this host: it has Torch 2.9 CPU and Taichi 1.7.4.
The default conda Python does not have these dependencies.

A small synthetic **Taichi** contact example (27 particles, two spherical tools):

```bash
/usr/bin/python3 scripts/run_reduced_order_mpm.py collect-demo \
  --output experiments/differentiable_mpm/runs/reduced_order/contact_data \
  --trajectories 10 --steps 48
/usr/bin/python3 scripts/run_reduced_order_mpm.py train \
  --data experiments/differentiable_mpm/runs/reduced_order/contact_data \
  --output experiments/differentiable_mpm/runs/reduced_order/contact_model \
  --rank 8 --epochs 100 --horizon 8
```

An existing calibrated experiment can be collected without changing its inputs:

```bash
/usr/bin/python3 scripts/run_reduced_order_mpm.py collect \
  --config path/to/experiment.json --trajectory-id recorded-001 \
  --steps 100 --output path/to/data/recorded-001.npz
```

For useful training, collect **independent initial-state/control trajectories** with
the same reference particle identities and tool geometry. The Python API
`collect_trajectory(..., reference_x=shared_reference)` supports varied internal
states and supplied tool paths. Different reconstructed chunks generally do not
share particle identity and cannot simply be mixed. A single recorded path is not
a training/validation/test dataset, and adjacent pieces of the same recording
should not be relabeled as independent trajectories.

Evaluate an independent archive:

```bash
/usr/bin/python3 scripts/run_reduced_order_mpm.py evaluate \
  --model path/to/model.pt --trajectories path/to/held_out.npz \
  --output path/to/evaluation.json
```

Output includes per-field RMS compression, one-step and recursive errors;
persistence and constant-velocity baselines; vector position RMS in metres;
nonpositive deformation determinant/plastic-volume counts; and rollout timing.
Latent timing excludes encoding and decoding. Collection timing, when available,
includes state transfers and may include first-use Taichi compilation; it is **not
an equivalent optimized-physics timing or evidence of a speedup**.

## Differentiable inference

```python
import torch
from experiments.differentiable_mpm.reduced_order import ReducedDynamics, control_features

model = ReducedDynamics.load('path/to/model.pt')
model.check_scene(trajectory.identity, trajectory.dt, trajectory.n_particles)
state = torch.as_tensor(trajectory.states[0], dtype=torch.float32)
parameters = torch.as_tensor(trajectory.parameters, dtype=torch.float32)
# poses: [T,2,7] in simulation/world frame, quaternion xyzw
# velocities: [T,2,6], supplied consistently with the commanded path
controls = control_features(poses, velocities)
z0 = model.codec.encode(state)
z = model.rollout(z0, controls, parameters)
predicted_states = model.codec.decode(z)
```

Keep `z` internally for repeated stepping; decode positions when an observation
is needed. This avoids full-state decoding at every latent step. A point cloud
alone cannot initialize velocity, affine velocity or deformation history. This
implementation therefore does **not** replace the existing point-cloud Gym adapter.
No automatic policy-to-surrogate conversion is performed.

## First measured check (2026-10-01)

Nine synthetic contact trajectories were collected through Taichi 1.7.4 CPU;
seven were used for training, one for validation, and one for test. Every trajectory
has 27 particles and 48 physics steps (9.6 ms simulated time). The collection
process reached its two-minute execution limit before saving the tenth trajectory.
The nine completed archives are kept alongside the model.

`runs/reduced_order/contact_verified/` contains `model.pt`, `report.json`,
`reloaded_evaluation.json`, and `data/`. Training selected epoch 25 of 100.
The model uses 40 allocated latent coordinates versus 675 full-state values;
zero-variance directions are disabled.

On the single independent test trajectory:

- Recursive position vector RMS: **2.887 mm**; final-frame vector RMS: **3.034 mm**.
- Unchanged-state vector RMS: **6.939 mm** (component RMS 4.006 mm).
- POD position vector reconstruction RMS: **1.421 mm** (component RMS 0.820 mm).
- No nonpositive `det(F)` or `Jp` values in the predicted trajectory.
- Saved-model reload reproduces every reported error metric exactly.

Material parameters and `Jp` are constant in this training example. It therefore
does not establish learned material generalization or evolving plastic history;
the separate collector tests exercise plastic-state updates.

These are short synthetic results, not real-episode scores or proof of speedup.
The pre-timing archives have no collection timing. There is not enough evidence
to replace calibrated MPM for policy evaluation yet. The 19 new tests pass;
11 existing policy-adapter tests and 5 retrieval-adapter tests also pass (the latter
require the local `dom_retrieval` parent directory on `PYTHONPATH`).

## Limits and checks

- Fixed topology, tool geometry and timestep; no claim of generalizing to arbitrary
  dough reconstructions or unseen contact/material regimes.
- Every physics-step control is stored; no endpoint-only approximation of an
  arbitrary intervening tool path.
- Two Taichi state slots bound solver state memory, but the archive stores all
  snapshots in RAM. Storage is `8 * (T+1) * 25 * N` bytes for float64 before
  compression. Use short runs for large particle sets. Dense SVD also needs the
  training snapshot matrix in memory.
- POD decode does not guarantee physical validity, conservation, positive `Jp`,
  positive `det(F)` or stable long rollouts. Report these checks; do not feed decoded
  states back into Taichi without separate physical validation.
- The synthetic contact example verifies collection and learning, not calibrated
  real-dough accuracy. Tool-path gradients belong to the learned approximation;
  they are not certified Taichi gradients.

Tests:

```bash
/usr/bin/python3 -m unittest \
  experiments.differentiable_mpm.tests.test_reduced_order \
  experiments.differentiable_mpm.tests.test_reduced_order_data -v
```
