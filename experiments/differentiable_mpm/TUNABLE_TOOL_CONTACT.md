# Tunable tool contact

For a non-adhesive experiment, fit **tool friction only**, keep stickiness zero, and use `coulomb-v1`. The original registered-tools configuration and archive are unchanged. Single-episode configurations use their existing physical parameter fields. Dataset schema v2 can additionally share floor retention, tool friction, and stickiness across episodes; schema v1 remains material-only.

## Parameters and supported models

The original seven physical parameter indices remain unchanged. Two differentiable entries are appended:

| Parameter | Physical range | Default fitting bounds | Coordinate scale |
| --- | --- | --- | --- |
| `tool_friction_coefficient` | finite, nonnegative | [0, 2] | 1 |
| `tool_stickiness` | [0, 1] | [0, 1] | 1 |

Set initial values under `parameters`, include active names in `fit_parameters`, and put their bounds under `parameter_bounds`. Bounds may only name fitted parameters. Both coefficients have linear optimizer coordinates, not sigmoid coordinates.

- `retention-v1`: stickiness is tunable; friction is inactive and cannot be fitted. The relative velocity is multiplied by `tool_retention * (1 - tool_contact_absorption) * (1 - tool_stickiness)` after removing inward normal motion. Fitting retention and stickiness together is rejected because only their product is identifiable. Fixed zero retention or absorption equal to one also makes stickiness inactive and is rejected when fitting it.
- `coulomb-v1`: friction is tunable; stickiness and absorption must be zero. Separating relative velocity remains unchanged. Retention cannot be fitted because it is inactive.
- `coulomb-adhesive-v1`: explicit optional joint mode. First apply the existing Coulomb response, then `v -= tool_stickiness * (v - collider_velocity)` while geometric contact is active. This acts during compression and withdrawal. Zero stickiness gives exactly the original Coulomb response. Absorption must be zero, and retention is inactive.

New contact fits require `tool_collision = "sdf"`. Both grid and projected-particle contact paths use the differentiable entries. Floor contact is unchanged: `floor_retention`, not a floor Coulomb coefficient, remains the tunable tangential multiplier.

The optional joint model is **velocity matching**, not a measured adhesion strength: no persistent bonds, separation-distance law, or pull-off force is implemented. The effect depends on time step and contact frequency. Pressing data can confound friction, stickiness, and material stiffness; sliding and withdrawal observations provide different information. Derivatives are piecewise smooth and are not guaranteed at contact activation or friction branch switches.

## Separate examples

- `configs/episode18_registered_friction_fit.json`: fits only friction, initially 0.3 with bounds [0, 2]; all material values remain fixed at the registered example's initial values.
- `configs/episode18_registered_adhesive_fit.json`: optionally fits friction and stickiness, initially 0.3 and zero. Stickiness bounds are [0, 1]. It uses the explicit joint model.

To fit material values together with friction, add their canonical names to `fit_parameters` and add corresponding bounds. The friction-only example does not silently load the previous material fit. To start from that fit, explicitly pass its `selected_parameters.json` with `--parameters`; keep the sibling `run_manifest.json` for historical fixed-contact recovery.

From the updated repository root on `/mnt`, input-only validation is:

```bash
python3 -m experiments.differentiable_mpm.calibrate validate \
  --config experiments/differentiable_mpm/configs/episode18_registered_friction_fit.json \
  --path episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --reference-policy frozen --no-runtime
```

The single-episode fitting command is supplied for the operator; it has **not** been launched:

```bash
python3 -m experiments.differentiable_mpm.calibrate fit \
  --config experiments/differentiable_mpm/configs/episode18_registered_friction_fit.json \
  --path episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --backend cuda --reference-policy frozen --iterations 20
```

Replace the config filename with `episode18_registered_adhesive_fit.json` only to opt into the joint approximation. No replay-mismatch override is included: unresolved CUDA replay differences still need qualification. CPU-f64 tests do not establish CUDA gradient accuracy or real-data convergence.

## Old configurations and saved selections

Old seven-value dictionaries inherit missing contact values from their simulation settings. Explicit physical contact values take precedence, and config validation mirrors them into exported simulation settings. The original indices and contact laws are preserved.

When `calibrate --parameters` loads an old seven-value selection, it verifies a sibling run manifest's identity and obtains missing contact values from `identity.prepared.simulation`. Otherwise it uses the supplied experiment's simulation settings, then defaults. Explicit new physical values remain authoritative. Selected-file identity mismatches are rejected; the original files are never rewritten. This preserves friction 0.3 in the actual historical Episode18 fit instead of substituting the new default 0.5.

This is a parameter-file migration, not permission to resume an old optimizer trajectory with a changed source identity. Existing exact-resume identity checks remain. The frozen reference simulator does not implement the new adhesive mode; its adapter rejects that mode instead of reporting misleading parity.

## Portable update

The old `episode18_registered_tools_v1.tar.gz` has not been rebuilt. Editing only JSON in that old bundle will not add differentiable contact parameters. Transfer these updated files together, preserving paths under `experiments/differentiable_mpm/`:

```text
state.py
solver.py
parameters.py
config.py
calibrate.py
dataset_config.py
reference_adapter.py
configs/episode18_registered_friction_fit.json
configs/episode18_registered_adhesive_fit.json
TUNABLE_TOOL_CONTACT.md
tests/test_tunable_contact.py
tests/test_checkpoint.py
tests/test_optimizer.py
tests/test_coulomb_contact.py
tests/test_solver.py
tests/test_forward_parity.py
tests/test_preservation.py
```

The existing registered data, frozen reference snapshots, manifests, raw episode data, and production simulator do not require changes. Preserve any remote local edits before replacing source files. No calibration, dependency installation, archive replacement, commit, or push was performed for this update.

## Verification

- Host optimizer/checkpoint/CLI/dataset regressions: 104 tests passed.
- Contact compatibility plus numerical checks: 11 tests passed in 141.220 seconds on single-thread CPU-f64. Actual reverse-mode derivatives match central finite differences for friction and stickiness in velocity, grid, and particle response paths, including inward, separating, and zero-tangent cases. Tests verify nonzero active gradients, exact zero-stickiness Coulomb equivalence, withdrawal damping, and both parameter derivatives through a three-step serial-P2G MPM rollout.
- Additional migration/example/reference-rejection tests plus CLI regressions: 34 tests passed. The synthetic nine-parameter optimizer test also passed with nontrivial friction and stickiness targets.
- Existing solver and preservation checks: 39 test methods passed. In the same 41-test invocation (1116.668 seconds), both forward-parity methods refused all 18 fixtures under strict reference policy because the existing production simulator differs from its recorded reference. No production source or reference hashes were changed to bypass that check. Rerunning only the two parity methods with the existing explicit `reference_policy('frozen')` passed all 18 corrected/legacy fixtures in 523.234 seconds, including nonzero fixed-stickiness SDF contact.
- Both new example configurations and the unchanged registered configuration passed config and input-file/hash validation. Actual historical selected parameters recovered friction 0.3 using their verified sibling run manifest.

Focused tests can be repeated from the repository root:

```bash
python3 -m unittest \
  experiments.differentiable_mpm.tests.test_tunable_contact \
  experiments.differentiable_mpm.tests.test_coulomb_contact \
  experiments.differentiable_mpm.tests.test_optimizer \
  experiments.differentiable_mpm.tests.test_checkpoint -q
```

Taichi emitted reverse-kernel compiler warnings about variables loaded before stores; the contact AD/FD assertions nevertheless passed. These warnings are recorded, not treated as evidence of correctness. No GPU derivative test or real calibration was run.
