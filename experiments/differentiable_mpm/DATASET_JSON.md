# Writing a dataset JSON

## Created files and current limits

- `configs/dataset_episode18_registered_materials_v1.json` is a schema-checked **single-episode** dataset using the new table-aligned, registered-tool Episode18 inputs. It fits five material parameters over frames 1–60. It does not claim multi-episode readiness or independent held-out episodes.
- `configs/dataset_multi_episode_TEMPLATE_NOT_RUNNABLE.json` is an editable **multi-episode template**. Its first entry is the actual prepared Episode18 configuration. Its second entry deliberately contains `REPLACE_WITH_PREPARED_EPISODE_CONFIG.json`, a placeholder recording ID, and `end_frame: null`. Replace all three with verified values. It is valid JSON syntax but intentionally fails dataset validation until completed.
- `data/dataset_manifest_readiness_v1/readiness.json` records the available-input findings and missing preparations.
- `data/dataset_manifest_readiness_v1/validation.json` records the authoring checks.

**Shared tool-friction fitting is not supported by the current dataset driver.** Its allowed shared parameters remain `youngs_modulus`, `poisson_ratio`, `viscosity`, `plastic_min`, and `plastic_max`. The separate contact update adds single-episode friction tuning, not dataset shared-friction support. Adding `tool_friction_coefficient` to this dataset's `shared_parameters.initial` or `fit` is rejected; these files do not conceal that limitation.

The prepared file preserves the referenced configuration's fixed Coulomb tool coefficient **0.5** and floor retention **0.4**, with zero tool stickiness. This is not the same as the later recorded fit that used fixed tool coefficient 0.3. Initial material values are starting guesses from the prepared configuration, not new fitted results. No configuration was changed to copy the later fit silently.

## What the JSON means

The created prepared file has this structure; it also supplies explicit parameter bounds and a nonsticky tool policy:

```json
{
  "schema": "taichidough/differentiable-dataset/v1",
  "name": "episode18-registered-materials-only-single-episode",
  "shared_parameters": {
    "initial": {
      "youngs_modulus": 130579.320726,
      "poisson_ratio": 0.3,
      "viscosity": 0.0,
      "plastic_min": 0.9,
      "plastic_max": 1.1
    },
    "fit": [
      "youngs_modulus",
      "poisson_ratio",
      "viscosity",
      "plastic_min",
      "plastic_max"
    ]
  },
  "episodes": [
    {
      "id": "snimanje_23_10-episode18",
      "config": "episode18_table_aligned_registered_tools.json",
      "membership": "training",
      "weight": 1.0,
      "scored_window": {
        "start_frame": 1,
        "end_frame": 60,
        "stride": 1
      }
    }
  ]
}
```

- `shared_parameters.initial` contains exactly the five material values, including values held fixed if `fit` selects only a subset. `fit` lists which values the optimizer changes; `bounds` in the complete file limits those coordinates.
- `episodes` is a list. Add a separate entry for every distinct physical recording used in the dataset, not a second processed copy of Episode18.
- `id` is a unique name used in logs and `--path` overrides.
- `config` names that episode's existing experiment JSON. It resolves relative to the **dataset JSON's directory**. In the created files both JSONs are in `configs/`, so only the episode config's basename is needed.
- `membership: "training"` includes the episode in parameter selection. `membership: "validation"` excludes the whole episode from selection and uses it for evaluation. At least one training episode is required.
- `weight` must be positive. Weight 1 for every training episode gives equal weight to each episode's mean observation loss.
- `scored_window.start_frame`, `end_frame`, and `stride` are explicit integer processed-frame indices; endpoints are inclusive. Their limits must be checked against that recording. Replay still begins at reconstructed frame 0, even if scoring starts later.
- Optional `path_overrides` inside an episode entry maps input names such as `episode`, `calibration`, `initial_particles`, or `reconstruction_metadata` to paths. Manifest-relative overrides resolve relative to the dataset JSON. CLI overrides are described below.

To fit two episodes, both completed entries must have `membership: "training"`. To train on two episodes and independently evaluate a third, add a third completed entry with `membership: "validation"`. Do not duplicate Episode18 as a second row with another ID: duplicate recording paths and fingerprints are rejected.

The current one-episode file scores frames 1–60 only. The referenced single-episode config's separate 61–97 window does **not** automatically create a held-out dataset episode; dataset `scored_window` and `membership` are explicit. Temporal continuation on the same recording is also not the same as testing independent tool motions. Current Episode18 geometry preprocessing used frames across that recording, so any material-only held-out scoring is conditional on that preprocessing.

One shared material is an assumption about batch, hydration, preparation, and relevant handling history. Being recorded in the same directory or on the same date does not establish that assumption.

## What is missing for additional real episodes

The checked experimental episode configs and the stored broad inventory identify preparations for Episode18 only. The current primary local directory

```text
../data/deformpath_training/DeformPath3/snimanje_23_10/
```

contains only `episode18_kugla` as an episode directory.

The earlier broad inventory lists 370 episode-named variant directories across ten roots. Grouping its stored session-qualified `recording_id` values gives 111 candidate recording groups, not 370 independent trials and not 111 verified compatible dough specimens. All four reconstruction candidates in that inventory concern Episode18. The readiness JSON preserves these groups and checks current file presence without loading large raw tensors. This is a scoped inventory, not proof that no other preparation exists anywhere on either computer.

For every additional recording, the missing work is to identify or prepare:

1. Its own verified experiment config and processed point-cloud/tool sequence with timestamps and valid frame mapping.
2. A camera/table calibration applicable to that recording and correctly composed registered tool geometry.
3. Its own initial particles and reconstruction metadata, including volume and content hashes.
4. A justified positive mass and consistent density/particle-volume assignment; do not copy Episode18's mass or reconstructed volume without evidence.
5. A valid scoring window and explicit training/held-out membership.
6. Confirmation that sharing one material vector is appropriate.

Do not fill these missing inputs by renaming Episode18's files or bypassing hash checks. The Episode18 reconstruction's unusually high density remains a separate physical-validity concern even though its files are internally consistent.

## Validate and run on `/mnt`

Copy the new **prepared single-episode dataset JSON** into `experiments/differentiable_mpm/configs/` in the already extracted registered-tools bundle. The referenced Episode18 config and derived inputs are already in that bundle. Copying the template does not make its missing episodes ready.

The exact supported override syntax is **`--path EPISODE_ID.INPUT_NAME=PATH`**. There is no `--episode-path` option. An override's ID must match the entry's `id`.

This command validates the prepared single-episode file, including its inputs and SDF preparation, without initializing the numerical runtime:

```bash
cd /mnt/Data/studenti/antonio_skara/TaichiDough_registered_tools_v1

python -u -m experiments.differentiable_mpm.calibrate_dataset validate \
  --dataset experiments/differentiable_mpm/configs/dataset_episode18_registered_materials_v1.json \
  --path snimanje_23_10-episode18.episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --backend cuda \
  --precision f32 \
  --physics-version corrected-v1 \
  --reference-policy frozen \
  --no-runtime
```

After successful input validation, this fits the five material parameters with tool friction fixed:

```bash
python -u -m experiments.differentiable_mpm.calibrate_dataset fit \
  --dataset experiments/differentiable_mpm/configs/dataset_episode18_registered_materials_v1.json \
  --path snimanje_23_10-episode18.episode=/mnt/Data/studenti/antonio_skara/data/DeformPath3/DeformPath3/snimanje_23_10/episode18_kugla \
  --backend cuda \
  --precision f32 \
  --physics-version corrected-v1 \
  --p2g-mode atomic \
  --reference-policy frozen \
  --segment-length 64 \
  --ignore-recompute-mismatch \
  --iterations 20
```

These commands use **one prepared episode**. Once the multi-episode template has real per-episode inputs and no placeholders, use that completed manifest with the same CLI and repeat `--path` for each entry whose inputs need relocation. The driver optimizes one shared parameter vector, evaluating episodes sequentially to limit GPU memory; it does not launch all episodes simultaneously on the GPU.

The replay-mismatch flag permits finite recomputation differences and may produce approximate gradients; invalid/nonfinite states still fail. No calibration command above was executed while creating these files.

## Checks performed for these files

- Both files parse as JSON.
- The prepared single-episode manifest loads successfully with the immutable original registered-tools bundle's dataset loader, using explicit actual-local-path overrides.
- All seven configured prepared-asset hashes match the verified Episode18 config.
- The incomplete template is rejected as intended: `Window indices and stride must be integers`, because its unknown second episode endpoint is deliberately `null`. Its placeholder config must also be replaced.
- No MPM runtime initialization, simulation, calibration, broad tests, or new reconstruction was performed. Full input preflight was not repeated; earlier registered-input preparation evidence is distinct from these schema and asset checks.
- Existing configs, runs, raw data, production/frozen references, and numerical source were not modified.
