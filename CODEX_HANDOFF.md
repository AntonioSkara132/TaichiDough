# Codex Handoff: TaichiDough / DeformPath Chunk Calibration

## Current Focus

We are treating DeformPath chunks as independent relaxed-state episodes for TaichiDough material calibration. Each chunk should start from its own frame-0 observed geometry; chunks do not carry hidden simulator state from previous chunks.

## Important Current Changes

- `experiments/differentiable_mpm/calibrate_dataset.py`
  - Added `--episode-batch-size`.
  - When set on `fit`, training episodes are split into independent episode minibatches.
  - Each batch is evaluated with up to `episode_batch_size` workers in parallel.
  - Parameters are updated after every batch.
  - Old full-dataset optimizer path is unchanged when `--episode-batch-size` is absent.
  - Minibatch mode writes `optimizer_state.json` with schema `taichidough/minibatch-adam-state/v1`.
  - `selected_parameters.json` records objective metadata:
    - `version: independent-episode-minibatch-v1`
    - effective `episode_batch_size`
    - final full training evaluation over all training chunks.

- `scripts/materialize_chunked_deformpath_dataset.py`
  - This already reconstructs the beginning of every chunk.
  - It runs `reconstruct_voxel_dough_from_deformpath.py --frame 0` per chunk.
  - The generated chunk `differentiable_mpm.json` points `initial_particles` at that chunk-local reconstruction.
  - Added `--validation-chunk NAME[,NAME...]` so some chunks can be held out as validation episodes while the rest remain training.

- Tests added/updated:
  - `experiments/differentiable_mpm/tests/test_calibrate_dataset.py`
    - New test proves minibatch mode updates after each independent training episode batch.
  - `experiments/differentiable_mpm/tests/test_materialize_chunked_deformpath_dataset.py`
    - New test covers validation chunk selection parsing and rejects unknown/all-validation selections.

## Verified

From `/home/antonio/diplomski_antonio/diplomski/TaichiDough`:

```zsh
python3 -m py_compile scripts/materialize_chunked_deformpath_dataset.py experiments/differentiable_mpm/calibrate_dataset.py experiments/differentiable_mpm/tests/test_materialize_chunked_deformpath_dataset.py
python3 -m unittest experiments.differentiable_mpm.tests.test_calibrate_dataset -v
python3 -m pytest experiments/differentiable_mpm/tests/test_materialize_chunked_deformpath_dataset.py -q
```

Result:

- `test_calibrate_dataset`: 23 tests OK
- `test_materialize_chunked_deformpath_dataset`: 2 tests passed

## Run Command Shape

```zsh
cd /home/antonio/diplomski_antonio/diplomski/TaichiDough

python3 -m experiments.differentiable_mpm.calibrate_dataset fit \
  --dataset /home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_kugla/chunks/dataset.json \
  --episode-batch-size 4 \
  --iterations 20 \
  --reference-policy frozen \
  --backend cpu \
  --precision f64 \
  --cpu-threads 1 \
  --output-dir experiments/differentiable_mpm/runs/episode18_chunk_minibatch_fit
```

`--episode-batch-size 4` means four independent chunk episodes can run in parallel, then one parameter update happens after that batch.

## Files That May Need Committing

```text
experiments/differentiable_mpm/calibrate_dataset.py
experiments/differentiable_mpm/tests/test_calibrate_dataset.py
experiments/differentiable_mpm/tests/test_materialize_chunked_deformpath_dataset.py
scripts/materialize_chunked_deformpath_dataset.py
scripts/chunk_deformpath_episode.py
scripts/preprocess_deformpath_episode.py
```

## Data That GitHub Probably Will Not Transfer

The prepared chunk dataset is data and should be copied separately if running on another computer:

```text
/home/antonio/diplomski_antonio/diplomski/data/deformpath_training/preprocessed_dataset/episode18_kugla/chunks
```

This directory was about 30 MB locally and includes:

- `dataset.json`
- `chunk_manifest.json`
- per-chunk `differentiable_mpm.json`
- per-chunk `pointclouds_interpolated.pt`
- per-chunk `paths_interpolated.pt`
- per-chunk `scene_calibration_v2.json`
- per-chunk reconstruction folders with frame-0 sampled particles

## Request For Other Session

If another Codex session is working in this project, please append below:

- What files you changed.
- What command/test you ran.
- What assumptions you made about chunk independence, train/validation split, or simulator initial state.
- Any branch/diff that conflicts with the files above.

## Other Session Notes

_Append here._
