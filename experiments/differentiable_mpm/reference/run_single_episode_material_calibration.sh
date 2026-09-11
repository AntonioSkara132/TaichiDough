#!/usr/bin/env bash
# Reconstruct, inspect, and calibrate one metric DeformPath episode.
#
# Required environment variables:
#   EPISODE             Directory containing pointclouds_interpolated.pt and paths_interpolated.pt
#   CALIBRATION         Optional explicit taichidough/scene-calibration/v2 JSON. When unset,
#                       use and verify the metric calibration attached to EPISODE.
#   TOOL_GEOMETRY       Measured taichidough/tool-geometry/v1 JSON
#   DOUGH_MASS_KG       Measured dough mass in kilograms
#
# Example:
#   EPISODE=/data/episode18_kugla \
#   CALIBRATION=data/calibration/episode18_scene_v2.json \
#   TOOL_GEOMETRY=configs/tool_geometry_measured.json \
#   DOUGH_MASS_KG=0.385 \
#   TRAIN_END_FRAME=60 VALIDATION_END_FRAME=120 \
#   bash scripts/run_single_episode_material_calibration.sh
#
# AprilTag collection must be completed first. This script intentionally does
# not invent a tag transform, floor plane, tool geometry, or dough mass.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

require_value() {
    local name=$1
    if [[ -z ${!name:-} ]]; then
        printf 'Missing required environment variable: %s\n' "$name" >&2
        exit 2
    fi
}

for name in EPISODE TOOL_GEOMETRY DOUGH_MASS_KG; do
    require_value "$name"
done

PYTHON=${PYTHON:-python3}
EPISODE=$(realpath -e "$EPISODE")
TOOL_GEOMETRY=$(realpath -e "$TOOL_GEOMETRY")
EXPLICIT_CALIBRATION=${CALIBRATION:-}
if [[ -n $EXPLICIT_CALIBRATION ]]; then
    CALIBRATION=$(realpath -e "$EXPLICIT_CALIBRATION")
else
    CALIBRATION=$(EPISODE="$EPISODE" REPO_ROOT="$REPO_ROOT" "$PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["REPO_ROOT"]) / "scripts"))
from deformpath_dynamics import resolve_episode_calibration

_, provenance = resolve_episode_calibration(Path(os.environ["EPISODE"]))
print(provenance["artifact_path"])
PY
)
fi

# An explicit calibration may be used only when it is the recorded calibration,
# unless ALLOW_CALIBRATION_OVERRIDE=1 was deliberately set for an investigation.
if [[ -n $EXPLICIT_CALIBRATION && ${ALLOW_CALIBRATION_OVERRIDE:-0} != 1 ]]; then
    EPISODE="$EPISODE" CALIBRATION="$CALIBRATION" REPO_ROOT="$REPO_ROOT" "$PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["REPO_ROOT"]) / "scripts"))
from deformpath_dynamics import resolve_episode_calibration
from deformpath_topview import load_calibration

attached, _ = resolve_episode_calibration(Path(os.environ["EPISODE"]))
override = load_calibration(Path(os.environ["CALIBRATION"]))
if attached.fingerprint != override.fingerprint:
    raise SystemExit("CALIBRATION does not match the episode-attached calibration; set ALLOW_CALIBRATION_OVERRIDE=1 only for an explicit investigation")
PY
fi

EPISODE="$EPISODE" CALIBRATION="$CALIBRATION" EXPLICIT_CALIBRATION="$EXPLICIT_CALIBRATION" REPO_ROOT="$REPO_ROOT" "$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(os.environ["REPO_ROOT"]) / "scripts"))
from deformpath_topview import load_calibration

path = Path(os.environ["CALIBRATION"])
calibration = load_calibration(path)
digest = hashlib.sha256(path.read_bytes()).hexdigest()
print(f"Calibration: {path}")
print(f"Source: {'episode attachment' if not os.environ.get('EXPLICIT_CALIBRATION') else 'explicit override'}")
print(f"Frames: source={calibration.source_frame}, scene={calibration.scene_frame}")
print(f"Floor plane: {calibration.floor_plane_scene.tolist() if calibration.floor_plane_scene is not None else None}")
print(f"SHA-256: {digest}")
print(f"Fingerprint: {calibration.fingerprint}")
PY

FRAME=${FRAME:-0}
PARTICLES=${PARTICLES:-24000}
GRID=${GRID:-48}
DT=${DT:-0.0002}
SUBSTEPS_PER_FRAME=${SUBSTEPS_PER_FRAME:-1}
REPLAY_STRIDE=${REPLAY_STRIDE:-1}
REPLAY_MAX_GAP_S=${REPLAY_MAX_GAP_S:-0.1}
DEPTH_SPLAT_RADIUS=${DEPTH_SPLAT_RADIUS:-1}
INITIAL_PARTICLE_SEED=${INITIAL_PARTICLE_SEED:-0}

# These are fixed while the material search varies only Young's modulus.
POISSON_RATIO=${POISSON_RATIO:-0.3}
VISCOSITY=${VISCOSITY:-0.0}
GRAVITY=${GRAVITY:--9.81}
FLOOR_FRICTION=${FLOOR_FRICTION:-0.4}
FLOOR_ABSORPTION=${FLOOR_ABSORPTION:-0.0}
TOOL_CONTACT_PADDING_WAS_SET=${TOOL_CONTACT_PADDING+x}
TOOL_CONTACT_PADDING=${TOOL_CONTACT_PADDING:-0.0}
TOOL_CONTACT_FRICTION=${TOOL_CONTACT_FRICTION:-0.2}
TOOL_CONTACT_ABSORPTION=${TOOL_CONTACT_ABSORPTION:-0.0}
TOOL_STICKINESS=${TOOL_STICKINESS:-0.0}
FLOOR_STICKINESS=${FLOOR_STICKINESS:-0.0}
FLOOR_PLASTIC_DAMPING_BAND=${FLOOR_PLASTIC_DAMPING_BAND:-0.0}
VELOCITY_DAMPING=${VELOCITY_DAMPING:-1.0}
PLASTIC_MIN=${PLASTIC_MIN:-0.9}
PLASTIC_MAX=${PLASTIC_MAX:-1.1}
PLASTIC_VELOCITY_DAMPING=${PLASTIC_VELOCITY_DAMPING:-1.0}
PLASTIC_AFFINE_DAMPING=${PLASTIC_AFFINE_DAMPING:-1.0}
JP_HARDENING=${JP_HARDENING:-0.0}
JP_MIN=${JP_MIN:-0.5}
JP_MAX=${JP_MAX:-2.0}

# Choose windows after checking the recorded tool motion. They are inclusive
# evaluator ranges; replay starts from FRAME for every candidate.
TRAIN_START_FRAME=${TRAIN_START_FRAME:-1}
TRAIN_END_FRAME=${TRAIN_END_FRAME:-60}
VALIDATION_START_FRAME=${VALIDATION_START_FRAME:-61}
VALIDATION_END_FRAME=${VALIDATION_END_FRAME:-120}

# The 2000 Pa baseline is always scored, so the initial search interval must
# contain it. Change these only before the first material run.
INITIAL_MIN_PA=${INITIAL_MIN_PA:-1000}
INITIAL_MAX_PA=${INITIAL_MAX_PA:-300000}
HARD_MIN_PA=${HARD_MIN_PA:-100}
HARD_MAX_PA=${HARD_MAX_PA:-1000000}
COARSE_COUNT=${COARSE_COUNT:-7}
REFINEMENT_ROUNDS=${REFINEMENT_ROUNDS:-2}
REFINEMENT_SUBDIVISIONS=${REFINEMENT_SUBDIVISIONS:-4}
MAX_BOUNDARY_EXPANSIONS=${MAX_BOUNDARY_EXPANSIONS:-2}
BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-2000}
SUBPROCESS_TIMEOUT_S=${SUBPROCESS_TIMEOUT_S:-3600}

# Visual STLs are render inputs. SDF collision uses the separately validated solids.
TOOL_COLLISION=${TOOL_COLLISION:-sdf}
TOOL_SDF_RESOLUTION=${TOOL_SDF_RESOLUTION:-64}
UR_TOOL_MESH=${UR_TOOL_MESH:-"$REPO_ROOT/meshes/ur_spathla.stl"}
KINOVA_TOOL_MESH=${KINOVA_TOOL_MESH:-"$REPO_ROOT/meshes/gen3_spathla.stl"}
UR_TOOL_COLLISION_MESH=${UR_TOOL_COLLISION_MESH:-"$REPO_ROOT/meshes/ur_spathla_collision_solid.stl"}
KINOVA_TOOL_COLLISION_MESH=${KINOVA_TOOL_COLLISION_MESH:-"$REPO_ROOT/meshes/gen3_spathla_collision_solid.stl"}
TOOL_COLLISION_MANIFEST=${TOOL_COLLISION_MANIFEST:-"$REPO_ROOT/meshes/tool_collision_meshes_v1.json"}
TOOL_SKIN_LABEL=${TOOL_SKIN_LABEL:-}
TOOL_MESH_SCALE=${TOOL_MESH_SCALE:-0.001}
if [[ $TOOL_COLLISION != box && $TOOL_COLLISION != sdf && $TOOL_COLLISION != none ]]; then
    printf 'TOOL_COLLISION must be box, sdf, or none; got %q\n' "$TOOL_COLLISION" >&2
    exit 2
fi
if [[ $TOOL_COLLISION == sdf ]]; then
    if [[ -z $TOOL_CONTACT_PADDING_WAS_SET ]]; then
        printf 'TOOL_CONTACT_PADDING must be explicitly set for TOOL_COLLISION=sdf\n' >&2
        exit 2
    fi
    require_value TOOL_SKIN_LABEL
    for name in UR_TOOL_MESH KINOVA_TOOL_MESH UR_TOOL_COLLISION_MESH KINOVA_TOOL_COLLISION_MESH TOOL_COLLISION_MANIFEST; do
        require_value "$name"
        printf -v "$name" '%s' "$(realpath -e "${!name}")"
    done
fi

EPISODE_NAME=$(basename "$EPISODE")
OUTPUT_ROOT=${OUTPUT_ROOT:-"$REPO_ROOT/data/single_episode_calibration/$EPISODE_NAME"}
OUTPUT_ROOT=$(mkdir -p "$OUTPUT_ROOT" && realpath -e "$OUTPUT_ROOT")
RECONSTRUCTION="$OUTPUT_ROOT/reconstruction/$EPISODE_NAME/frame_$(printf '%04d' "$FRAME")"
MANIFEST="$OUTPUT_ROOT/material_calibration_manifest.json"
RESULT="$OUTPUT_ROOT/material_calibration_result.json"

export PYTHON EPISODE CALIBRATION TOOL_GEOMETRY DOUGH_MASS_KG FRAME PARTICLES GRID DT SUBSTEPS_PER_FRAME
export REPLAY_STRIDE REPLAY_MAX_GAP_S DEPTH_SPLAT_RADIUS INITIAL_PARTICLE_SEED POISSON_RATIO VISCOSITY GRAVITY
export FLOOR_FRICTION FLOOR_ABSORPTION TOOL_CONTACT_PADDING TOOL_CONTACT_FRICTION TOOL_CONTACT_ABSORPTION
export TOOL_STICKINESS FLOOR_STICKINESS FLOOR_PLASTIC_DAMPING_BAND VELOCITY_DAMPING PLASTIC_MIN PLASTIC_MAX
export PLASTIC_VELOCITY_DAMPING PLASTIC_AFFINE_DAMPING JP_HARDENING JP_MIN JP_MAX TRAIN_START_FRAME TRAIN_END_FRAME
export VALIDATION_START_FRAME VALIDATION_END_FRAME INITIAL_MIN_PA INITIAL_MAX_PA HARD_MIN_PA HARD_MAX_PA COARSE_COUNT
export REFINEMENT_ROUNDS REFINEMENT_SUBDIVISIONS MAX_BOUNDARY_EXPANSIONS BOOTSTRAP_SAMPLES SUBPROCESS_TIMEOUT_S
export TOOL_COLLISION TOOL_SDF_RESOLUTION UR_TOOL_MESH KINOVA_TOOL_MESH UR_TOOL_COLLISION_MESH KINOVA_TOOL_COLLISION_MESH TOOL_COLLISION_MANIFEST TOOL_SKIN_LABEL TOOL_MESH_SCALE REPO_ROOT OUTPUT_ROOT
export RECONSTRUCTION MANIFEST RESULT

# Validate required physical inputs and make the reconstruction once. Existing
# reconstruction metadata is preserved so its particle fingerprint cannot be
# replaced accidentally during a resumed material search.
"$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np

calibration = json.loads(Path(os.environ["CALIBRATION"]).read_text(encoding="utf-8"))
if calibration.get("schema") != "taichidough/scene-calibration/v2":
    raise SystemExit("CALIBRATION must use taichidough/scene-calibration/v2")
if calibration.get("floor_plane_scene") is None:
    raise SystemExit("CALIBRATION must record floor_plane_scene")
tools = json.loads(Path(os.environ["TOOL_GEOMETRY"]).read_text(encoding="utf-8"))
if tools.get("schema") != "taichidough/tool-geometry/v1" or len(tools.get("tools", [])) != 2:
    raise SystemExit("TOOL_GEOMETRY must describe exactly two measured tools")
if float(os.environ["DOUGH_MASS_KG"]) <= 0:
    raise SystemExit("DOUGH_MASS_KG must be positive")
for value in ("FRAME", "TRAIN_START_FRAME", "TRAIN_END_FRAME", "VALIDATION_START_FRAME", "VALIDATION_END_FRAME"):
    if int(os.environ[value]) < 0:
        raise SystemExit(f"{value} must be non-negative")
if int(os.environ["TRAIN_END_FRAME"]) <= int(os.environ["TRAIN_START_FRAME"]):
    raise SystemExit("TRAIN_END_FRAME must exceed TRAIN_START_FRAME")
if int(os.environ["VALIDATION_END_FRAME"]) <= int(os.environ["VALIDATION_START_FRAME"]):
    raise SystemExit("VALIDATION_END_FRAME must exceed VALIDATION_START_FRAME")
if max(int(os.environ["TRAIN_START_FRAME"]), int(os.environ["VALIDATION_START_FRAME"])) <= min(int(os.environ["TRAIN_END_FRAME"]), int(os.environ["VALIDATION_END_FRAME"])):
    raise SystemExit("training and validation frame ranges must not overlap")
PY

if [[ ! -f "$RECONSTRUCTION/reconstruction_metadata.json" ]]; then
    "$PYTHON" scripts/reconstruct_voxel_dough_from_deformpath.py \
        --episode-dir "$EPISODE" \
        --frame "$FRAME" \
        --output-dir "$OUTPUT_ROOT/reconstruction" \
        --num-particles "$PARTICLES" \
        --voxel-size "${VOXEL_SIZE_M:-0.003}" \
        --fill-mode floor \
        --fill-axis y \
        --fill-direction negative \
        --floor-clearance "${FLOOR_CLEARANCE_M:-0.003}" \
        --floor-min-thickness "${FLOOR_MIN_THICKNESS_M:-0.001}" \
        --floor-max-thickness "${FLOOR_MAX_THICKNESS_M:-0.25}" \
        --footprint-dilate "${FOOTPRINT_DILATE:-1}" \
        --calibration "$CALIBRATION" \
        --save-pt
else
    printf 'Using existing reconstruction: %s\n' "$RECONSTRUCTION"
fi

# Extract the floor height, camera dimensions, and measured-mass density from
# reconstruction metadata. All candidate simulations use these exact values.
eval "$("$PYTHON" - <<'PY'
import json
import os
from pathlib import Path

metadata = json.loads((Path(os.environ["RECONSTRUCTION"]) / "reconstruction_metadata.json").read_text())
volume = float(metadata["object_volume_m3"])
plane = metadata["fill"]["floor_plane_scene"]
if abs(float(plane[0])) > 1e-6 or abs(float(plane[2])) > 1e-6 or abs(abs(float(plane[1])) - 1.0) > 1e-6:
    raise SystemExit("reconstruction floor is not compatible with the simulator constant-Y floor")
floor_y = -float(plane[3]) / float(plane[1])
calibration = json.loads(Path(os.environ["CALIBRATION"]).read_text())
camera = calibration["camera"]
print(f"DENSITY_KG_M3={float(os.environ['DOUGH_MASS_KG']) / volume!r}")
print(f"FLOOR_Y={floor_y!r}")
print(f"DEPTH_WIDTH={int(camera['width'])}")
print(f"DEPTH_HEIGHT={int(camera['height'])}")
PY
)"
export DENSITY_KG_M3 FLOOR_Y DEPTH_WIDTH DEPTH_HEIGHT

# Produce the zero-time diagnostics once. Dynamic simulator/evaluator runs for
# every candidate are issued by calibrate_youngs_modulus.py below.
STATIC_DIR="$OUTPUT_ROOT/static"
if [[ ! -f "$STATIC_DIR/taichi/camera_parameters.json" ]]; then
    mkdir -p "$STATIC_DIR"
    "$PYTHON" scripts/taichi_viscoelastic_mpm_scene.py \
        --cpu --steps 0 --save-initial-frame --no-publish-dough-center \
        --tool-collision none \
        --view deformpath_top \
        --particles "$PARTICLES" --grid "$GRID" --dt "$DT" \
        --density "$DENSITY_KG_M3" --object-mass-kg "$DOUGH_MASS_KG" --floor-y "$FLOOR_Y" \
        --initial-particles "$RECONSTRUCTION/sampled_particles_xyz.npy" \
        --initial-particles-metadata "$RECONSTRUCTION/reconstruction_metadata.json" \
        --initial-particles-calibration "$CALIBRATION" \
        --initial-particles-fit none --initial-particles-axis-map xyz \
        --save-depth-pointclouds --depth-pointcloud-frame camera_optical --depth-pointcloud-format xyz \
        --depth-width "$DEPTH_WIDTH" --depth-height "$DEPTH_HEIGHT" \
        --depth-splat-radius "$DEPTH_SPLAT_RADIUS" \
        --output-dir "$STATIC_DIR/taichi"
fi

"$PYTHON" scripts/evaluate_static_topview_match.py \
    --episode-dir "$EPISODE" --frame "$FRAME" --calibration "$CALIBRATION" \
    --taichi-metadata "$STATIC_DIR/taichi/camera_parameters.json" \
    --output-dir "$STATIC_DIR/evaluation"
"$PYTHON" scripts/visualize_static_topview_benchmark.py \
    --reconstruction-metadata "$RECONSTRUCTION/reconstruction_metadata.json" \
    --taichi-metadata "$STATIC_DIR/taichi/camera_parameters.json" \
    --metrics "$STATIC_DIR/evaluation/static_topview_metrics.json" \
    --calibration "$CALIBRATION" --output-dir "$STATIC_DIR/report"

# Generate a fully fingerprinted single-episode manifest. The manifest stores
# paths plus content hashes for every file used by the candidate cache.
"$PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import sys

root = Path(os.environ["REPO_ROOT"])
sys.path.insert(0, str(root / "scripts"))
from material_calibration import sequence_fingerprint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def value(name: str, cast=float):
    return cast(os.environ[name])

reconstruction = Path(os.environ["RECONSTRUCTION"])
geometry = Path(os.environ["TOOL_GEOMETRY"])
episode = Path(os.environ["EPISODE"])
calibration = Path(os.environ["CALIBRATION"])
simulator = root / "scripts" / "taichi_viscoelastic_mpm_scene.py"
evaluator = root / "scripts" / "evaluate_dynamic_topview_match.py"
particles = reconstruction / "sampled_particles_xyz.npy"
metadata = reconstruction / "reconstruction_metadata.json"

simulator_arguments = {
    "--particles": value("PARTICLES", int),
    "--grid": value("GRID", int),
    "--dt": value("DT"),
    "--substeps-per-frame": value("SUBSTEPS_PER_FRAME", int),
    "--replay-stride": value("REPLAY_STRIDE", int),
    "--replay-max-gap": value("REPLAY_MAX_GAP_S"),
    "--depth-width": value("DEPTH_WIDTH", int),
    "--depth-height": value("DEPTH_HEIGHT", int),
    "--depth-splat-radius": value("DEPTH_SPLAT_RADIUS", int),
    "--depth-pointcloud-max-points": 0,
    "--poisson-ratio": value("POISSON_RATIO"),
    "--viscosity": value("VISCOSITY"),
    "--density": value("DENSITY_KG_M3"),
    "--object-mass-kg": value("DOUGH_MASS_KG"),
    "--gravity": value("GRAVITY"),
    "--floor-y": value("FLOOR_Y"),
    "--floor-friction": value("FLOOR_FRICTION"),
    "--floor-absorption": value("FLOOR_ABSORPTION"),
    "--tool-contact-padding": value("TOOL_CONTACT_PADDING"),
    "--tool-contact-friction": value("TOOL_CONTACT_FRICTION"),
    "--tool-contact-absorption": value("TOOL_CONTACT_ABSORPTION"),
    "--tool-stickiness": value("TOOL_STICKINESS"),
    "--floor-stickiness": value("FLOOR_STICKINESS"),
    "--floor-plastic-damping-band": value("FLOOR_PLASTIC_DAMPING_BAND"),
    "--velocity-damping": value("VELOCITY_DAMPING"),
    "--pure-viscoelastic": True,
    "--plastic-min": value("PLASTIC_MIN"),
    "--plastic-max": value("PLASTIC_MAX"),
    "--plastic-velocity-damping": value("PLASTIC_VELOCITY_DAMPING"),
    "--plastic-affine-damping": value("PLASTIC_AFFINE_DAMPING"),
    "--use-jp": False,
    "--jp-hardening": value("JP_HARDENING"),
    "--jp-min": value("JP_MIN"),
    "--jp-max": value("JP_MAX"),
    "--cpu": True,
    "--initial-particles-fit": "none",
    "--initial-particles-axis-map": "xyz",
    "--initial-particles-scale": 1.0,
    "--initial-particles-offset": [0.0, 0.0, 0.0],
    "--initial-particles-raw-scene-coordinates": False,
    "--initial-particles-seed": value("INITIAL_PARTICLE_SEED", int),
    "--no-publish-dough-center": True,
    "--tool-collision": os.environ["TOOL_COLLISION"],
}
recorded_setup = {
    "dough_mass_kg": value("DOUGH_MASS_KG"),
    "geometry_description": f"Measured tool geometry; collision mode {os.environ['TOOL_COLLISION']}",
    "tool_geometry_fingerprint": sha256(geometry),
}
if os.environ["TOOL_COLLISION"] == "sdf":
    ur_mesh = Path(os.environ["UR_TOOL_MESH"])
    kinova_mesh = Path(os.environ["KINOVA_TOOL_MESH"])
    ur_collision_mesh = Path(os.environ["UR_TOOL_COLLISION_MESH"])
    kinova_collision_mesh = Path(os.environ["KINOVA_TOOL_COLLISION_MESH"])
    collision_manifest = Path(os.environ["TOOL_COLLISION_MANIFEST"])
    simulator_arguments.update({
        "--tool-sdf-resolution": value("TOOL_SDF_RESOLUTION", int),
        "--ur-tool-mesh": str(ur_mesh),
        "--kinova-tool-mesh": str(kinova_mesh),
        "--ur-tool-collision-mesh": str(ur_collision_mesh),
        "--kinova-tool-collision-mesh": str(kinova_collision_mesh),
        "--tool-mesh-scale": value("TOOL_MESH_SCALE"),
    })
    recorded_setup.update({
        "tool_skin_label": os.environ["TOOL_SKIN_LABEL"],
        "ur_visual_mesh_sha256": sha256(ur_mesh),
        "kinova_visual_mesh_sha256": sha256(kinova_mesh),
    })

inputs = {
    "geometry": {"path": str(geometry), "sha256": sha256(geometry)},
    "reconstruction_metadata": {"path": str(metadata), "sha256": sha256(metadata)},
    "initial_particles": {"path": str(particles), "sha256": sha256(particles)},
    "sequence": {"episode_dir": str(episode), "fingerprint": sequence_fingerprint(episode)},
    "calibration": {"path": str(calibration), "sha256": sha256(calibration)},
    "simulator": {"path": str(simulator), "sha256": sha256(simulator)},
    "evaluator": {"path": str(evaluator), "sha256": sha256(evaluator)},
}
if os.environ["TOOL_COLLISION"] == "sdf":
    inputs.update({
        "ur_collision_mesh": {"path": str(ur_collision_mesh), "sha256": sha256(ur_collision_mesh)},
        "kinova_collision_mesh": {"path": str(kinova_collision_mesh), "sha256": sha256(kinova_collision_mesh)},
        "collision_manifest": {"path": str(collision_manifest), "sha256": sha256(collision_manifest)},
    })

manifest = {
    "schema": "taichidough/material-calibration-manifest/v1",
    "name": f"single-episode-{episode.name}",
    "inputs": inputs,
    "fixed_parameters": {
        "recorded_setup": recorded_setup,
        "simulator_arguments": simulator_arguments,
        "evaluator_arguments": {
            "--view": "deformpath_top",
            "--frame-stride": value("REPLAY_STRIDE", int),
            "--pair-tolerance": value("DT"),
            "--cell-size": 0.003,
            "--trim-quantile": 0.005,
        },
    },
    "windows": [
        {"name": "training", "split": "training", "start_frame": value("TRAIN_START_FRAME", int), "end_frame": value("TRAIN_END_FRAME", int), "weight": 1.0},
        {"name": "held-out", "split": "validation", "start_frame": value("VALIDATION_START_FRAME", int), "end_frame": value("VALIDATION_END_FRAME", int), "weight": 1.0},
    ],
    "loss": {
        "weights": {"depth_change": 1.0, "mask_iou": 1.0, "observed_to_simulation_distance": 1.0, "real_coverage": 1.0},
        "depth_scale_m": 0.01,
        "distance_scale_m": 0.01,
        "huber_delta": 1.0,
        "min_common_pixels": 50,
        "min_observed_pixels": 50,
        "min_simulation_pixels": 50,
        "min_observed_points": 50,
        "min_simulation_points": 50,
        "nearest_chunk_size": 1024,
        "require_all_frames": True,
    },
    "search": {
        "default_youngs_modulus_pa": 2000.0,
        "initial_min_pa": value("INITIAL_MIN_PA"),
        "initial_max_pa": value("INITIAL_MAX_PA"),
        "hard_min_pa": value("HARD_MIN_PA"),
        "hard_max_pa": value("HARD_MAX_PA"),
        "coarse_count": value("COARSE_COUNT", int),
        "refinement_rounds": value("REFINEMENT_ROUNDS", int),
        "refinement_subdivisions": value("REFINEMENT_SUBDIVISIONS", int),
        "boundary_expansion_factor": 3.0,
        "max_boundary_expansions": value("MAX_BOUNDARY_EXPANSIONS", int),
        "flat_relative_tolerance": 0.02,
        "flat_absolute_tolerance": 1e-6,
        "flat_minimum_log10_span": 0.15,
        "bootstrap_samples": value("BOOTSTRAP_SAMPLES", int),
        "bootstrap_confidence": 0.95,
        "bootstrap_seed": 0,
    },
    "commands": {
        "python": os.environ["PYTHON"],
        "simulator": [
            "{python}", "{inputs.simulator.path}", "{simulator_arguments}",
            "--output-dir", "{simulation_dir}", "--youngs-modulus", "{youngs_modulus_pa}",
            "--replay-episode", "{inputs.sequence.episode_dir}",
            "--replay-start-frame", "{window.replay_start_frame}", "--replay-end-frame", "{window.end_frame}",
            "--initial-particles", "{inputs.initial_particles.path}",
            "--initial-particles-metadata", "{inputs.reconstruction_metadata.path}",
            "--initial-particles-calibration", "{inputs.calibration.path}",
            "--tool-geometry", "{inputs.geometry.path}",
        ],
        "evaluator": [
            "{python}", "{inputs.evaluator.path}", "{evaluator_arguments}",
            "--episode-dir", "{inputs.sequence.episode_dir}",
            "--taichi-metadata", "{simulation_metadata}", "--calibration", "{inputs.calibration.path}",
            "--output-dir", "{evaluation_dir}",
        ],
    },
    "execution": {
        "cache_dir": str(Path(os.environ["OUTPUT_ROOT"]) / "candidate_cache"),
        "result_path": os.environ["RESULT"],
        "subprocess_timeout_s": value("SUBPROCESS_TIMEOUT_S"),
    },
}
Path(os.environ["MANIFEST"]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(f"Wrote manifest: {os.environ['MANIFEST']}")
PY

"$PYTHON" scripts/calibrate_youngs_modulus.py "$MANIFEST" --validate-only
set +e
"$PYTHON" scripts/calibrate_youngs_modulus.py "$MANIFEST"
calibration_status=$?
set -e
if (( calibration_status != 0 && calibration_status != 2 )); then
    exit "$calibration_status"
fi
"$PYTHON" scripts/visualize_material_calibration.py "$RESULT" --output-dir "$OUTPUT_ROOT/material_report"

printf '\nCalibration result: %s\nMaterial report: %s\nStatic report: %s\n' \
    "$RESULT" "$OUTPUT_ROOT/material_report/material_calibration_report.html" "$STATIC_DIR/report/index.html"
if (( calibration_status == 2 )); then
    printf 'The search completed but needs review; inspect the material report before changing its manifest.\n' >&2
fi
exit "$calibration_status"
