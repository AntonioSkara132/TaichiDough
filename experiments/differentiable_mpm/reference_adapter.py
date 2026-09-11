"""Load and execute the byte-preserved simulator without changing its source files."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
from typing import Mapping

import numpy as np

from .state import ParticleState, SDFData, SimulationConfig, STATE_NAMES, ToolControl, validate_parameters


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[1]
_IMPORT_LOCK = threading.RLock()
_MODULES: dict[tuple[Path, Path, str], SimpleNamespace] = {}
_REFERENCE_POLICY = ContextVar("taichidough_reference_policy", default="strict")


@contextmanager
def reference_policy(policy: str):
    """Explicitly allow verified snapshot baselines despite later live-source edits.

    Strict remains the default. Frozen permits live-source drift only when that
    source has an intact recorded snapshot; other live helpers still must match.
    The previous policy is restored on normal exit and on exceptions.
    """
    if policy not in {"strict", "frozen"}:
        raise ValueError("Reference policy must be strict or frozen")
    token = _REFERENCE_POLICY.set(policy)
    try:
        yield policy
    finally:
        _REFERENCE_POLICY.reset(token)


def current_reference_policy() -> str:
    """Return the explicit policy to propagate to a child-process command."""
    return _REFERENCE_POLICY.get()


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key/source in reference records: {key}")
        result[key] = value
    return result


def _supplemental_snapshots(experiment: Path, records: list[dict]):
    path = experiment / "reference_snapshots.json"
    if not path.exists():
        return {}, None
    raw = path.read_bytes()
    document = json.loads(raw, object_pairs_hook=_unique_json_object)
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("Unsupported supplemental snapshot schema")
    if set(document) - {"schema_version", "snapshots", "recovered_from"}:
        raise ValueError("Unknown supplemental snapshot fields")
    mapping = document.get("snapshots")
    if not isinstance(mapping, dict):
        raise ValueError("Supplemental snapshots must map recorded source names to snapshot paths")
    known = {row["source"] for row in records}
    declared = {row["source"] for row in records if "snapshot" in row}
    used_paths = {_relative_file(experiment, row["snapshot"]) for row in records if "snapshot" in row}
    for source, snapshot in mapping.items():
        if source not in known:
            raise ValueError(f"Unlisted supplemental snapshot source: {source}")
        if source in declared:
            raise ValueError(f"Duplicate snapshot source: {source}")
        if not isinstance(snapshot, str):
            raise ValueError("A supplemental snapshot value must be a relative file path, not a new expected hash")
        resolved = _relative_file(experiment, snapshot)
        if not resolved.is_relative_to((experiment / "reference").resolve()):
            raise ValueError("Supplemental snapshots must be inside the experiment reference directory")
        if resolved in used_paths:
            raise ValueError(f"Duplicate snapshot path: {snapshot}")
        used_paths.add(resolved)
    return mapping, hashlib.sha256(raw).hexdigest()


def _relative_file(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Reference manifest requires a relative file path: {value!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Reference path escapes its directory: {value!r}")
    return path


def verify_reference(
    repo_root: Path | None = None,
    experiment_root: Path | None = None,
) -> dict:
    """Verify the recorded baseline and report any explicitly permitted live drift."""
    root = Path(repo_root or REPOSITORY_ROOT).resolve()
    experiment = Path(experiment_root or EXPERIMENT_ROOT).resolve()
    policy = _REFERENCE_POLICY.get()
    manifest_path = experiment / "reference_manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes, object_pairs_hook=_unique_json_object)
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("files"), list):
        raise ValueError("Unsupported reference manifest")
    if not manifest["files"]:
        raise ValueError("Reference manifest has no source files")
    supplement, supplement_hash = _supplemental_snapshots(experiment, manifest["files"])
    checked = []
    seen = set()
    # Verify snapshots before considering any exception for a changed live source.
    for record in manifest["files"]:
        source_name = record["source"]
        expected = record["sha256"]
        if source_name in seen:
            raise ValueError(f"Duplicate reference source: {source_name}")
        seen.add(source_name)
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError(f"Invalid recorded SHA-256 for {source_name}")
        _relative_file(root, source_name)
        verified = {"source": source_name, "sha256": expected, "expected_sha256": expected}
        snapshot_name = record.get("snapshot", supplement.get(source_name))
        if snapshot_name is not None:
            snapshot = _relative_file(experiment, snapshot_name)
            if not snapshot.is_file():
                raise ValueError(f"Frozen snapshot is missing or not a file: {snapshot_name}")
            snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            if snapshot_hash != expected:
                raise ValueError(f"Frozen snapshot differs from preserved reference: {snapshot_name}")
            verified.update(snapshot=snapshot_name, snapshot_sha256=snapshot_hash,
                            snapshot_origin="manifest" if "snapshot" in record else "supplement")
        checked.append(verified)
    expected_snapshots = {
        "reference/taichi_viscoelastic_mpm_scene.py",
        "reference/deformpath_dynamics.py",
        "reference/deformpath_topview.py",
    }
    actual_snapshots = [record["snapshot"] for record in checked if record.get("snapshot_origin") == "manifest"]
    if len(actual_snapshots) != 3 or set(actual_snapshots) != expected_snapshots:
        raise ValueError("Reference manifest must record all three frozen helper modules")
    changed = []
    for verified in checked:
        source_name = verified["source"]
        source = _relative_file(root, source_name)
        if source.exists() and not source.is_file():
            raise ValueError(f"Working source is not a file: {source_name}")
        exists = source.is_file()
        actual = hashlib.sha256(source.read_bytes()).hexdigest() if exists else None
        unchanged = actual == verified["expected_sha256"]
        verified.update(actual_sha256=actual, original_exists=exists, original_unchanged=unchanged)
        if not unchanged:
            if policy == "strict" or "snapshot" not in verified:
                reason = "differs from preserved reference" if exists else "is missing"
                raise ValueError(f"Working source {reason}: {source_name}")
            changed.append({key: verified[key] for key in (
                "source", "expected_sha256", "actual_sha256", "original_exists", "snapshot", "snapshot_sha256"
            )})
    return {"valid": True, "policy": policy, "repository_root": str(root), "files": checked,
            "originals_unchanged": not changed, "changed_originals": changed,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "supplemental_snapshots_sha256": supplement_hash}


def verified_reference_path(source_name: str, repo_root: Path | None = None,
                            experiment_root: Path | None = None) -> Path:
    """Return executable baseline bytes, never a drifted live helper."""
    root = Path(repo_root or REPOSITORY_ROOT).resolve()
    experiment = Path(experiment_root or EXPERIMENT_ROOT).resolve()
    verification = verify_reference(root, experiment)
    for row in verification["files"]:
        if row["source"] == source_name:
            return _relative_file(experiment, row["snapshot"]) if "snapshot" in row else _relative_file(root, source_name)
    raise ValueError(f"Unlisted reference source: {source_name}")


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@contextmanager
def _temporary_aliases(modules: Mapping[str, ModuleType]):
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in modules}
    sys.modules.update(modules)
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def get_reference_modules(repo_root: Path | None = None, experiment_root: Path | None = None) -> SimpleNamespace:
    """Load verified baseline modules and bind their imports to the same baseline."""
    root = Path(repo_root or REPOSITORY_ROOT).resolve()
    experiment = Path(experiment_root or EXPERIMENT_ROOT).resolve()
    verification = verify_reference(root, experiment)
    paths = {row["source"]: (_relative_file(experiment, row["snapshot"]) if "snapshot" in row
                             else _relative_file(root, row["source"])) for row in verification["files"]}
    identity = json.dumps({"manifest": verification["manifest_sha256"],
                           "supplement": verification["supplemental_snapshots_sha256"],
                           "paths": {name: str(path) for name, path in paths.items()}}, sort_keys=True)
    signature = hashlib.sha256(identity.encode()).hexdigest()
    cache_key = (root, experiment, signature)
    with _IMPORT_LOCK:
        if cache_key in _MODULES:
            return _MODULES[cache_key]
        suffix = hashlib.sha256((str(root) + str(experiment) + signature).encode()).hexdigest()[:16]
        package_name = f"_taichidough_frozen_reference_{suffix}"
        package = ModuleType(package_name)
        package.__path__ = [str(experiment / "reference")]
        sys.modules[package_name] = package

        def load(name):
            return _load_module(f"{package_name}.{name}", paths[f"scripts/{name}.py"])

        topview = load("deformpath_topview")
        with _temporary_aliases({"deformpath_topview": topview}):
            dynamics = load("deformpath_dynamics")
            with _temporary_aliases({"deformpath_dynamics": dynamics}):
                simulator = load("taichi_viscoelastic_mpm_scene")
                material = load("material_calibration")
                with _temporary_aliases({"material_calibration": material}):
                    calibrator = load("calibrate_youngs_modulus")
                    evaluator = load("evaluate_dynamic_topview_match")
        simulator.PROJECT_ROOT = root
        simulator.OUTPUT_DIR = experiment / "runs" / "reference"
        if simulator.apply_calibration is not topview.apply_calibration or dynamics.apply_calibration is not topview.apply_calibration:
            raise RuntimeError("Reference geometry modules imported a different calibration helper")
        if evaluator.apply_calibration is not topview.apply_calibration or evaluator.ToolReplay is not dynamics.ToolReplay:
            raise RuntimeError("Reference evaluator imported different geometry/replay helpers")
        if calibrator.partial_view_loss is not material.partial_view_loss:
            raise RuntimeError("Reference calibration imported a different material loss")
        modules = SimpleNamespace(simulator=simulator, dynamics=dynamics, topview=topview,
                                  material_calibration=material, calibrate_youngs_modulus=calibrator,
                                  evaluate_dynamic_topview_match=evaluator, source_paths=paths)
        _MODULES[cache_key] = modules
        return modules


def load_reference(repo_root: Path | None = None, experiment_root: Path | None = None) -> ModuleType:
    """Return the baseline simulator with corrected runtime asset/output paths."""
    return get_reference_modules(repo_root, experiment_root).simulator


def run_reference_evaluator(argv, import_report=None, repo_root=None, experiment_root=None):
    """Run the preserved evaluator with baseline aliases held during lazy imports."""
    modules = get_reference_modules(repo_root, experiment_root)
    aliases = {
        "deformpath_topview": modules.topview,
        "deformpath_dynamics": modules.dynamics,
        "taichi_viscoelastic_mpm_scene": modules.simulator,
        "material_calibration": modules.material_calibration,
        "calibrate_youngs_modulus": modules.calibrate_youngs_modulus,
        "evaluate_dynamic_topview_match": modules.evaluate_dynamic_topview_match,
    }
    with _IMPORT_LOCK, _temporary_aliases(aliases):
        if import_report is not None:
            report = {
                "reference_policy": current_reference_policy(),
                "simulator_project_root": str(modules.simulator.PROJECT_ROOT),
                "modules": {name: {"path": str(Path(module.__file__).resolve()),
                                   "sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
                            for name, module in aliases.items()},
            }
            with Path(import_report).open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
        previous_argv = sys.argv
        sys.argv = [modules.evaluate_dynamic_topview_match.__file__, *argv]
        try:
            return modules.evaluate_dynamic_topview_match.main()
        finally:
            sys.argv = previous_argv


def reference_arguments(config: SimulationConfig, parameters: Mapping[str, float]) -> SimpleNamespace:
    """Map every forward-physics setting to the frozen build_sim argument names."""
    validate_parameters(parameters)
    if config.precision != "f32":
        raise ValueError("The preserved simulator is float32; reference parity requires precision=f32")
    values = asdict(config)
    values.update(parameters)
    values.update(
        particles=config.n_particles,
        density=config.particle_mass / config.particle_volume,
        mass_properties={
            "particle_mass_kg": config.particle_mass,
            "particle_volume_m3": config.particle_volume,
        },
        pure_viscoelastic=config.plasticity == "none",
        tool_contact_friction=parameters["tool_retention"],
        floor_friction=parameters["floor_retention"],
        tool_close_time=1.0,
        tool_motion_start=0.0,
        ros_control=False,
        replay_episode="frozen-reference-recorded-controls",
        tool_half_extents=(0.05, 0.05, 0.05),
        record_sdf_contact_diagnostics=True,
    )
    return SimpleNamespace(**values)


def reference_sdf_fields(data: SDFData) -> SimpleNamespace:
    """Upload the same fixed distance and precomputed-normal volumes as the new solver."""
    import taichi as ti

    data.validate()
    r = data.resolution
    sdf = ti.field(ti.f32, shape=(2, r, r, r))
    gradients = ti.Vector.field(3, ti.f32, shape=(2, r, r, r))
    minimums = ti.Vector.field(3, ti.f32, shape=2)
    spacings = ti.Vector.field(3, ti.f32, shape=2)
    for field, array in (
        (sdf, data.distances), (gradients, data.gradients),
        (minimums, data.minimums), (spacings, data.spacings),
    ):
        field.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    return SimpleNamespace(sdf=sdf, gradients=gradients, minimums=minimums,
                           spacings=spacings, resolution=r)


class ReferenceStepper:
    """Forward-only reference runner; initialize Taichi in its dedicated process first."""

    def __init__(
        self, config: SimulationConfig, parameters: Mapping[str, float],
        sdf: SDFData | None = None, repo_root: Path | None = None,
    ):
        import taichi as ti

        if ti.lang.impl.get_runtime().prog is None:
            raise RuntimeError("Initialize Taichi explicitly before constructing ReferenceStepper")
        self.config = config
        module = load_reference(repo_root)
        if config.tool_collision == "sdf" and sdf is None:
            raise ValueError("Reference SDF collision requires SDFData")
        collision = reference_sdf_fields(sdf) if sdf is not None else None
        result = module.build_sim(reference_arguments(config, parameters), collision, True)
        _x, _tool_x, initialize, self._substep, _visuals, self._set_tools, self._reset_contact, self._contact = result
        closure = inspect.getclosurevars(self._substep).nonlocals
        missing = set(STATE_NAMES) - set(closure)
        if missing:
            raise RuntimeError(f"Frozen substep no longer exposes required state fields: {sorted(missing)}")
        self.fields = {name: closure[name] for name in STATE_NAMES}
        initialize()
        self._loaded = False

    def load_state(self, state: ParticleState):
        state.validate()
        if len(state.x) != self.config.n_particles:
            raise ValueError("Reference state particle count differs from configuration")
        for name, array in state.arrays().items():
            self.fields[name].from_numpy(np.ascontiguousarray(array, dtype=np.float32))
        self._loaded = True

    def state(self) -> ParticleState:
        if not self._loaded:
            raise RuntimeError("Load the explicit initial state before reading or stepping")
        return ParticleState(**{name: field.to_numpy() for name, field in self.fields.items()})

    def advance(self, control: ToolControl) -> list[dict]:
        if not self._loaded:
            raise RuntimeError("Load the explicit initial state before reading or stepping")
        control.validate()
        self._set_tools(np.ascontiguousarray(control.poses, dtype=np.float32),
                        np.ascontiguousarray(control.velocities, dtype=np.float32))
        self._reset_contact()
        self._substep(float(control.time))
        return self._contact()


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Execute the verified baseline evaluator")
    parser.add_argument("--reference-policy", choices=("strict", "frozen"), default="strict")
    parser.add_argument("--import-report", type=Path)
    parser.add_argument("evaluator_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    forwarded = args.evaluator_args
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    with reference_policy(args.reference_policy):
        return run_reference_evaluator(forwarded, import_report=args.import_report)


if __name__ == "__main__":
    raise SystemExit(main())
