"""Rigid table alignment for metric calibrations without rewriting source tensors."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping

import numpy as np


TABLE_ALIGNED_FRAME = "table-aligned"
GRAVITY_ASSUMPTION = (
    "The measured table normal defines physical up; the scene tilt is treated as "
    "coordinate-calibration error. Gravity remains along negative table Y, rather "
    "than preserving the previous scene's gravity vector."
)


def _normalized_upward_plane(value):
    plane = np.asarray(value, dtype=np.float64)
    if plane.shape != (4,) or not np.isfinite(plane).all():
        raise ValueError("Table plane must contain four finite coefficients")
    length = float(np.linalg.norm(plane[:3]))
    if length <= 1e-12:
        raise ValueError("Table plane normal must be nonzero")
    plane = plane / length
    if plane[1] <= 0:
        raise ValueError("Table plane must have an upward normal with positive Y")
    return plane


def _rigid_matrix(value, name):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0):
        raise ValueError(f"{name} must have an orthonormal rotation")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-7, rtol=0):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def table_alignment_transform(plane_scene, *, translation_xz=(0.0, 0.0)):
    """Map the upward plane n.p+d=0 to Y=0 with minimum-angle rotation.

    The optional X/Z translation can center the recorded motion in the simulation
    domain. It does not change table height. New Y equals old signed distance.
    """
    plane = _normalized_upward_plane(plane_scene)
    translation = np.asarray(translation_xz, dtype=np.float64)
    if translation.shape != (2,) or not np.isfinite(translation).all():
        raise ValueError("translation_xz must contain two finite metric offsets")
    normal = plane[:3]
    cross = np.cross(normal, [0.0, 1.0, 0.0])
    x, y, z = cross
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    rotation = np.eye(3) + skew + (skew @ skew) / (1.0 + normal[1])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [translation[0], plane[3], translation[1]]
    return _rigid_matrix(transform, "table_from_previous_scene")


def compose_table_calibration(document, plane_scene, *, translation_xz=(0.0, 0.0),
                              scene_frame=TABLE_ALIGNED_FRAME, provenance=None):
    """Return a new v2 calibration; source observations and tool poses stay unchanged."""
    if not isinstance(document, Mapping) or document.get("schema") != "taichidough/scene-calibration/v2":
        raise ValueError("Table alignment requires metric scene-calibration/v2")
    if not isinstance(scene_frame, str) or not scene_frame or scene_frame == document.get("scene_frame"):
        raise ValueError("Derived scene_frame must name a different nonempty coordinate frame")
    if not document.get("source_frame") or not document.get("scene_frame"):
        raise ValueError("Source calibration must name source_frame and scene_frame")
    if not isinstance(document.get("camera"), Mapping):
        raise ValueError("Source calibration must contain a camera object")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("Additional provenance must be an object")
    transform = table_alignment_transform(plane_scene, translation_xz=translation_xz)
    old_source = _rigid_matrix(document["scene_from_source"], "scene_from_source")
    old_camera = _rigid_matrix(document["scene_from_camera"], "scene_from_camera")
    result = copy.deepcopy(dict(document))
    result["name"] = str(document.get("name", "scene")) + "-table-aligned"
    result["scene_frame"] = scene_frame
    result["scene_from_source"] = (transform @ old_source).tolist()
    result["scene_from_camera"] = (transform @ old_camera).tolist()
    result["floor_plane_scene"] = [0.0, 1.0, 0.0, 0.0]
    if "scene_from_camera" in result["camera"]:
        result["camera"]["scene_from_camera"] = copy.deepcopy(result["scene_from_camera"])
    if "translation" in result:
        result["translation"] = np.asarray(result["scene_from_source"])[:3, 3].tolist()
    result.pop("fingerprint", None)
    original_json = json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    result["provenance"] = {
        "method": "rigid-table-plane-alignment",
        "source_calibration_provenance": copy.deepcopy(document.get("provenance", {})),
        "source_calibration_fingerprint": hashlib.sha256(original_json.encode("utf-8")).hexdigest(),
        "table_alignment": {
            "previous_scene_frame": document["scene_frame"],
            "estimated_plane_previous_scene": _normalized_upward_plane(plane_scene).tolist(),
            "table_from_previous_scene": transform.tolist(),
            "translation_xz_m": np.asarray(translation_xz, dtype=float).tolist(),
            "source_tensors_transformed": False,
            "gravity_assumption": GRAVITY_ASSUMPTION,
        },
    }
    if provenance:
        result["provenance"]["derivation_inputs"] = copy.deepcopy(dict(provenance))
    return result
