"""Check the audited G2P formula on an exact affine grid velocity.

This is an isolated algebraic test, not a Taichi rollout or simulator patch.
The source expression is taichi_viscoelastic_mpm_scene.py:1374-1378 at
cb76169436d334c127f165ce5c3f779100bb82d4.
"""

import itertools
import json
from pathlib import Path

import numpy as np


A = np.array([[1.2, -0.4, 0.8], [0.3, 3.0, -0.6], [-0.2, 0.5, -1.0]])
b = np.array([0.12, -0.21, 0.06])
fx = np.array([0.83, 1.18, 0.94])
w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
rows = []
for grid in (24, 48, 96):
    dx = 1.0 / grid
    moment = np.zeros((3, 3))
    mean_velocity = np.zeros(3)
    for i, j, k in itertools.product(range(3), repeat=3):
        dpos = (np.array([i, j, k]) - fx) * dx
        weight = w[i][0] * w[j][1] * w[k][2]
        velocity = b + A @ dpos
        moment += weight * np.outer(velocity, dpos)
        mean_velocity += weight * velocity
    implemented = 4.0 / dx * moment
    affine_preserving = 4.0 / dx**2 * moment
    rows.append({
        "grid": grid,
        "dx": dx,
        "expected_A_yy_per_s": float(A[1, 1]),
        "implemented_C_yy": float(implemented[1, 1]),
        "max_abs_implemented_minus_dx_A": float(np.max(np.abs(implemented - dx * A))),
        "max_abs_affine_preserving_minus_A": float(np.max(np.abs(affine_preserving - A))),
        "max_abs_mean_velocity_minus_b": float(np.max(np.abs(mean_velocity - b))),
    })
result = {"kind": "algebraic_test_not_simulation", "rows": rows}
print(json.dumps(result, indent=2))
Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
