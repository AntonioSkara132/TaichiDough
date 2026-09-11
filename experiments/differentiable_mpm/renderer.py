"""Differentiable local splatting through a calibrated metric camera.

The depth and occupancy estimates are ratios of front-biased moments. A minimum
per-pixel depth is used only to rescale exponentials; it cancels from both ratios.
Its derivative is therefore unnecessary, including when the minimum changes.
Local footprint membership and clipping at the image bounds remain piecewise.
"""
from dataclasses import dataclass

import numpy as np
import taichi as ti

from .state import InvalidStateError


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    camera_from_scene: np.ndarray
    near_m: float = 1e-4
    far_m: float = 1e6

    def __post_init__(self):
        transform = np.asarray(self.camera_from_scene, dtype=np.float64).copy()
        if not isinstance(self.width, int) or not isinstance(self.height, int) or min(self.width, self.height) < 2:
            raise ValueError("Camera width and height must be integers >= 2")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy, self.near_m, self.far_m]).all():
            raise ValueError("Nonfinite camera parameters")
        if min(self.fx, self.fy, self.near_m) <= 0 or self.far_m <= self.near_m:
            raise ValueError("Focal lengths must be positive and require 0 < near_m < far_m")
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError("camera_from_scene must be a finite 4x4 rigid transform")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
            raise ValueError("Camera rotation must be proper and orthonormal")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError("Invalid homogeneous camera transform")
        transform.setflags(write=False)
        object.__setattr__(self, "camera_from_scene", transform)

    def as_dict(self):
        return {"width": self.width, "height": self.height, "fx": self.fx,
                "fy": self.fy, "cx": self.cx, "cy": self.cy, "near_m": self.near_m, "far_m": self.far_m,
                "camera_from_scene": self.camera_from_scene.tolist()}

    def scaled(self, width, height):
        sx, sy = width / self.width, height / self.height
        return Camera(width, height, self.fx * sx, self.fy * sy,
                      self.cx * sx, self.cy * sy, self.camera_from_scene, self.near_m, self.far_m)

    def optical_points(self, depth, valid):
        depth = np.asarray(depth)
        valid = np.asarray(valid, dtype=bool)
        if depth.shape != (self.height, self.width) or valid.shape != depth.shape:
            raise ValueError("Depth and mask must match the camera dimensions")
        row, column = np.nonzero(valid)
        z = depth[row, column]
        return np.column_stack(((column + 0.5 - self.cx) * z / self.fx,
                                (row + 0.5 - self.cy) * z / self.fy, z))


@dataclass
class RenderResult:
    depth: np.ndarray
    coverage: np.ndarray
    valid: np.ndarray
    optical_positions: np.ndarray
    projected_positions: np.ndarray
    active_particles: np.ndarray
    front_depth: np.ndarray


@ti.data_oriented
class LocalSplatRenderer:
    """Local compact-polynomial splats with generated Taichi reverse kernels.

    Each footprint weight is the product of (1 - (offset/radius)^2)^2 on
    its two axes, and is zero outside the radius. Both value and first
    derivative vanish at the support boundary.

    Depth is sum(q*z)/sum(q), q = footprint*exp(-(z-front)/temperature).
    Coverage is 1-exp(-opacity_gain*sum(q*footprint)/sum(q)). This normalized
    front-footprint estimate prevents duplicate hidden particles from increasing
    coverage merely by increasing particle count. It is a training renderer,
    not the working simulator's hard z-buffer evaluator.
    """

    def __init__(self, camera, n_particles, footprint_radius=2,
                 visibility_temperature_m=0.005, opacity_gain=8.0, precision="f32"):
        if n_particles < 1:
            raise ValueError("The renderer needs at least one particle")
        if not isinstance(footprint_radius, int) or not 1 <= footprint_radius <= 8:
            raise ValueError("footprint_radius must be an integer in [1,8]")
        if not np.isfinite([visibility_temperature_m, opacity_gain]).all() or min(visibility_temperature_m, opacity_gain) <= 0:
            raise ValueError("Visibility temperature and opacity gain must be positive")
        if precision not in {"f32", "f64"}:
            raise ValueError("precision must be f32 or f64")
        self.camera = camera
        self.n_particles = int(n_particles)
        self.radius = int(footprint_radius)
        self.temperature = float(visibility_temperature_m)
        self.opacity_gain = float(opacity_gain)
        self.fp = ti.f32 if precision == "f32" else ti.f64
        self.numpy_dtype = np.float32 if precision == "f32" else np.float64
        self.positions = ti.Vector.field(3, self.fp, self.n_particles, needs_grad=True)
        self.optical = ti.Vector.field(3, self.fp, self.n_particles, needs_grad=True)
        self.projected = ti.Vector.field(2, self.fp, self.n_particles, needs_grad=True)
        self.active = ti.field(ti.i32, self.n_particles)
        self.rotation = ti.Matrix.field(3, 3, self.fp, shape=())
        self.translation = ti.Vector.field(3, self.fp, shape=())
        self.rotation[None] = camera.camera_from_scene[:3, :3]
        self.translation[None] = camera.camera_from_scene[:3, 3]
        dimensions = (camera.height, camera.width)
        self.front = ti.field(self.fp, dimensions)
        self.density = ti.field(self.fp, dimensions, needs_grad=True)
        self.depth_moment = ti.field(self.fp, dimensions, needs_grad=True)
        self.footprint_moment = ti.field(self.fp, dimensions, needs_grad=True)
        self.depth = ti.field(self.fp, dimensions, needs_grad=True)
        self.coverage = ti.field(self.fp, dimensions, needs_grad=True)
        self.valid = ti.field(ti.i32, dimensions)
        self.differentiable_fields = (self.positions, self.optical, self.projected,
                                     self.density, self.depth_moment, self.footprint_moment,
                                     self.depth, self.coverage)

    @ti.func
    def _footprint(self, uv, column, row):
        du = (uv[0] - (ti.cast(column, self.fp) + 0.5)) / self.radius
        dv = (uv[1] - (ti.cast(row, self.fp) + 0.5)) / self.radius
        weight = ti.cast(0.0, self.fp)
        if ti.abs(du) < 1.0 and ti.abs(dv) < 1.0:
            weight = (1.0 - du * du) ** 2 * (1.0 - dv * dv) ** 2
        return weight

    @ti.kernel
    def _clear(self):
        for row, column in self.density:
            self.front[row, column] = ti.math.inf
            self.density[row, column] = 0.0
            self.depth_moment[row, column] = 0.0
            self.footprint_moment[row, column] = 0.0
            self.depth[row, column] = 0.0
            self.coverage[row, column] = 0.0
            self.valid[row, column] = 0

    @ti.kernel
    def _project(self):
        for p in self.positions:
            optical = self.rotation[None] @ self.positions[p] + self.translation[None]
            self.optical[p] = optical
            uv = ti.Vector.zero(self.fp, 2)
            active = 0
            if self.camera.near_m < optical[2] < self.camera.far_m:
                uv = ti.Vector([self.camera.fx * optical[0] / optical[2] + self.camera.cx,
                                self.camera.fy * optical[1] / optical[2] + self.camera.cy])
                if 0.0 <= uv[0] < self.camera.width and 0.0 <= uv[1] < self.camera.height:
                    active = 1
            self.projected[p] = uv
            self.active[p] = active

    @ti.kernel
    def _find_front(self):
        for p in self.positions:
            if self.active[p] == 1:
                uv = self.projected[p]
                base = ti.cast(ti.floor(uv - 0.5), ti.i32)
                for dr, dc in ti.static(ti.ndrange((-self.radius + 1, self.radius + 1),
                                                  (-self.radius + 1, self.radius + 1))):
                    row, column = base[1] + dr, base[0] + dc
                    if 0 <= row < self.camera.height and 0 <= column < self.camera.width:
                        if self._footprint(uv, column, row) > 0.0:
                            ti.atomic_min(self.front[row, column], self.optical[p][2])

    @ti.kernel
    def _splat(self):
        for p in self.positions:
            if self.active[p] == 1:
                uv = self.projected[p]
                z = self.optical[p][2]
                base = ti.cast(ti.floor(uv - 0.5), ti.i32)
                for dr, dc in ti.static(ti.ndrange((-self.radius + 1, self.radius + 1),
                                                  (-self.radius + 1, self.radius + 1))):
                    row, column = base[1] + dr, base[0] + dc
                    if 0 <= row < self.camera.height and 0 <= column < self.camera.width:
                        weight = self._footprint(uv, column, row)
                        if weight > 0.0:
                            q = weight * ti.exp(-(z - self.front[row, column]) / self.temperature)
                            self.density[row, column] += q
                            self.depth_moment[row, column] += q * z
                            self.footprint_moment[row, column] += q * weight

    @ti.kernel
    def _finish(self):
        for row, column in self.density:
            if self.density[row, column] > 0.0:
                self.depth[row, column] = self.depth_moment[row, column] / self.density[row, column]
                mean_footprint = self.footprint_moment[row, column] / self.density[row, column]
                self.coverage[row, column] = 1.0 - ti.exp(-self.opacity_gain * mean_footprint)
                self.valid[row, column] = 1

    def forward(self, positions):
        positions = np.asarray(positions, dtype=self.numpy_dtype)
        if positions.shape != (self.n_particles, 3) or not np.isfinite(positions).all():
            raise InvalidStateError("Renderer positions must be finite and match particle count")
        self.positions.from_numpy(np.ascontiguousarray(positions))
        self._clear()
        self._project()
        self._find_front()
        self._splat()
        self._finish()
        return self.result()

    def result(self):
        return RenderResult(self.depth.to_numpy(), self.coverage.to_numpy(),
                            self.valid.to_numpy().astype(bool), self.optical.to_numpy(),
                            self.projected.to_numpy(), self.active.to_numpy().astype(bool),
                            self.front.to_numpy())

    def clear_gradients(self):
        for field in self.differentiable_fields:
            field.grad.fill(0)

    def reverse(self):
        """Propagate seeded depth/coverage/optical adjoints into scene positions."""
        self._finish.grad()
        self._splat.grad()
        self._project.grad()
        return self.positions.grad.to_numpy()
