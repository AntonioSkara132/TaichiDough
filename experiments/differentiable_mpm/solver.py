"""Staged MLS-MPM with explicit spectral and generated kernel adjoints.

Constitutive/contact equations are shared by both physics versions. corrected-v1
uses affine-consistent G2P and floor-based padded stencils; legacy-v1 retains the
frozen reference transfer. Discrete decisions are differentiated within the active branch.
"""
import numpy as np
import taichi as ti

from .state import (DEFAULT_PARAMETERS, PARAMETER_NAMES, STATE_NAMES, InvalidStateError,
                    ParticleState, SimulationConfig, SDFData, ToolControl, validate_parameters)
from .spectral import clamp_forward, clamp_vjp, polar_forward, polar_vjp


@ti.data_oriented
class Stepper:
    def __init__(self, config: SimulationConfig, parameters=None, capacity=65, sdf: SDFData = None):
        self.config = config
        self.n = config.n_particles
        self.capacity = int(capacity)
        if self.capacity < 2:
            raise ValueError("State capacity must be at least two")
        self.dtype = ti.f32 if config.precision == "f32" else ti.f64
        if ti.lang.impl.current_cfg().default_fp != self.dtype:
            raise ValueError("Taichi default_fp must match SimulationConfig.precision")
        self.np_dtype = config.numpy_dtype
        self.grid = config.grid
        self.dt = config.dt
        self.dx = 1.0 / self.grid
        self.inv_dx = float(self.grid)
        self.corrected_physics = config.physics_version == "corrected-v1"
        self.allocated_grid = config.allocated_grid
        self.grid_offset = -1 if self.corrected_physics else 0
        self.g2p_affine_scale = 4 * self.inv_dx * (self.inv_dx if self.corrected_physics else 1.0)
        self.use_sdf = config.tool_collision == "sdf"
        self.use_plasticity = config.plasticity == "stretch-clamp"
        if self.use_sdf:
            if sdf is None:
                raise ValueError("SDF collision requires SDFData")
            sdf.validate()
        elif sdf is not None:
            raise ValueError("SDF data was supplied with collision disabled")
        self.x = ti.Vector.field(3, self.dtype, shape=(self.capacity, self.n), needs_grad=True)
        self.v = ti.Vector.field(3, self.dtype, shape=(self.capacity, self.n), needs_grad=True)
        self.C = ti.Matrix.field(3, 3, self.dtype, shape=(self.capacity, self.n), needs_grad=True)
        self.F = ti.Matrix.field(3, 3, self.dtype, shape=(self.capacity, self.n), needs_grad=True)
        self.Jp = ti.field(self.dtype, shape=(self.capacity, self.n), needs_grad=True)
        self.saved_x_adjoint = ti.Vector.field(3, self.dtype, shape=self.n)
        self.saved_v_adjoint = ti.Vector.field(3, self.dtype, shape=self.n)
        self.saved_C_adjoint = ti.Matrix.field(3, 3, self.dtype, shape=self.n)
        self.saved_F_adjoint = ti.Matrix.field(3, 3, self.dtype, shape=self.n)
        self.saved_Jp_adjoint = ti.field(self.dtype, shape=self.n)
        self.parameters = ti.field(self.dtype, shape=len(PARAMETER_NAMES), needs_grad=True)
        self.trial = ti.Matrix.field(3, 3, self.dtype, shape=self.n, needs_grad=True)
        self.corrected = ti.Matrix.field(3, 3, self.dtype, shape=self.n, needs_grad=True)
        self.history = ti.field(self.dtype, shape=self.n, needs_grad=True)
        self.rotation = ti.Matrix.field(3, 3, self.dtype, shape=self.n, needs_grad=True)
        self.affine = ti.Matrix.field(3, 3, self.dtype, shape=self.n, needs_grad=True)
        self.adv_x = ti.Vector.field(3, self.dtype, shape=self.n, needs_grad=True)
        self.adv_v = ti.Vector.field(3, self.dtype, shape=self.n, needs_grad=True)
        self.new_C = ti.Matrix.field(3, 3, self.dtype, shape=self.n, needs_grad=True)
        self.yielded = ti.field(ti.i32, shape=self.n)
        grid_shape, grid_offset = (self.allocated_grid,) * 3, (self.grid_offset,) * 3
        self.grid_m = ti.field(self.dtype, shape=grid_shape, offset=grid_offset, needs_grad=True)
        self.grid_p = ti.Vector.field(3, self.dtype, shape=grid_shape, offset=grid_offset, needs_grad=True)
        self.grid_u = ti.Vector.field(3, self.dtype, shape=grid_shape, offset=grid_offset, needs_grad=True)
        self.grid_v = ti.Vector.field(3, self.dtype, shape=grid_shape, offset=grid_offset, needs_grad=True)
        self.tool_center = ti.Vector.field(3, self.dtype, shape=2)
        self.tool_rotation = ti.Matrix.field(3, 3, self.dtype, shape=2)
        self.tool_velocity = ti.Vector.field(3, self.dtype, shape=2)
        self.tool_omega = ti.Vector.field(3, self.dtype, shape=2)
        self.sdf_resolution = sdf.resolution if self.use_sdf else 2
        self.sdf = ti.field(self.dtype, shape=(2,) + (self.sdf_resolution,) * 3)
        self.sdf_normal = ti.Vector.field(3, self.dtype, shape=(2,) + (self.sdf_resolution,) * 3)
        self.sdf_minimum = ti.Vector.field(3, self.dtype, shape=2)
        self.sdf_spacing = ti.Vector.field(3, self.dtype, shape=2)
        # Yield, grid hit/inward by tool, particle hit/inward by tool, floor hit/band.
        self.counts = ti.field(ti.i32, shape=12)
        self.invalid = ti.field(ti.i32, shape=6)
        if self.use_sdf:
            self.sdf.from_numpy(np.asarray(sdf.distances, dtype=self.np_dtype))
            self.sdf_normal.from_numpy(np.asarray(sdf.gradients, dtype=self.np_dtype))
            self.sdf_minimum.from_numpy(np.asarray(sdf.minimums, dtype=self.np_dtype))
            self.sdf_spacing.from_numpy(np.asarray(sdf.spacings, dtype=self.np_dtype))
        self.set_parameters(DEFAULT_PARAMETERS if parameters is None else parameters)
        self._set_control(ToolControl.stationary())

    def _slot(self, slot, next_slot=False):
        slot = int(slot)
        maximum = self.capacity - (2 if next_slot else 1)
        if not 0 <= slot <= maximum:
            raise IndexError(f"State slot {slot} outside [0,{maximum}]")
        return slot

    def set_parameters(self, parameters):
        values = {name: float(value) for name, value in parameters.items()}
        validate_parameters(values)
        self.parameters.from_numpy(np.array([values[name] for name in PARAMETER_NAMES], dtype=self.np_dtype))
        self._parameter_values = values

    def parameter_gradients(self):
        return dict(zip(PARAMETER_NAMES, self.parameters.grad.to_numpy().astype(float).tolist()))

    def clear_parameter_gradients(self):
        self.parameters.grad.fill(0)

    def clear_state_gradients(self):
        self._clear_state_gradients()

    @ti.kernel
    def _clear_state_gradients(self):
        for t, p in self.Jp:
            self.x.grad[t, p] = ti.Vector.zero(self.dtype, 3)
            self.v.grad[t, p] = ti.Vector.zero(self.dtype, 3)
            self.C.grad[t, p] = ti.Matrix.zero(self.dtype, 3, 3)
            self.F.grad[t, p] = ti.Matrix.zero(self.dtype, 3, 3)
            self.Jp.grad[t, p] = 0

    @ti.kernel
    def _upload(self, slot: ti.i32, destination: ti.template(), values: ti.types.ndarray(), add: ti.i32, rank: ti.template()):
        for p in range(self.n):
            if ti.static(rank == 0):
                if add:
                    destination[slot, p] += values[p]
                else:
                    destination[slot, p] = values[p]
            elif ti.static(rank == 1):
                for i in ti.static(range(3)):
                    if add:
                        destination[slot, p][i] += values[p, i]
                    else:
                        destination[slot, p][i] = values[p, i]
            else:
                for i, j in ti.static(ti.ndrange(3, 3)):
                    if add:
                        destination[slot, p][i, j] += values[p, i, j]
                    else:
                        destination[slot, p][i, j] = values[p, i, j]

    @ti.kernel
    def _download(self, slot: ti.i32, source: ti.template(), values: ti.types.ndarray(), rank: ti.template()):
        for p in range(self.n):
            if ti.static(rank == 0):
                values[p] = source[slot, p]
            elif ti.static(rank == 1):
                for i in ti.static(range(3)):
                    values[p, i] = source[slot, p][i]
            else:
                for i, j in ti.static(ti.ndrange(3, 3)):
                    values[p, i, j] = source[slot, p][i, j]

    def load_state(self, slot, state):
        slot = self._slot(slot)
        state.validate()
        if len(state.x) != self.n:
            raise ValueError("State particle count does not match configuration")
        for name in STATE_NAMES:
            self._upload(slot, getattr(self, name), np.ascontiguousarray(getattr(state, name), dtype=self.np_dtype),
                         0, getattr(state, name).ndim - 1)

    def state(self, slot):
        return self._read_state(self._slot(slot), adjoint=False)

    def _read_state(self, slot, adjoint):
        result = ParticleState.initial(np.empty((self.n, 3), dtype=self.np_dtype), self.np_dtype)
        for name in STATE_NAMES:
            field = getattr(self, name)
            self._download(slot, field.grad if adjoint else field, getattr(result, name), getattr(result, name).ndim - 1)
        return result

    def load_adjoint(self, slot, state, add=False):
        slot = self._slot(slot)
        state.validate()
        if len(state.x) != self.n:
            raise ValueError("Adjoint particle count does not match configuration")
        for name in STATE_NAMES:
            self._upload(slot, getattr(self, name).grad,
                         np.ascontiguousarray(getattr(state, name), dtype=self.np_dtype), int(add), getattr(state, name).ndim - 1)

    def adjoint(self, slot):
        return self._read_state(self._slot(slot), adjoint=True)

    def seed_positions(self, slot, array, weight=1.0):
        values = np.asarray(array, dtype=self.np_dtype)
        if values.shape != (self.n, 3) or not np.isfinite(values).all() or not np.isfinite(weight):
            raise ValueError("Position adjoints must be finite [N,3] values")
        self._upload(self._slot(slot), self.x.grad, np.ascontiguousarray(values * weight), 1, 1)

    def _set_control(self, control):
        control.validate()
        self._upload_control(np.ascontiguousarray(control.poses, dtype=self.np_dtype),
                             np.ascontiguousarray(control.velocities, dtype=self.np_dtype))

    @ti.kernel
    def _upload_control(self, poses: ti.types.ndarray(), velocities: ti.types.ndarray()):
        for k in range(2):
            self.tool_center[k] = ti.Vector([poses[k, 0], poses[k, 1], poses[k, 2]])
            x, y, z, w = poses[k, 3], poses[k, 4], poses[k, 5], poses[k, 6]
            self.tool_rotation[k] = ti.Matrix([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])
            self.tool_velocity[k] = ti.Vector([velocities[k, 0], velocities[k, 1], velocities[k, 2]])
            self.tool_omega[k] = ti.Vector([velocities[k, 3], velocities[k, 4], velocities[k, 5]])

    @ti.func
    def _finite(self, value):
        return not ti.math.isnan(value) and not ti.math.isinf(value)

    @ti.func
    def _stencil_base(self, pos):
        base = (pos * self.inv_dx - 0.5).cast(ti.i32)
        if ti.static(self.corrected_physics):
            base = ti.floor(pos * self.inv_dx - 0.5).cast(ti.i32)
        return base

    @ti.func
    def _safe_stencil(self, pos):
        base = self._stencil_base(pos)
        return (base[0] >= self.grid_offset and base[0] + 2 < self.grid and
                base[1] >= self.grid_offset and base[1] + 2 < self.grid and
                base[2] >= self.grid_offset and base[2] + 2 < self.grid)

    @ti.kernel
    def _reset_diagnostics(self):
        for i in self.invalid:
            self.invalid[i] = self.n
        for i in self.counts:
            self.counts[i] = 0

    @ti.kernel
    def _check_state(self, slot: ti.i32, end: ti.i32):
        for p in range(self.n):
            finite = self._finite(self.Jp[slot, p])
            for i in ti.static(range(3)):
                finite = finite and self._finite(self.x[slot, p][i]) and self._finite(self.v[slot, p][i])
                for j in ti.static(range(3)):
                    finite = finite and self._finite(self.C[slot, p][i, j]) and self._finite(self.F[slot, p][i, j])
            if not finite:
                ti.atomic_min(self.invalid[3 * end], p)
            elif not self._safe_stencil(self.x[slot, p]):
                ti.atomic_min(self.invalid[3 * end + 1], p)

    def _raise_invalid(self):
        values = self.invalid.to_numpy()
        labels = ("input nonfinite", "input stencil", "singular/inverted deformation",
                  "output nonfinite", "output stencil", "reserved")
        for index, particle in enumerate(values):
            if particle < self.n:
                raise InvalidStateError(f"{labels[index]} at particle {int(particle)}")

    @ti.kernel
    def _trial(self, slot: ti.i32):
        for p in range(self.n):
            self.trial[p] = (ti.Matrix.identity(self.dtype, 3) + self.dt * self.C[slot, p]) @ self.F[slot, p]

    @ti.kernel
    def _project(self, slot: ti.i32):
        for p in range(self.n):
            trial = self.trial[p]
            U, sig, V = ti.svd(trial, self.dtype)
            minimum = ti.min(sig[0, 0], ti.min(sig[1, 1], sig[2, 2]))
            if minimum < self.config.min_singular_value or trial.determinant() <= 0 or not self._finite(minimum):
                ti.atomic_min(self.invalid[2], p)
            self.yielded[p] = 0
            if ti.static(self.use_plasticity):
                corrected, yielded = clamp_forward(trial, self.parameters[3], self.parameters[4])
                self.corrected[p] = corrected
                self.yielded[p] = yielded
                if yielded:
                    ti.atomic_add(self.counts[0], 1)
            else:
                self.corrected[p] = trial

    @ti.kernel
    def _project_backward(self, slot: ti.i32):
        for p in range(self.n):
            if ti.static(self.use_plasticity):
                gradient, lower, upper = clamp_vjp(self.trial[p], self.parameters[3], self.parameters[4],
                                                   self.corrected.grad[p])
                self.trial.grad[p] += gradient
                self.parameters.grad[3] += lower
                self.parameters.grad[4] += upper
            else:
                self.trial.grad[p] += self.corrected.grad[p]

    @ti.kernel
    def _history(self, slot: ti.i32):
        for p in range(self.n):
            value = self.Jp[slot, p]
            if ti.static(self.use_plasticity and self.config.use_jp):
                value = ti.min(ti.max(value * self.trial[p].determinant() / self.corrected[p].determinant(),
                                      self.config.jp_min), self.config.jp_max)
            self.history[p] = value

    @ti.kernel
    def _polar(self, slot: ti.i32):
        for p in range(self.n):
            self.rotation[p] = polar_forward(self.corrected[p])

    @ti.kernel
    def _polar_backward(self, slot: ti.i32):
        for p in range(self.n):
            self.corrected.grad[p] += polar_vjp(self.corrected[p], self.rotation.grad[p])

    @ti.kernel
    def _stress(self, slot: ti.i32):
        for p in range(self.n):
            young, nu, viscosity = self.parameters[0], self.parameters[1], self.parameters[2]
            hardening = ti.exp(self.config.jp_hardening * (1.0 - self.history[p]))
            mu = young / (2 * (1 + nu)) * hardening
            la = young * nu / ((1 + nu) * (1 - 2 * nu)) * hardening
            F = self.corrected[p]
            J = F.determinant()
            elastic = 2 * mu * (F - self.rotation[p]) @ F.transpose()
            elastic += ti.Matrix.identity(self.dtype, 3) * la * J * (J - 1)
            viscous = viscosity * (self.C[slot, p] + self.C[slot, p].transpose())
            stress = -self.dt * self.config.particle_volume * 4 * self.inv_dx * self.inv_dx * (elastic + viscous)
            self.affine[p] = stress + self.config.particle_mass * self.C[slot, p]

    @ti.kernel
    def _store_material(self, slot: ti.i32):
        for p in range(self.n):
            self.F[slot + 1, p] = self.corrected[p]
            self.Jp[slot + 1, p] = self.history[p]

    @ti.kernel
    def _clear_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0
            self.grid_p[I] = ti.Vector.zero(self.dtype, 3)
            self.grid_v[I] = ti.Vector.zero(self.dtype, 3)

    @ti.kernel
    def _p2g(self, slot: ti.i32):
        ti.loop_config(serialize=ti.static(self.config.p2g_mode == "serial"))
        for p in range(self.n):
            base = self._stencil_base(self.x[slot, p])
            fx = self.x[slot, p] * self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(self.dtype) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_p[base + offset] += weight * (self.config.particle_mass * self.v[slot, p] + self.affine[p] @ dpos)
                self.grid_m[base + offset] += weight * self.config.particle_mass

    @ti.func
    def _sdf_at(self, tool, local_pos):
        gp = (local_pos - self.sdf_minimum[tool]) / self.sdf_spacing[tool]
        valid = (gp[0] >= 0 and gp[0] <= self.sdf_resolution - 1 and
                 gp[1] >= 0 and gp[1] <= self.sdf_resolution - 1 and
                 gp[2] >= 0 and gp[2] <= self.sdf_resolution - 1)
        distance = ti.cast(1e6, self.dtype)
        gradient = ti.Vector.zero(self.dtype, 3)
        if valid:
            distance = 0
            base = ti.max(0, ti.min(self.sdf_resolution - 2, ti.floor(gp).cast(ti.i32)))
            f = ti.max(0.0, ti.min(1.0, gp - base.cast(self.dtype)))
            for a, b, c in ti.static(ti.ndrange(2, 2, 2)):
                wx = f[0] if a == 1 else 1.0 - f[0]
                wy = f[1] if b == 1 else 1.0 - f[1]
                wz = f[2] if c == 1 else 1.0 - f[2]
                weight = wx * wy * wz
                ix, iy, iz = base[0] + a, base[1] + b, base[2] + c
                distance += weight * self.sdf[tool, ix, iy, iz]
                gradient += weight * self.sdf_normal[tool, ix, iy, iz]
            gradient /= ti.max(gradient.norm(), 1e-6)
        return valid, distance, gradient

    @ti.func
    def _tool_grid_response(self, pos, velocity):
        closest = ti.cast(1e6, self.dtype)
        selected = -1
        normal = ti.Vector.zero(self.dtype, 3)
        collider_v = ti.Vector.zero(self.dtype, 3)
        result = velocity
        for tool in ti.static(range(2)):
            rotation = self.tool_rotation[tool]
            center = self.tool_center[tool]
            valid, distance, local_normal = self._sdf_at(tool, rotation.transpose() @ (pos - center))
            if valid and distance < self.config.tool_contact_padding and distance < closest:
                selected = tool
                closest = distance
                normal = rotation @ local_normal
                collider_v = self.tool_velocity[tool] + self.tool_omega[tool].cross(pos - center)
        if selected >= 0:
            ti.atomic_add(self.counts[1 + selected], 1)
            relative = velocity - collider_v
            vn = relative.dot(normal)
            if vn < 0:
                relative -= normal * vn
                ti.atomic_add(self.counts[3 + selected], 1)
            relative *= self.parameters[5] * (1 - self.config.tool_contact_absorption)
            relative *= 1 - self.config.tool_stickiness
            result = collider_v + relative
        return result

    @ti.kernel
    def _normalize_grid(self):
        for I in ti.grouped(self.grid_m):
            # Truncating floor-adjacent stencils can have negative mass. The
            # reference leaves their raw accumulated momentum unnormalized.
            velocity = self.grid_p[I]
            if self.grid_m[I] > 0:
                velocity = self.grid_p[I] / self.grid_m[I]
            self.grid_u[I] = velocity

    @ti.kernel
    def _normalize_grid_backward(self):
        for I in ti.grouped(self.grid_m):
            momentum_bar = self.grid_u.grad[I]
            if self.grid_m[I] > 0:
                # Equivalent to -p.dot(u_bar)/m**2, without an overflowing
                # reciprocal square at tiny positive occupied grid masses.
                momentum_bar = self.grid_u.grad[I] / self.grid_m[I]
                self.grid_m.grad[I] -= self.grid_u[I].dot(momentum_bar)
            self.grid_p.grad[I] += momentum_bar

    @ti.kernel
    def _grid_operations(self):
        for I in ti.grouped(self.grid_m):
            velocity = self.grid_u[I]
            if self.grid_m[I] > 0:
                velocity[1] += self.dt * self.config.gravity
                pos = I.cast(self.dtype) * self.dx
                if pos[1] <= self.config.floor_y and velocity[1] < 0:
                    velocity[1] *= -self.config.floor_absorption
                    velocity[0] *= self.parameters[6] * (1 - self.config.floor_stickiness)
                    velocity[2] *= self.parameters[6] * (1 - self.config.floor_stickiness)
                    ti.atomic_add(self.counts[9], 1)
                if ti.static(self.use_sdf):
                    velocity = self._tool_grid_response(pos, velocity)
                if I[0] < 3 and velocity[0] < 0:
                    velocity[0] = 0
                if I[0] > self.grid - 3 and velocity[0] > 0:
                    velocity[0] = 0
                if I[1] > self.grid - 3 and velocity[1] > 0:
                    velocity[1] = 0
                if I[2] < 3 and velocity[2] < 0:
                    velocity[2] = 0
                if I[2] > self.grid - 3 and velocity[2] > 0:
                    velocity[2] = 0
            self.grid_v[I] = velocity

    @ti.kernel
    def _g2p(self, slot: ti.i32):
        for p in range(self.n):
            base = self._stencil_base(self.x[slot, p])
            fx = self.x[slot, p] * self.inv_dx - base.cast(self.dtype)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            velocity = ti.Vector.zero(self.dtype, 3)
            affine = ti.Matrix.zero(self.dtype, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(self.dtype) - fx) * self.dx
                g_v = self.grid_v[base + offset]
                weight = w[i][0] * w[j][1] * w[k][2]
                velocity += weight * g_v
                # dpos is in metres: corrected-v1 uses 4/dx**2; legacy-v1 uses 4/dx.
                affine += self.g2p_affine_scale * weight * g_v.outer_product(dpos)
            damped_velocity = velocity * self.config.velocity_damping
            self.adv_v[p] = damped_velocity
            self.adv_x[p] = self.x[slot, p] + self.dt * damped_velocity
            self.new_C[p] = affine

    @ti.func
    def _tool_particle_response(self, pos, velocity):
        result_x, result_v = pos, velocity
        for tool in ti.static(range(2)):
            center = self.tool_center[tool]
            rotation = self.tool_rotation[tool]
            local_pos = rotation.transpose() @ (result_x - center)
            valid, distance, local_normal = self._sdf_at(tool, local_pos)
            if valid and distance < self.config.tool_contact_padding:
                ti.atomic_add(self.counts[5 + tool], 1)
                penetration = self.config.tool_contact_padding - distance + 1e-4
                normal = rotation @ local_normal
                result_x += normal * penetration
                collider_v = self.tool_velocity[tool] + self.tool_omega[tool].cross(result_x - center)
                relative = result_v - collider_v
                vn = relative.dot(normal)
                if vn < 0:
                    relative -= normal * vn
                    ti.atomic_add(self.counts[7 + tool], 1)
                relative *= self.parameters[5] * (1 - self.config.tool_contact_absorption)
                relative *= 1 - self.config.tool_stickiness
                result_v = collider_v + relative
        return result_x, result_v

    @ti.kernel
    def _particle_operations(self, slot: ti.i32):
        for p in range(self.n):
            pos, velocity, affine = self.adv_x[p], self.adv_v[p], self.new_C[p]
            if ti.static(self.use_sdf):
                pos, velocity = self._tool_particle_response(pos, velocity)
            floor_contact = 0
            if pos[1] < self.config.floor_y:
                pos[1] = self.config.floor_y
                if velocity[1] < 0:
                    velocity[1] *= -self.config.floor_absorption
                velocity[0] *= self.parameters[6] * (1 - self.config.floor_stickiness)
                velocity[2] *= self.parameters[6] * (1 - self.config.floor_stickiness)
                floor_contact = 1
                ti.atomic_add(self.counts[10], 1)
            if pos[1] < self.config.floor_y + self.config.floor_plastic_damping_band:
                if floor_contact == 0:
                    if velocity[1] < 0:
                        velocity[1] *= -self.config.floor_absorption
                    velocity[0] *= self.parameters[6] * (1 - self.config.floor_stickiness)
                    velocity[2] *= self.parameters[6] * (1 - self.config.floor_stickiness)
                affine *= self.config.plastic_affine_damping
                ti.atomic_add(self.counts[11], 1)
            if self.yielded[p] == 1:
                velocity *= self.config.plastic_velocity_damping
                affine *= self.config.plastic_affine_damping
            self.x[slot + 1, p] = pos
            self.v[slot + 1, p] = velocity
            self.C[slot + 1, p] = affine

    @ti.kernel
    def _clear_scratch_gradients(self):
        for p in range(self.n):
            self.trial.grad[p] = ti.Matrix.zero(self.dtype, 3, 3)
            self.corrected.grad[p] = ti.Matrix.zero(self.dtype, 3, 3)
            self.history.grad[p] = 0
            self.rotation.grad[p] = ti.Matrix.zero(self.dtype, 3, 3)
            self.affine.grad[p] = ti.Matrix.zero(self.dtype, 3, 3)
            self.adv_x.grad[p] = ti.Vector.zero(self.dtype, 3)
            self.adv_v.grad[p] = ti.Vector.zero(self.dtype, 3)
            self.new_C.grad[p] = ti.Matrix.zero(self.dtype, 3, 3)
        for I in ti.grouped(self.grid_m):
            self.grid_m.grad[I] = 0
            self.grid_p.grad[I] = ti.Vector.zero(self.dtype, 3)
            self.grid_u.grad[I] = ti.Vector.zero(self.dtype, 3)
            self.grid_v.grad[I] = ti.Vector.zero(self.dtype, 3)

    @ti.kernel
    def _preserve_output_adjoint(self, slot: ti.i32, restore: ti.template()):
        for p in range(self.n):
            if ti.static(restore):
                self.x.grad[slot + 1, p] = self.saved_x_adjoint[p]
                self.v.grad[slot + 1, p] = self.saved_v_adjoint[p]
                self.C.grad[slot + 1, p] = self.saved_C_adjoint[p]
                self.F.grad[slot + 1, p] = self.saved_F_adjoint[p]
                self.Jp.grad[slot + 1, p] = self.saved_Jp_adjoint[p]
            else:
                self.saved_x_adjoint[p] = self.x.grad[slot + 1, p]
                self.saved_v_adjoint[p] = self.v.grad[slot + 1, p]
                self.saved_C_adjoint[p] = self.C.grad[slot + 1, p]
                self.saved_F_adjoint[p] = self.F.grad[slot + 1, p]
                self.saved_Jp_adjoint[p] = self.Jp.grad[slot + 1, p]

    def _prepare(self, slot, control):
        self._set_control(control)
        self._reset_diagnostics()
        self._check_state(slot, 0)
        self._raise_invalid()
        self._trial(slot)
        self._project(slot)
        self._raise_invalid()
        self._history(slot)
        self._polar(slot)
        self._stress(slot)
        self._clear_grid()
        self._p2g(slot)
        self._normalize_grid()
        self._grid_operations()
        self._g2p(slot)

    def advance(self, slot, control):
        slot = self._slot(slot, next_slot=True)
        self._prepare(slot, control)
        self._store_material(slot)
        self._particle_operations(slot)
        self._check_state(slot + 1, 1)
        self._raise_invalid()

    def reverse_step(self, slot, control):
        slot = self._slot(slot, next_slot=True)
        self._prepare(slot, control)
        self._preserve_output_adjoint(slot, False)
        self._clear_scratch_gradients()
        self._particle_operations.grad(slot)
        self._g2p.grad(slot)
        self._grid_operations.grad()
        self._normalize_grid_backward()
        self._p2g.grad(slot)
        self._stress.grad(slot)
        self._polar_backward(slot)
        self._store_material.grad(slot)
        self._history.grad(slot)
        self._project_backward(slot)
        self._trial.grad(slot)
        self._preserve_output_adjoint(slot, True)

    def diagnostics(self):
        names = ("yielded", "grid_tool0", "grid_tool1", "grid_inward0", "grid_inward1",
                 "particle_tool0", "particle_tool1", "particle_inward0", "particle_inward1",
                 "grid_floor", "particle_floor", "floor_band")
        return dict(zip(names, self.counts.to_numpy().astype(int).tolist()))
