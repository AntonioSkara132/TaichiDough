import argparse
import json
import socket
import struct
import time as wall_time
from pathlib import Path

import numpy as np
import taichi as ti
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "taichi_mpm_camera_views"
SCENE_CENTER = [0.5, 0.3, 0.5]
TOOL_Y = 0.32
TOOL_Z_OFFSET = 0.24
TOOL_TRAVEL = 0.28
DOUGH_RADIUS = [0.07, 0.026, 0.06]
PARTICLE_RENDER_RADIUS = 0.006

# Visual mesh origins from rf_lab_setup/urdf/rf_lab.urdf.xacro.  /tool_poses
# already reports the corresponding *_spathla_frame transforms, so the fixed
# robot-to-tool joints must not be applied a second time.
UR_TOOL_VISUAL_ORIGIN = np.array([-0.002395874, -0.017992075, -0.019913439], dtype=np.float32)
UR_TOOL_VISUAL_RPY = np.array([4.5910, 1.379415965, -1.740698498], dtype=np.float32)
KINOVA_TOOL_VISUAL_ORIGIN = np.array([-0.02345833, -0.02261066, -0.01297941], dtype=np.float32)
KINOVA_TOOL_VISUAL_RPY = np.array([3.12897712, 0.06996522, -3.11041673], dtype=np.float32)

TOOL_INITIAL_POSES = np.array(
    [
        [SCENE_CENTER[0], TOOL_Y, SCENE_CENTER[2] + TOOL_Z_OFFSET, 0.0, 0.0, 0.0, 1.0],
        [SCENE_CENTER[0], TOOL_Y, SCENE_CENTER[2] - TOOL_Z_OFFSET, 0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def find_tool_mesh(filename: str, override: Path | None = None) -> Path:
    """Use explicit meshes, the sourced ROS package, or bundled standalone assets."""
    if override is not None:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Tool mesh does not exist: {path}")
        return path
    try:
        from ament_index_python.packages import get_package_share_directory, PackageNotFoundError
    except ImportError:
        pass
    else:
        try:
            installed = Path(get_package_share_directory("ur_dual_bringup")) / "meshes" / filename
        except PackageNotFoundError:
            pass
        else:
            if installed.is_file():
                return installed
    bundled = PROJECT_ROOT / "meshes" / filename
    if not bundled.is_file():
        raise FileNotFoundError(f"Tool mesh missing: {bundled}; provide --ur-tool-mesh/--kinova-tool-mesh")
    return bundled


def load_binary_stl(path: Path, scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Load an STL as an unindexed triangle mesh for Taichi's scene renderer."""
    raw = path.read_bytes()
    if len(raw) < 84:
        raise ValueError(f"{path} is too short to be a binary STL file")
    triangle_count = struct.unpack_from("<I", raw, 80)[0]
    expected_size = 84 + triangle_count * 50
    if len(raw) != expected_size:
        raise ValueError(
            f"{path} is not a supported binary STL file "
            f"(expected {expected_size} bytes, found {len(raw)})"
        )
    triangle_dtype = np.dtype(
        [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
    )
    triangles = np.frombuffer(raw, dtype=triangle_dtype, count=triangle_count, offset=84)
    vertices = np.array(triangles["vertices"].reshape(-1, 3) * scale, dtype=np.float32)
    indices = np.arange(vertices.shape[0], dtype=np.int32)
    return vertices, indices


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def transformed_tool_mesh(
    vertices: np.ndarray,
    pose: np.ndarray,
    visual_origin: np.ndarray,
    visual_rotation: np.ndarray,
) -> np.ndarray:
    """Apply mesh-to-link URDF origin, then link-to-scene tool pose."""
    local_vertices = vertices @ visual_rotation.T + visual_origin
    return local_vertices @ quaternion_to_matrix(pose[3:]).T + pose[:3]


CAMERA_VIEWS = {
    "front_dough": {
        "position": [0.5, 0.38, 1.25],
        "lookAt": SCENE_CENTER,
        "fieldOfView": 57.0,
        "zNear": 0.01,
        "zFar": 2.0,
    },
    "top_dough": {
        "position": [0.5, 0.75, 0.5],
        "lookAt": SCENE_CENTER,
        "fieldOfView": 57.0,
        "zNear": 0.01,
        "zFar": 2.0,
    },
    "side_dough": {
        "position": [1.25, 0.36, 0.5],
        "lookAt": SCENE_CENTER,
        "fieldOfView": 57.0,
        "zNear": 0.01,
        "zFar": 2.0,
    },
}


def build_sim(args):
    dim = 3
    n_particles = args.particles
    n_grid = args.grid
    tool_vis_resolution = 14
    tool_vis_count = 2 * tool_vis_resolution * tool_vis_resolution * tool_vis_resolution
    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    dt = args.dt

    p_vol = (dx * 0.5) ** dim
    p_rho = args.density
    p_mass = p_vol * p_rho
    E = args.youngs_modulus
    nu = args.poisson_ratio
    mu_0 = E / (2 * (1 + nu))
    lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))
    viscosity = args.viscosity
    gravity = args.gravity
    floor_y = args.floor_y
    tool_close_time = args.tool_close_time
    tool_motion_start = args.tool_motion_start
    tool_contact_padding = args.tool_contact_padding
    tool_contact_friction = args.tool_contact_friction
    tool_contact_absorption = args.tool_contact_absorption
    tool_stickiness = args.tool_stickiness
    floor_friction = args.floor_friction
    floor_stickiness = args.floor_stickiness
    floor_absorption = args.floor_absorption
    floor_plastic_damping_band = args.floor_plastic_damping_band
    velocity_damping = args.velocity_damping
    pure_viscoelastic = 1 if args.pure_viscoelastic else 0
    plastic_min = args.plastic_min
    plastic_max = args.plastic_max
    plastic_velocity_damping = args.plastic_velocity_damping
    plastic_affine_damping = args.plastic_affine_damping
    use_jp = 0 if not args.use_jp else 1
    jp_hardening = args.jp_hardening
    jp_min = args.jp_min
    jp_max = args.jp_max
    scripted_tools = 0 if (args.ros_control or getattr(args, "ros_tool_poses", False)) else 1
    scene_center_x = SCENE_CENTER[0]
    scene_center_y = SCENE_CENTER[1]
    scene_center_z = SCENE_CENTER[2]
    dough_radius_x = DOUGH_RADIUS[0]
    dough_radius_y = DOUGH_RADIUS[1]
    dough_radius_z = DOUGH_RADIUS[2]
    tool_y = TOOL_Y
    tool_z_offset = TOOL_Z_OFFSET
    tool_travel = TOOL_TRAVEL

    x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
    Jp = ti.field(dtype=ti.f32, shape=n_particles)
    yielded = ti.field(dtype=ti.i32, shape=n_particles)
    grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
    tool_x = ti.Vector.field(dim, dtype=ti.f32, shape=tool_vis_count)
    tool_center = ti.Vector.field(dim, dtype=ti.f32, shape=2)
    tool_quat = ti.Vector.field(4, dtype=ti.f32, shape=2)
    tool_velocity = ti.Vector.field(dim, dtype=ti.f32, shape=2)

    @ti.func
    def quat_to_matrix(q):
        xq, yq, zq, wq = q[0], q[1], q[2], q[3]
        return ti.Matrix([
            [1.0 - 2.0 * (yq * yq + zq * zq), 2.0 * (xq * yq - zq * wq), 2.0 * (xq * zq + yq * wq)],
            [2.0 * (xq * yq + zq * wq), 1.0 - 2.0 * (xq * xq + zq * zq), 2.0 * (yq * zq - xq * wq)],
            [2.0 * (xq * zq - yq * wq), 2.0 * (yq * zq + xq * wq), 1.0 - 2.0 * (xq * xq + yq * yq)],
        ])

    @ti.kernel
    def set_tool_state(poses: ti.types.ndarray(), velocities: ti.types.ndarray()):
        for i in range(2):
            tool_center[i] = ti.Vector([poses[i, 0], poses[i, 1], poses[i, 2]])
            tool_quat[i] = ti.Vector([poses[i, 3], poses[i, 4], poses[i, 5], poses[i, 6]])
            tool_velocity[i] = ti.Vector([velocities[i, 0], velocities[i, 1], velocities[i, 2]])

    @ti.func
    def tool_pose_and_velocity(tool_id, t):
        motion_time = ti.max(t - tool_motion_start, 0.0)
        progress = 0.0
        if t >= tool_motion_start:
            progress = ti.min(motion_time / tool_close_time, 1.0)

        center = tool_center[tool_id]
        cvel = tool_velocity[tool_id]
        if scripted_tools == 1:
            if tool_id == 0:
                center = ti.Vector([scene_center_x, tool_y, scene_center_z + tool_z_offset - tool_travel * progress])
                if t >= tool_motion_start and progress < 1.0:
                    cvel = ti.Vector([0.0, 0.0, -tool_travel / tool_close_time])
                else:
                    cvel = ti.Vector([0.0, 0.0, 0.0])
            else:
                center = ti.Vector([scene_center_x, tool_y, scene_center_z - tool_z_offset + tool_travel * progress])
                if t >= tool_motion_start and progress < 1.0:
                    cvel = ti.Vector([0.0, 0.0, tool_travel / tool_close_time])
                else:
                    cvel = ti.Vector([0.0, 0.0, 0.0])

        return center, cvel

    @ti.func
    def box_collision_velocity_and_normal(pos, t):
        inflated_half = ti.Vector([0.05, 0.05, 0.05]) + tool_contact_padding

        hit = 0
        normal = ti.Vector([0.0, 0.0, 0.0])
        collider_v = ti.Vector([0.0, 0.0, 0.0])

        for k in ti.static(range(2)):
            center, cvel = tool_pose_and_velocity(k, t)
            rot = quat_to_matrix(tool_quat[k])
            q = rot.transpose() @ (pos - center)
            aq = ti.abs(q)
            inside = aq.x < inflated_half.x and aq.y < inflated_half.y and aq.z < inflated_half.z
            if inside:
                penetration = inflated_half - aq
                min_pen = penetration.x
                local_normal = ti.Vector([1.0, 0.0, 0.0])
                if q.x < 0.0:
                    local_normal = ti.Vector([-1.0, 0.0, 0.0])

                if penetration.y < min_pen:
                    min_pen = penetration.y
                    local_normal = ti.Vector([0.0, 1.0, 0.0])
                    if q.y < 0.0:
                        local_normal = ti.Vector([0.0, -1.0, 0.0])

                if penetration.z < min_pen:
                    local_normal = ti.Vector([0.0, 0.0, 1.0])
                    if q.z < 0.0:
                        local_normal = ti.Vector([0.0, 0.0, -1.0])

                hit = 1
                normal = rot @ local_normal
                collider_v = cvel

        return hit, normal, collider_v

    @ti.func
    def project_particle_out_of_tools(pos, vel, t):
        new_pos = pos
        new_vel = vel
        inflated_half = ti.Vector([0.05, 0.05, 0.05]) + tool_contact_padding

        for k in ti.static(range(2)):
            center, cvel = tool_pose_and_velocity(k, t)
            rot = quat_to_matrix(tool_quat[k])
            q = rot.transpose() @ (new_pos - center)
            aq = ti.abs(q)
            inside = aq.x < inflated_half.x and aq.y < inflated_half.y and aq.z < inflated_half.z
            if inside:
                penetration = inflated_half - aq
                min_pen = penetration.x
                if penetration.y < min_pen:
                    min_pen = penetration.y
                if penetration.z < min_pen:
                    min_pen = penetration.z

                local_normal = ti.Vector([0.0, 0.0, 0.0])
                if min_pen == penetration.x:
                    if q.x < 0.0:
                        q.x = -(inflated_half.x + 1e-4)
                        local_normal = ti.Vector([-1.0, 0.0, 0.0])
                    else:
                        q.x = inflated_half.x + 1e-4
                        local_normal = ti.Vector([1.0, 0.0, 0.0])
                elif min_pen == penetration.y:
                    if q.y < 0.0:
                        q.y = -(inflated_half.y + 1e-4)
                        local_normal = ti.Vector([0.0, -1.0, 0.0])
                    else:
                        q.y = inflated_half.y + 1e-4
                        local_normal = ti.Vector([0.0, 1.0, 0.0])
                else:
                    if q.z < 0.0:
                        q.z = -(inflated_half.z + 1e-4)
                        local_normal = ti.Vector([0.0, 0.0, -1.0])
                    else:
                        q.z = inflated_half.z + 1e-4
                        local_normal = ti.Vector([0.0, 0.0, 1.0])

                normal = rot @ local_normal
                new_pos = center + rot @ q

                rel_v = new_vel - cvel
                vn = rel_v.dot(normal)
                if vn < 0.0:
                    rel_v -= normal * vn
                rel_v *= tool_contact_friction * (1.0 - tool_contact_absorption)
                rel_v *= 1.0 - tool_stickiness
                new_vel = cvel + rel_v

        return new_pos, new_vel

    @ti.kernel
    def initialize():
        for i in range(n_particles):
            # Random points in a dough-like ellipsoid centered in the unit scene.
            p = ti.Vector([0.0, 0.0, 0.0])
            accepted = False
            for _ in range(32):
                candidate = ti.Vector([
                    ti.random(ti.f32) * 2.0 - 1.0,
                    ti.random(ti.f32) * 2.0 - 1.0,
                    ti.random(ti.f32) * 2.0 - 1.0,
                ])
                if candidate.dot(candidate) <= 1.0 and not accepted:
                    p = candidate
                    accepted = True
            if not accepted:
                p = ti.Vector([0.0, 0.0, 0.0])

            x[i] = ti.Vector([scene_center_x, scene_center_y, scene_center_z]) + p * ti.Vector([
                dough_radius_x,
                dough_radius_y,
                dough_radius_z,
            ])
            v[i] = ti.Vector([0.0, 0.0, 0.0])
            C[i] = ti.Matrix.zero(ti.f32, dim, dim)
            F[i] = ti.Matrix.identity(ti.f32, dim)
            Jp[i] = 1.0
            yielded[i] = 0

        tool_center[0] = ti.Vector([scene_center_x, tool_y, scene_center_z + tool_z_offset])
        tool_center[1] = ti.Vector([scene_center_x, tool_y, scene_center_z - tool_z_offset])
        tool_quat[0] = ti.Vector([0.0, 0.0, 0.0, 1.0])
        tool_quat[1] = ti.Vector([0.0, 0.0, 0.0, 1.0])
        tool_velocity[0] = ti.Vector([0.0, 0.0, 0.0])
        tool_velocity[1] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def update_tool_visuals(time: ti.f32):
        half = ti.Vector([0.05, 0.05, 0.05])
        motion_time = ti.max(time - tool_motion_start, 0.0)
        progress = 0.0
        if time >= tool_motion_start:
            progress = ti.min(motion_time / tool_close_time, 1.0)

        for p in range(tool_vis_count):
            local_id = p % (tool_vis_resolution * tool_vis_resolution * tool_vis_resolution)
            ix = local_id % tool_vis_resolution
            iy = (local_id // tool_vis_resolution) % tool_vis_resolution
            iz = local_id // (tool_vis_resolution * tool_vis_resolution)
            uvw = ti.Vector([
                ix / (tool_vis_resolution - 1),
                iy / (tool_vis_resolution - 1),
                iz / (tool_vis_resolution - 1),
            ])
            local = (uvw * 2.0 - 1.0) * half
            tool_id = 0
            if p >= tool_vis_count // 2:
                tool_id = 1

            center = tool_center[tool_id]
            if scripted_tools == 1:
                if tool_id == 0:
                    center = ti.Vector([scene_center_x, tool_y, scene_center_z + tool_z_offset - tool_travel * progress])
                else:
                    center = ti.Vector([scene_center_x, tool_y, scene_center_z - tool_z_offset + tool_travel * progress])

            tool_x[p] = center + quat_to_matrix(tool_quat[tool_id]) @ local

    @ti.kernel
    def substep(time: ti.f32):
        for I in ti.grouped(grid_m):
            grid_v[I] = ti.Vector.zero(ti.f32, dim)
            grid_m[I] = 0.0

        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            F[p] = (ti.Matrix.identity(ti.f32, dim) + dt * C[p]) @ F[p]
            old_J = F[p].determinant()
            yielded[p] = 0
            if pure_viscoelastic == 0:
                U, sig, V = ti.svd(F[p])
                for d in ti.static(range(dim)):
                    unclamped = sig[d, d]
                    clamped = ti.min(ti.max(unclamped, plastic_min), plastic_max)
                    if ti.abs(unclamped - clamped) > 1e-6:
                        yielded[p] = 1
                    sig[d, d] = clamped
                F[p] = U @ sig @ V.transpose()
                if use_jp == 1:
                    new_J = F[p].determinant()
                    Jp[p] = ti.min(ti.max(Jp[p] * old_J / new_J, jp_min), jp_max)

            J = F[p].determinant()
            hardening = ti.exp(jp_hardening * (1.0 - Jp[p]))
            mu = mu_0 * hardening
            la = lambda_0 * hardening
            r, _ = ti.polar_decompose(F[p])
            elastic_stress = 2 * mu * (F[p] - r) @ F[p].transpose()
            elastic_stress += ti.Matrix.identity(ti.f32, dim) * la * J * (J - 1)
            viscous_stress = viscosity * (C[p] + C[p].transpose())
            stress = -dt * p_vol * 4 * inv_dx * inv_dx * (elastic_stress + viscous_stress)
            affine = stress + p_mass * C[p]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i].x * w[j].y * w[k].z
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

        for I in ti.grouped(grid_m):
            if grid_m[I] > 0:
                grid_v[I] = grid_v[I] / grid_m[I]
                grid_v[I].y += dt * gravity

                pos = I.cast(float) * dx

                if pos.y < floor_y and grid_v[I].y < 0.0:
                    grid_v[I].y *= -floor_absorption
                    grid_v[I].x *= floor_friction * (1.0 - floor_stickiness)
                    grid_v[I].z *= floor_friction * (1.0 - floor_stickiness)

                hit, normal, collider_v = box_collision_velocity_and_normal(pos, time)
                if hit == 1:
                    rel_v = grid_v[I] - collider_v
                    vn = rel_v.dot(normal)
                    if vn < 0.0:
                        rel_v -= normal * vn
                    rel_v *= tool_contact_friction * (1.0 - tool_contact_absorption)
                    rel_v *= 1.0 - tool_stickiness
                    grid_v[I] = collider_v + rel_v

                bound = 3
                if I.x < bound and grid_v[I].x < 0:
                    grid_v[I].x = 0
                if I.x > n_grid - bound and grid_v[I].x > 0:
                    grid_v[I].x = 0
                if I.y > n_grid - bound and grid_v[I].y > 0:
                    grid_v[I].y = 0
                if I.z < bound and grid_v[I].z < 0:
                    grid_v[I].z = 0
                if I.z > n_grid - bound and grid_v[I].z > 0:
                    grid_v[I].z = 0

        for p in x:
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]
            new_v = ti.Vector.zero(ti.f32, dim)
            new_C = ti.Matrix.zero(ti.f32, dim, dim)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * dx
                g_v = grid_v[base + offset]
                weight = w[i].x * w[j].y * w[k].z
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            v[p] = new_v * velocity_damping
            x[p] += dt * v[p]
            projected_x, projected_v = project_particle_out_of_tools(x[p], v[p], time)
            x[p] = projected_x
            v[p] = projected_v
            if x[p].y < floor_y + floor_plastic_damping_band:
                if v[p].y < 0.0:
                    v[p].y *= -floor_absorption
                v[p].x *= floor_friction * (1.0 - floor_stickiness)
                v[p].z *= floor_friction * (1.0 - floor_stickiness)
                new_C *= plastic_affine_damping
            if yielded[p] == 1:
                v[p] *= plastic_velocity_damping
                new_C *= plastic_affine_damping
            C[p] = new_C

    return x, tool_x, initialize, substep, update_tool_visuals, set_tool_state


def compute_camera_basis(position, look_at):
    position = np.asarray(position, dtype=np.float32)
    look_at = np.asarray(look_at, dtype=np.float32)
    forward = look_at - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    corrected_up = np.cross(right, forward)
    return right, corrected_up, forward


def optical_to_scene_transform(config):
    """Inverse of the camera coordinates used by depth_to_xyz (+y down)."""
    right, up, forward = compute_camera_basis(config["position"], config["lookAt"])
    return np.column_stack((right, -up, forward)), np.asarray(config["position"], dtype=np.float32)


def resolve_tool_mapping(config, matrix, offset, scale):
    """Use the cloud camera by default; retain explicit calibration overrides."""
    rotation, translation = optical_to_scene_transform(config)
    rotation = np.asarray([
        default if str(value).lower() == "auto" else float(value)
        for value, default in zip(matrix, rotation.ravel())
    ], dtype=np.float32).reshape(3, 3)
    translation = np.asarray([
        default if str(value).lower() == "auto" else float(value)
        for value, default in zip(offset, translation)
    ], dtype=np.float32)
    scale = np.asarray(scale, dtype=np.float32)
    if not (np.isfinite(rotation).all() and np.isfinite(translation).all() and np.isfinite(scale).all()):
        raise ValueError("ROS tool mapping must contain finite values")
    # A reflection cannot be encoded by a tool quaternion. Do not silently
    # use different transforms for the mesh orientation and its translation.
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError("--ros-tool-pose-matrix must be a proper rotation (determinant +1); use auto for the cloud camera")
    if np.any(scale <= 0.0) or not np.allclose(scale, scale[0]):
        raise ValueError("ROS tool scale must be positive and uniform to preserve rigid tool geometry")
    return rotation, translation


def map_tool_poses(poses, rotation, offset, scale):
    """Compose scene<-optical with optical<-tool; tool axes stay local."""
    mapped = np.array(poses, dtype=np.float32, copy=True)
    mapped[:, :3] = (poses[:, :3] * scale) @ rotation.T + offset
    for index, pose in enumerate(poses):
        mapped[index, 3:] = matrix_to_quaternion(rotation @ quaternion_to_matrix(pose[3:]))
    return mapped


def compute_particle_depth(points, width, height, config, particle_radius=PARTICLE_RENDER_RADIUS):
    """Render metric depth by splatting particles as z-buffered camera-space spheres."""
    right, up, forward = compute_camera_basis(config["position"], config["lookAt"])
    camera_pos = np.asarray(config["position"], dtype=np.float32)
    rel = points - camera_pos[None, :]
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = rel @ forward

    f = height / (2.0 * np.tan(np.radians(config["fieldOfView"]) / 2.0))
    u = f * cam_x / cam_z + width * 0.5
    v = height * 0.5 - f * cam_y / cam_z

    valid = (
        (cam_z - particle_radius > config["zNear"])
        & (cam_z < config["zFar"])
        & (u >= -f * particle_radius / cam_z)
        & (u < width + f * particle_radius / cam_z)
        & (v >= -f * particle_radius / cam_z)
        & (v < height + f * particle_radius / cam_z)
    )

    depth = np.full((height, width), config["zFar"], dtype=np.float32)
    for center_u, center_v, center_z in zip(u[valid], v[valid], cam_z[valid]):
        pixel_radius = max(1, int(np.ceil(f * particle_radius / (center_z - particle_radius))))
        u_min = max(0, int(np.floor(center_u)) - pixel_radius)
        u_max = min(width, int(np.floor(center_u)) + pixel_radius + 1)
        v_min = max(0, int(np.floor(center_v)) - pixel_radius)
        v_max = min(height, int(np.floor(center_v)) + pixel_radius + 1)
        if u_min >= u_max or v_min >= v_max:
            continue

        pixel_u, pixel_v = np.meshgrid(
            np.arange(u_min, u_max, dtype=np.float32) + 0.5,
            np.arange(v_min, v_max, dtype=np.float32) + 0.5,
        )
        local_x = (pixel_u - center_u) * center_z / f
        local_y = (pixel_v - center_v) * center_z / f
        radial_squared = local_x * local_x + local_y * local_y
        covered = radial_squared <= particle_radius * particle_radius
        if not covered.any():
            continue

        sphere_depth = center_z - np.sqrt(np.maximum(particle_radius * particle_radius - radial_squared, 0.0))
        patch = depth[v_min:v_max, u_min:u_max]
        np.minimum(patch, np.where(covered, sphere_depth, config["zFar"]), out=patch)

    return depth


def depth_to_xyz(depth, field_of_view_degrees, z_far):
    """Back-project depth into a ROS optical frame: +x right, +y down, +z forward."""
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth < z_far)
    pixel_v, pixel_u = np.nonzero(valid)
    if pixel_u.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    z = depth[pixel_v, pixel_u]
    focal_length = height / (2.0 * np.tan(np.radians(field_of_view_degrees) / 2.0))
    x = (pixel_u.astype(np.float32) + 0.5 - width * 0.5) * z / focal_length
    y = (pixel_v.astype(np.float32) + 0.5 - height * 0.5) * z / focal_length
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


def render_particle_depth(points, width, height, view_name, config, output_dir, frame_idx):
    depth = compute_particle_depth(points, width, height, config)

    finite = depth < config["zFar"]
    normalized = np.zeros_like(depth, dtype=np.uint8)
    if finite.any():
        d = depth[finite]
        normalized[finite] = ((1.0 - (d - d.min()) / max(d.max() - d.min(), 1e-6)) * 255).astype(np.uint8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{view_name}_depth_{frame_idx:06d}"
    np.save(output_dir / f"{stem}.npy", depth)
    Image.fromarray(normalized, "L").save(output_dir / f"{stem}.png")

    return {
        "name": view_name,
        "depth_array": str(output_dir / f"{stem}.npy"),
        "depth_image": str(output_dir / f"{stem}.png"),
        "position": config["position"],
        "lookAt": config["lookAt"],
        "fieldOfView": config["fieldOfView"],
        "zNear": config["zNear"],
        "zFar": config["zFar"],
        "width": width,
        "height": height,
    }


class RosPointCloudPublisher:
    """Minimal rclpy publisher kept optional so non-ROS simulation still works."""

    def __init__(self, topic, frame_id, node_name="taichi_dough_camera"):
        import rclpy
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import PointCloud2

        self.rclpy = rclpy
        self.PointCloud2 = PointCloud2
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init(args=None)
        self.node = rclpy.create_node(node_name)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.publisher = self.node.create_publisher(PointCloud2, topic, qos)
        self.node.get_logger().info(f"Publishing Taichi depth point clouds on {topic} in frame '{frame_id}'")
        self.frame_id = frame_id

    def publish(self, xyz):
        from sensor_msgs.msg import PointField

        cloud = self.PointCloud2()
        cloud.header.stamp = self.node.get_clock().now().to_msg()
        cloud.header.frame_id = self.frame_id
        cloud.height = 1
        cloud.width = int(xyz.shape[0])
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.is_bigendian = False
        cloud.point_step = 12
        cloud.row_step = cloud.point_step * cloud.width
        cloud.data = np.ascontiguousarray(xyz, dtype="<f4").tobytes()
        cloud.is_dense = True
        self.publisher.publish(cloud)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        self.node.destroy_node()
        if self.owns_context and self.rclpy.ok():
            self.rclpy.shutdown()


def quaternion_to_matrix(quaternion):
    x, y, z, w = normalize_quaternion(quaternion)
    return np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


def matrix_to_quaternion(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([(matrix[2, 1] - matrix[1, 2]) / s,
                      (matrix[0, 2] - matrix[2, 0]) / s,
                      (matrix[1, 0] - matrix[0, 1]) / s, 0.25 * s])
    else:
        axis = int(np.argmax(np.diag(matrix)))
        nxt = (axis + 1) % 3
        last = (axis + 2) % 3
        s = 2.0 * np.sqrt(max(1.0 + matrix[axis, axis] - matrix[nxt, nxt] - matrix[last, last], 1e-12))
        q = np.zeros(4)
        q[axis] = 0.25 * s
        q[3] = (matrix[last, nxt] - matrix[nxt, last]) / s
        q[nxt] = (matrix[nxt, axis] + matrix[axis, nxt]) / s
        q[last] = (matrix[last, axis] + matrix[axis, last]) / s
    return normalize_quaternion(q)


class RosToolPoseSubscriber:
    """Read two robot tool poses and map them into Taichi scene coordinates."""

    def __init__(self, topic, scale, matrix, offset, expected_frame, node_name="taichi_dough_tools"):
        import rclpy
        from dual_description.msg import PoseStampedArray

        self.rclpy = rclpy
        self.owns_context = not rclpy.ok()
        if self.owns_context:
            rclpy.init(args=None)
        self.node = rclpy.create_node(node_name)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.rotation = np.asarray(matrix, dtype=np.float32).reshape(3, 3)
        self.offset = np.asarray(offset, dtype=np.float32)
        self.expected_frame = expected_frame
        self.poses = TOOL_INITIAL_POSES.copy()
        self.velocities = np.zeros((2, 3), dtype=np.float32)
        self._last_positions = None
        self._last_time = None
        self.received = False
        self.subscription = self.node.create_subscription(PoseStampedArray, topic, self._callback, 10)
        self.node.get_logger().info(
            f"Following two tool poses from {topic} in '{expected_frame}'; rotation={self.rotation.tolist()} "
            f"scale={self.scale.tolist()} offset={self.offset.tolist()}"
        )

    def _callback(self, msg):
        if len(msg.poses) != 2:
            self.node.get_logger().warn("Expected exactly two poses on tool pose topic", throttle_duration_sec=2.0)
            return
        if msg.header.frame_id != self.expected_frame or any(
            stamped.header.frame_id and stamped.header.frame_id != self.expected_frame for stamped in msg.poses
        ):
            self.node.get_logger().warn(
                f"Tool poses must use the point-cloud frame '{self.expected_frame}'; received '{msg.header.frame_id}'",
                throttle_duration_sec=2.0,
            )
            return
        poses = np.asarray(
            [[p.pose.position.x, p.pose.position.y, p.pose.position.z,
              p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w] for p in msg.poses],
            dtype=np.float32,
        )
        mapped = map_tool_poses(poses, self.rotation, self.offset, self.scale)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._last_positions is not None and self._last_time is not None and stamp > self._last_time:
            self.velocities = (mapped[:, :3] - self._last_positions) / float(stamp - self._last_time)
        self.poses = mapped
        self._last_positions = mapped[:, :3].copy()
        self._last_time = stamp
        self.received = True

    def poll(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        return self.poses.copy(), self.velocities.copy()

    def close(self):
        self.node.destroy_node()
        if self.owns_context and self.rclpy.ok():
            self.rclpy.shutdown()


def normalize_quaternion(quaternion):
    quaternion = np.asarray(quaternion, dtype=np.float32)
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return quaternion / norm


def quaternion_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.array(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float32,
    )


def quaternion_conjugate(quaternion):
    return np.array([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]], dtype=np.float32)


def integrate_orientation(quaternion, angular_velocity, dt):
    angular_velocity = np.asarray(angular_velocity, dtype=np.float32)
    angle = np.linalg.norm(angular_velocity) * dt
    if angle < 1e-12:
        return normalize_quaternion(quaternion)

    axis = angular_velocity / np.linalg.norm(angular_velocity)
    half_angle = 0.5 * angle
    delta = np.array(
        [
            axis[0] * np.sin(half_angle),
            axis[1] * np.sin(half_angle),
            axis[2] * np.sin(half_angle),
            np.cos(half_angle),
        ],
        dtype=np.float32,
    )
    return normalize_quaternion(quaternion_multiply(delta, quaternion))


def angular_velocity_from_quaternions(old_quaternion, new_quaternion, dt):
    dt = max(float(dt), 1e-8)
    delta = quaternion_multiply(new_quaternion, quaternion_conjugate(old_quaternion))
    delta = normalize_quaternion(delta)
    if delta[3] < 0.0:
        delta = -delta
    vector_norm = np.linalg.norm(delta[:3])
    if vector_norm < 1e-8:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, delta[3])
    axis = delta[:3] / vector_norm
    return (axis * angle / dt).astype(np.float32)


class UdpRigidBoxControl:
    def __init__(self, receive_ports=(5005, 5007), transmit_ports=(5006, 5008), host="127.0.0.1", max_vel=1.0):
        self.host = host
        self.receive_sockets = []
        self.transmit_sockets = []
        self.transmit_addrs = [(host, port) for port in transmit_ports]
        self.poses = TOOL_INITIAL_POSES.copy()
        self.velocities = np.zeros((2, 6), dtype=np.float32)
        self.pose_control_active = np.zeros(2, dtype=bool)
        self.max_vel = max_vel

        for port in receive_ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
            sock.setblocking(False)
            self.receive_sockets.append(sock)

        for _ in transmit_ports:
            self.transmit_sockets.append(socket.socket(socket.AF_INET, socket.SOCK_DGRAM))

        print(f"ROS-style UDP box control listening on {receive_ports}, transmitting poses on {transmit_ports}")

    def close(self):
        for sock in self.receive_sockets + self.transmit_sockets:
            sock.close()

    def poll_and_integrate(self, dt):
        pose_driven = np.zeros(2, dtype=bool)
        for i, sock in enumerate(self.receive_sockets):
            try:
                while True:
                    data, _ = sock.recvfrom(1024)
                    msg = json.loads(data.decode())
                    if "vel" in msg:
                        vel = np.asarray(msg["vel"], dtype=np.float32)
                        if vel.size < 6:
                            vel = np.pad(vel, (0, 6 - vel.size))
                        self.velocities[i] = vel[:6]
                        self.pose_control_active[i] = False
                    elif "pose" in msg:
                        pose = np.asarray(msg["pose"], dtype=np.float32)
                        if pose.size >= 7:
                            old_pose = self.poses[i].copy()
                            pose_dt = float(msg.get("dt", dt))
                            self.poses[i] = pose[:7]
                            self.poses[i, 3:7] = normalize_quaternion(self.poses[i, 3:7])
                            self.velocities[i, :3] = (self.poses[i, :3] - old_pose[:3]) / max(pose_dt, 1e-8)
                            self.velocities[i, 3:6] = angular_velocity_from_quaternions(
                                old_pose[3:7], self.poses[i, 3:7], pose_dt
                            )
                            self.pose_control_active[i] = True
                            pose_driven[i] = True
            except BlockingIOError:
                pass

        linear = np.clip(self.velocities[:, :3], -self.max_vel, self.max_vel)
        angular = self.velocities[:, 3:6]
        for i in range(2):
            if not self.pose_control_active[i]:
                self.poses[i, :3] += linear[i] * dt
                self.poses[i, 3:7] = integrate_orientation(self.poses[i, 3:7], angular[i], dt)
            elif not pose_driven[i]:
                linear[i] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        return self.poses.astype(np.float32), linear.astype(np.float32)

    def transmit_poses(self):
        for i, sock in enumerate(self.transmit_sockets):
            msg = {"pose": self.poses[i].tolist()}
            sock.sendto(json.dumps(msg).encode(), self.transmit_addrs[i])


class DoughCenterTransmitter:
    def __init__(self, port=5010, host="127.0.0.1"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (host, port)
        print(f"Publishing dough center over UDP to {host}:{port}")

    def send(self, center):
        msg = {"center": np.asarray(center, dtype=float).tolist()}
        self.sock.sendto(json.dumps(msg).encode(), self.addr)

    def close(self):
        self.sock.close()


def save_frame_outputs(points, args, views, run_dir, frame_idx, step):
    np.save(run_dir / f"particles_{frame_idx:06d}.npy", points)
    frame = {"frame": frame_idx, "step": step, "views": []}
    for view_name in views:
        frame["views"].append(
            render_particle_depth(points, args.width, args.height, view_name, CAMERA_VIEWS[view_name], run_dir, frame_idx)
        )
    return frame


def draw_gui_frame(
    window,
    canvas,
    scene,
    camera,
    particles,
    ur_tool_vertices,
    ur_tool_indices,
    kinova_tool_vertices,
    kinova_tool_indices,
    free_camera=True,
    movement_speed=0.03,
    frame_path=None,
):
    if free_camera:
        camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.RMB)
    else:
        camera.position(0., 1.35, -1.35)
        camera.lookat(*SCENE_CENTER)
        camera.up(0.0, 1.0, 0.0)
    scene.set_camera(camera)
    scene.ambient_light((0.35, 0.35, 0.35))
    scene.point_light(pos=(0.4, 0.9, 1.1), color=(1.0, 1.0, 1.0))
    scene.particles(particles, radius=0.006, color=(0.78, 0.55, 0.36))
    # Taichi 1.7's normal-generation cache requires Taichi fields here;
    # NumPy arrays are unhashable and crash scene.mesh().
    scene.mesh(ur_tool_vertices, indices=ur_tool_indices, color=(0.22, 0.42, 0.75))
    scene.mesh(kinova_tool_vertices, indices=kinova_tool_indices, color=(0.75, 0.33, 0.22))
    canvas.scene(scene)
    if frame_path is not None:
        window.save_image(str(frame_path))
    window.show()


def create_video_writer(output_path, fps, frame_size):
    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer for {output_path}")
    return writer


def main():
    parser = argparse.ArgumentParser(description="Taichi MLS-MPM viscoelastic dough scene prototype.")
    parser.add_argument("--particles", type=int, default=24000)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--substeps-per-frame", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument(
        "--particle-render-radius",
        type=float,
        default=PARTICLE_RENDER_RADIUS,
        help="Sphere-splat radius in simulation metres used for depth rendering.",
    )
    parser.add_argument(
        "--max-pointcloud-points",
        type=int,
        default=4096,
        help="Randomly subsample each visible cloud to at most this many points.",
    )
    parser.add_argument("--view", choices=list(CAMERA_VIEWS), action="append")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--youngs-modulus", type=float, default=2000)
    parser.add_argument("--poisson-ratio", type=float, default=0.35)
    parser.add_argument("--viscosity", type=float, default=2.5)
    parser.add_argument("--density", type=float, default=1100.0)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--floor-y", type=float, default=0.33)
    parser.add_argument("--floor-friction", type=float, default=0.7)
    parser.add_argument(
        "--floor-absorption",
        type=float,
        default=0.0,
        help="Normal bounce kept at floor impact. 0 removes downward velocity, 1 is fully elastic bounce.",
    )
    parser.add_argument(
        "--tool-close-time",
        type=float,
        default=4.0,
        help="Seconds for each scripted tool to travel 0.28 m (default 0.07 m/s).",
    )
    parser.add_argument("--tool-motion-start", type=float, default=1.0)
    parser.add_argument("--tool-contact-padding", type=float, default=0.035)
    parser.add_argument("--tool-contact-friction", type=float, default=0.75)
    parser.add_argument(
        "--tool-contact-absorption",
        type=float,
        default=0.0,
        help="Extra damping at tool contacts. 0 keeps old response, 1 sticks to the tool velocity.",
    )
    parser.add_argument(
        "--tool-stickiness",
        type=float,
        default=0.0,
        help="Blend contact velocity toward tool velocity. 0 disables sticky tool contact.",
    )
    parser.add_argument(
        "--floor-stickiness",
        type=float,
        default=0.0,
        help="Extra damping of horizontal velocity at floor contact. 0 disables sticky floor contact.",
    )
    parser.add_argument(
        "--floor-plastic-damping-band",
        type=float,
        default=0.02,
        help="Height above the floor where particle velocity-gradient damping is applied.",
    )
    parser.add_argument("--velocity-damping", type=float, default=0.998)
    parser.add_argument(
        "--pure-viscoelastic",
        action="store_true",
        help="Disable SVD clamp plasticity and use the older viscoelastic-only model.",
    )
    parser.add_argument("--plastic-min", type=float, default=0.88, help="Minimum singular value kept in F.")
    parser.add_argument("--plastic-max", type=float, default=1.08, help="Maximum singular value kept in F.")
    parser.add_argument(
        "--plastic-velocity-damping",
        type=float,
        default=0.92,
        help="Extra velocity damping applied only to particles that yielded this step.",
    )
    parser.add_argument(
        "--plastic-affine-damping",
        type=float,
        default=0.80,
        help="Extra C/velocity-gradient damping applied only to particles that yielded this step.",
    )
    parser.add_argument("--use-jp", action="store_true", help="Track accumulated plastic volume change Jp.")
    parser.add_argument(
        "--jp-hardening",
        type=float,
        default=0.0,
        help="Hardening coefficient applied as exp(jp_hardening * (1 - Jp)).",
    )
    parser.add_argument("--jp-min", type=float, default=0.6)
    parser.add_argument("--jp-max", type=float, default=2.0)
    parser.add_argument("--cpu", action="store_true", help="Use CPU backend instead of GPU.")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=False, help="Open a live Taichi 3D viewer.")
    parser.add_argument("--no-save", action="store_true", help="Run without writing camera/particle frames.")
    parser.add_argument("--gui-fps-substeps", type=int, default=4, help="MPM substeps between GUI redraws.")
    parser.add_argument("--free-camera", action="store_true", help="Allow mouse/keyboard control of the GUI camera.", default=True)
    parser.add_argument("--camera-speed", type=float, default=0.01, help="Movement speed for --free-camera.")
    parser.add_argument("--record-video", action="store_true", help="Record live GUI frames to an MP4.")
    parser.add_argument("--video-path", type=Path, default=OUTPUT_DIR / "viscoelastic_mpm_gui.mp4")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--video-simulation-time",
        action="store_true",
        help="Set video FPS so playback duration matches simulated time exactly.",
    )
    parser.add_argument(
        "--keep-video-frames",
        action="store_true",
        help="Keep temporary PNG frames used for video encoding.",
    )
    parser.add_argument(
        "--ros-control",
        action="store_true",
        help="Control the two mesh tools with the same UDP vel/pose ports used by the SOFA ROS bridge.",
    )
    parser.add_argument(
        "--ros-tool-poses",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Follow two PoseStampedArray poses from ROS instead of scripted tool motion.",
    )
    parser.add_argument("--ros-tool-poses-topic", default="/tool_poses")
    parser.add_argument("--ros-tool-pose-scale", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("SX", "SY", "SZ"))
    parser.add_argument("--ros-tool-pose-matrix", nargs=9, default=["auto"] * 9, metavar="R", help="Optical-to-scene rotation; auto uses the point-cloud camera axes.")
    parser.add_argument("--ros-tool-pose-offset", nargs=3, default=["auto"] * 3, metavar=("OX", "OY", "OZ"), help="Optical origin in the scene; auto uses the point-cloud camera position.")
    parser.add_argument("--ros-tool-max-vel", type=float, default=1.0)
    parser.add_argument(
        "--ur-tool-mesh",
        type=Path,
        default=None,
        help="Override the UR tool mesh; defaults to the ROS package or bundled meshes/ur_spathla.stl.",
    )
    parser.add_argument(
        "--kinova-tool-mesh",
        type=Path,
        default=None,
        help="Override the Kinova tool mesh; defaults to the ROS package or bundled meshes/gen3_spathla.stl.",
    )
    parser.add_argument(
        "--tool-mesh-scale",
        type=float,
        default=0.001,
        help="Scale applied to STL coordinates in millimetres.",
    )
    parser.add_argument(
        "--publish-dough-center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Publish particle mean center over UDP.",
    )
    parser.add_argument("--dough-center-port", type=int, default=5010)
    parser.add_argument("--ros-pointcloud", action="store_true", help="Publish a live ROS 2 PointCloud2 from metric depth.")
    parser.add_argument("--ros-pointcloud-view", choices=list(CAMERA_VIEWS), default="front_dough")
    parser.add_argument("--ros-pointcloud-topic", default="/taichi_dough/front_dough/points")
    parser.add_argument("--ros-pointcloud-frame", default="front_dough_optical_frame")
    parser.add_argument(
        "--ros-pointcloud-every",
        type=int,
        default=10,
        help="Publish every N outer simulation steps.",
    )
    parser.add_argument(
        "--timing-report-interval",
        type=float,
        default=2.0,
        help="Wall-clock seconds between sim/real speed reports. 0 disables reports.",
    )
    args = parser.parse_args()
    if args.ros_control and args.ros_tool_poses:
        parser.error("--ros-control and --ros-tool-poses are mutually exclusive")
    if args.tool_mesh_scale <= 0.0:
        parser.error("--tool-mesh-scale must be positive")
    if args.tool_close_time <= 0.0:
        parser.error("--tool-close-time must be positive")
    try:
        scene_rotation, scene_offset = resolve_tool_mapping(
            CAMERA_VIEWS[args.ros_pointcloud_view], args.ros_tool_pose_matrix,
            args.ros_tool_pose_offset, args.ros_tool_pose_scale,
        )
    except ValueError as error:
        parser.error(str(error))

    ur_tool_mesh_path = find_tool_mesh("ur_spathla.stl", args.ur_tool_mesh)
    kinova_tool_mesh_path = find_tool_mesh("gen3_spathla.stl", args.kinova_tool_mesh)
    ur_tool_mesh, ur_tool_indices = load_binary_stl(ur_tool_mesh_path, args.tool_mesh_scale)
    kinova_tool_mesh, kinova_tool_indices = load_binary_stl(kinova_tool_mesh_path, args.tool_mesh_scale)
    print(f"UR tool mesh: {ur_tool_mesh_path}")
    print(f"Kinova tool mesh: {kinova_tool_mesh_path}")

    ti.init(arch=ti.cpu if args.cpu else ti.gpu)

    # Keep geometry and visual origins in their URDF link axes. Only the tool
    # pose is mapped from the optical frame; no second coordinate conversion.
    scene_scale = float(args.ros_tool_pose_scale[0]) if args.ros_tool_poses else 1.0
    ur_tool_mesh *= scene_scale
    kinova_tool_mesh *= scene_scale
    ur_visual_origin = UR_TOOL_VISUAL_ORIGIN * scene_scale
    kinova_visual_origin = KINOVA_TOOL_VISUAL_ORIGIN * scene_scale
    ur_visual_rotation = rpy_to_matrix(UR_TOOL_VISUAL_RPY)
    kinova_visual_rotation = rpy_to_matrix(KINOVA_TOOL_VISUAL_RPY)

    # Keep mesh buffers as Taichi fields.  Their contents are updated each GUI
    # frame after applying the current ROS tool poses.
    ur_tool_vertex_field = ti.Vector.field(3, dtype=ti.f32, shape=ur_tool_mesh.shape[0])
    ur_tool_index_field = ti.field(dtype=ti.i32, shape=ur_tool_indices.shape[0])
    ur_tool_index_field.from_numpy(ur_tool_indices)
    kinova_tool_vertex_field = ti.Vector.field(3, dtype=ti.f32, shape=kinova_tool_mesh.shape[0])
    kinova_tool_index_field = ti.field(dtype=ti.i32, shape=kinova_tool_indices.shape[0])
    kinova_tool_index_field.from_numpy(kinova_tool_indices)

    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    views = args.view or list(CAMERA_VIEWS)

    x, tool_x, initialize, substep, update_tool_visuals, set_tool_state = build_sim(args)
    initialize()
    set_tool_state(TOOL_INITIAL_POSES, np.zeros((2, 3), dtype=np.float32))
    tool_poses = TOOL_INITIAL_POSES.copy()
    update_tool_visuals(0.0)
    ros_control = UdpRigidBoxControl(max_vel=args.ros_tool_max_vel) if args.ros_control else None
    ros_tool_poses = RosToolPoseSubscriber(
        args.ros_tool_poses_topic, args.ros_tool_pose_scale, scene_rotation, scene_offset,
        expected_frame=args.ros_pointcloud_frame,
    ) if args.ros_tool_poses else None
    dough_center_tx = DoughCenterTransmitter(port=args.dough_center_port) if args.publish_dough_center else None
    pointcloud_publisher = None
    pointcloud_rng = np.random.default_rng()
    if args.ros_pointcloud:
        if args.ros_pointcloud_every < 1:
            parser.error("--ros-pointcloud-every must be at least 1")
        if args.max_pointcloud_points < 1:
            parser.error("--max-pointcloud-points must be at least 1")
        pointcloud_publisher = RosPointCloudPublisher(args.ros_pointcloud_topic, args.ros_pointcloud_frame)

    def publish_pointcloud(points):
        if pointcloud_publisher is None:
            return
        camera_config = CAMERA_VIEWS[args.ros_pointcloud_view]
        depth = compute_particle_depth(
            points,
            args.width,
            args.height,
            camera_config,
            args.particle_render_radius,
        )
        xyz = depth_to_xyz(depth, camera_config["fieldOfView"], camera_config["zFar"])
        if xyz.shape[0] > args.max_pointcloud_points:
            selected = pointcloud_rng.choice(
                xyz.shape[0],
                size=args.max_pointcloud_points,
                replace=False,
            )
            xyz = xyz[selected]
        pointcloud_publisher.publish(xyz)

    metadata = {
        "scene": "Taichi 3D MLS-MPM approximation of scene_with_camera.py",
        "model": "compressible corotated/Neo-Hookean stress plus viscosity, with optional SVD clamp plasticity",
        "frames": [],
        "video": {},
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }

    frame_idx = 0
    step = 0
    timing_start_wall = wall_time.perf_counter()
    timing_start_sim = 0.0
    timing_last_wall = timing_start_wall
    timing_last_sim = timing_start_sim

    def report_timing(sim_time):
        nonlocal timing_last_wall, timing_last_sim
        if args.timing_report_interval <= 0.0:
            return
        now_wall = wall_time.perf_counter()
        wall_dt = now_wall - timing_last_wall
        if wall_dt < args.timing_report_interval:
            return
        sim_dt = sim_time - timing_last_sim
        total_wall = max(now_wall - timing_start_wall, 1e-12)
        total_sim = sim_time - timing_start_sim
        instant_ratio = sim_dt / max(wall_dt, 1e-12)
        average_ratio = total_sim / total_wall
        suggested_deformpath_hz = 30.0 * instant_ratio
        print(
            "Timing: "
            f"sim={sim_time:.3f}s wall={total_wall:.3f}s "
            f"sim/real={instant_ratio:.3f} avg={average_ratio:.3f} "
            f"suggested_deformpath_rate={suggested_deformpath_hz:.2f} Hz",
            flush=True,
        )
        timing_last_wall = now_wall
        timing_last_sim = sim_time

    if args.gui:
        window = ti.ui.Window("Taichi Viscoelastic MPM Dough", (args.width, args.height), vsync=True)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(0., 1.35, 1.35)
        camera.lookat(*SCENE_CENTER)
        camera.up(0.0, 1.0, 0.0)
        video_writer = None
        video_frame_dir = run_dir / "video_frames"
        gui_frame_idx = 0

        if args.record_video:
            seconds_per_gui_frame = args.dt * args.substeps_per_frame * args.gui_fps_substeps
            if args.video_simulation_time:
                args.video_fps = 1.0 / seconds_per_gui_frame
            metadata["video"] = {
                "path": str(args.video_path),
                "fps": args.video_fps,
                "seconds_per_gui_frame": seconds_per_gui_frame,
                "simulation_time_playback": args.video_simulation_time,
            }
            print(
                "Recording video at "
                f"{args.video_fps:.6g} fps; each frame is {seconds_per_gui_frame:.6g} simulated seconds."
            )
            video_frame_dir.mkdir(parents=True, exist_ok=True)
            video_writer = create_video_writer(args.video_path, args.video_fps, (args.width, args.height))

        while window.running and step < args.steps:
            for _ in range(args.gui_fps_substeps):
                if ros_control is not None:
                    poses, linear_velocities = ros_control.poll_and_integrate(args.dt * args.substeps_per_frame)
                    set_tool_state(poses, linear_velocities)
                    tool_poses = poses
                    ros_control.transmit_poses()
                elif ros_tool_poses is not None:
                    poses, linear_velocities = ros_tool_poses.poll()
                    set_tool_state(poses, linear_velocities)
                    tool_poses = poses
                time = step * args.dt * args.substeps_per_frame
                for _ in range(args.substeps_per_frame):
                    substep(time)
                    time += args.dt
                step += 1

            update_tool_visuals(step * args.dt * args.substeps_per_frame)
            if dough_center_tx is not None:
                dough_center_tx.send(x.to_numpy().mean(axis=0))
            if pointcloud_publisher is not None and step % args.ros_pointcloud_every == 0:
                publish_pointcloud(x.to_numpy())
            frame_path = None
            if args.record_video:
                frame_path = video_frame_dir / f"gui_{gui_frame_idx:06d}.png"
            if ros_control is None and ros_tool_poses is None:
                progress = min(max((step * args.dt * args.substeps_per_frame - args.tool_motion_start) / args.tool_close_time, 0.0), 1.0)
                tool_poses = TOOL_INITIAL_POSES.copy()
                tool_poses[0, 2] -= TOOL_TRAVEL * progress
                tool_poses[1, 2] += TOOL_TRAVEL * progress
            ur_tool_vertex_field.from_numpy(
                transformed_tool_mesh(
                    ur_tool_mesh,
                    tool_poses[0],
                    ur_visual_origin,
                    ur_visual_rotation,
                )
            )
            kinova_tool_vertex_field.from_numpy(
                transformed_tool_mesh(
                    kinova_tool_mesh,
                    tool_poses[1],
                    kinova_visual_origin,
                    kinova_visual_rotation,
                )
            )
            draw_gui_frame(
                window,
                canvas,
                scene,
                camera,
                x,
                ur_tool_vertex_field,
                ur_tool_index_field,
                kinova_tool_vertex_field,
                kinova_tool_index_field,
                free_camera=args.free_camera,
                movement_speed=args.camera_speed,
                frame_path=frame_path,
            )

            if args.record_video:
                import cv2

                frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
                if frame is None:
                    raise RuntimeError(f"Could not read recorded GUI frame {frame_path}")
                video_writer.write(frame)
                if not args.keep_video_frames:
                    frame_path.unlink()
                gui_frame_idx += 1

            if not args.no_save and (step % args.save_every == 0 or step == args.steps - 1):
                metadata["frames"].append(save_frame_outputs(x.to_numpy(), args, views, run_dir, frame_idx, step))
                frame_idx += 1
            report_timing(step * args.dt * args.substeps_per_frame)

        if video_writer is not None:
            video_writer.release()
            print(f"Wrote GUI video to {args.video_path}")
    else:
        for step in range(args.steps):
            if ros_control is not None:
                poses, linear_velocities = ros_control.poll_and_integrate(args.dt * args.substeps_per_frame)
                set_tool_state(poses, linear_velocities)
                ros_control.transmit_poses()
            elif ros_tool_poses is not None:
                poses, linear_velocities = ros_tool_poses.poll()
                set_tool_state(poses, linear_velocities)
            time = step * args.dt * args.substeps_per_frame
            for _ in range(args.substeps_per_frame):
                substep(time)
                time += args.dt
            update_tool_visuals(time)
            if dough_center_tx is not None:
                dough_center_tx.send(x.to_numpy().mean(axis=0))
            if pointcloud_publisher is not None and (step + 1) % args.ros_pointcloud_every == 0:
                publish_pointcloud(x.to_numpy())

            if not args.no_save and (step % args.save_every == 0 or step == args.steps - 1):
                metadata["frames"].append(save_frame_outputs(x.to_numpy(), args, views, run_dir, frame_idx, step))
                frame_idx += 1
            report_timing((step + 1) * args.dt * args.substeps_per_frame)

    metadata_path = run_dir / "camera_parameters.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {frame_idx} Taichi MPM frames to {run_dir}")
    print(f"Wrote metadata to {metadata_path}")
    if ros_control is not None:
        ros_control.close()
    if ros_tool_poses is not None:
        ros_tool_poses.close()
    if dough_center_tx is not None:
        dough_center_tx.close()
    if pointcloud_publisher is not None:
        pointcloud_publisher.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
