import argparse
import json
from pathlib import Path

import numpy as np
import taichi as ti
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "taichi_dough_fall"

CAMERA = {
    "position": [0.90, 0.28, 0.5],
    "lookAt": [0.5, 0.28, 0.5],
    "fieldOfView": 57.0,
    "zNear": 0.01,
    "zFar": 2.0,
}


def build_sim(args):
    dim = 3
    n_particles = args.particles
    n_grid = args.grid
    dx = 1.0 / n_grid
    inv_dx = float(n_grid)
    dt = args.dt

    p_vol = (dx * 0.5) ** dim
    p_mass = p_vol * args.density
    mu_0 = args.youngs_modulus / (2 * (1 + args.poisson_ratio))
    lambda_0 = (
        args.youngs_modulus
        * args.poisson_ratio
        / ((1 + args.poisson_ratio) * (1 - 2 * args.poisson_ratio))
    )
    gravity = args.gravity
    floor_y = args.floor_y
    viscosity = args.viscosity
    velocity_damping = args.velocity_damping
    friction = args.floor_friction

    x = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    v = ti.Vector.field(dim, dtype=ti.f32, shape=n_particles)
    C = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
    F = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=n_particles)
    grid_v = ti.Vector.field(dim, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
    grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

    floor_res = 28
    floor_count = floor_res * floor_res
    floor_x = ti.Vector.field(dim, dtype=ti.f32, shape=floor_count)

    @ti.kernel
    def initialize():
        for i in range(n_particles):
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

            x[i] = ti.Vector([0.5, args.start_y, 0.5]) + p * ti.Vector([
                args.radius_x,
                args.radius_y,
                args.radius_z,
            ])
            v[i] = ti.Vector([0.0, 0.0, 0.0])
            C[i] = ti.Matrix.zero(ti.f32, dim, dim)
            F[i] = ti.Matrix.identity(ti.f32, dim)

        for i in range(floor_count):
            ix = i % floor_res
            iz = i // floor_res
            floor_x[i] = ti.Vector([
                0.20 + 0.60 * ix / (floor_res - 1),
                floor_y,
                0.20 + 0.60 * iz / (floor_res - 1),
            ])

    @ti.kernel
    def substep():
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
            J = F[p].determinant()
            R, _ = ti.polar_decompose(F[p])
            elastic_stress = 2 * mu_0 * (F[p] - R) @ F[p].transpose()
            elastic_stress += ti.Matrix.identity(ti.f32, dim) * lambda_0 * J * (J - 1)
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
                grid_v[I] /= grid_m[I]
                grid_v[I].y += dt * gravity

                pos = I.cast(float) * dx
                if pos.y < floor_y and grid_v[I].y < 0.0:
                    grid_v[I].y = 0.0
                    grid_v[I].x *= friction
                    grid_v[I].z *= friction

                bound = 3
                if I.x < bound and grid_v[I].x < 0.0:
                    grid_v[I].x = 0.0
                if I.x > n_grid - bound and grid_v[I].x > 0.0:
                    grid_v[I].x = 0.0
                if I.y > n_grid - bound and grid_v[I].y > 0.0:
                    grid_v[I].y = 0.0
                if I.z < bound and grid_v[I].z < 0.0:
                    grid_v[I].z = 0.0
                if I.z > n_grid - bound and grid_v[I].z > 0.0:
                    grid_v[I].z = 0.0

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
                weight = w[i].x * w[j].y * w[k].z
                g_v = grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            v[p] = new_v * velocity_damping
            x[p] += dt * v[p]
            C[p] = new_C

    return x, floor_x, initialize, substep


def camera_basis(position, look_at):
    position = np.asarray(position, dtype=np.float32)
    look_at = np.asarray(look_at, dtype=np.float32)
    forward = look_at - position
    forward /= np.linalg.norm(forward)
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    return right, np.cross(right, forward), forward


def render_depth(points, width, height, output_dir, frame_idx):
    right, up, forward = camera_basis(CAMERA["position"], CAMERA["lookAt"])
    rel = points - np.asarray(CAMERA["position"], dtype=np.float32)[None, :]
    cam_x = rel @ right
    cam_y = rel @ up
    cam_z = rel @ forward

    f = height / (2.0 * np.tan(np.radians(CAMERA["fieldOfView"]) / 2.0))
    u = (f * cam_x / cam_z + width * 0.5).astype(np.int32)
    v = (height * 0.5 - f * cam_y / cam_z).astype(np.int32)
    valid = (
        (cam_z > CAMERA["zNear"])
        & (cam_z < CAMERA["zFar"])
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    depth = np.full((height, width), CAMERA["zFar"], dtype=np.float32)
    for px, py, z in zip(u[valid], v[valid], cam_z[valid]):
        depth[py, px] = min(depth[py, px], z)

    finite = depth < CAMERA["zFar"]
    image = np.zeros_like(depth, dtype=np.uint8)
    if finite.any():
        d = depth[finite]
        image[finite] = ((1.0 - (d - d.min()) / max(d.max() - d.min(), 1e-6)) * 255).astype(np.uint8)

    stem = f"fall_depth_{frame_idx:06d}"
    np.save(output_dir / f"{stem}.npy", depth)
    Image.fromarray(image, "L").save(output_dir / f"{stem}.png")
    return {"depth_array": str(output_dir / f"{stem}.npy"), "depth_image": str(output_dir / f"{stem}.png")}


def draw_gui(window, canvas, scene, camera, particles, floor, frame_path=None, free_camera=False, movement_speed=0.03):
    if free_camera:
        camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.RMB)
    else:
        camera.position(*CAMERA["position"])
        camera.lookat(*CAMERA["lookAt"])
        camera.up(0.0, 1.0, 0.0)
    scene.set_camera(camera)
    scene.ambient_light((0.35, 0.35, 0.35))
    scene.point_light(pos=(0.4, 0.9, 1.1), color=(1.0, 1.0, 1.0))
    scene.particles(floor, radius=0.006, color=(0.35, 0.38, 0.42))
    scene.particles(particles, radius=0.006, color=(0.78, 0.55, 0.36))
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
    parser = argparse.ArgumentParser(description="Taichi MLS-MPM dough falling onto a floor.")
    parser.add_argument("--particles", type=int, default=24000)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--steps", type=int, default=360)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--substeps-per-frame", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default =600)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start-y", type=float, default=0.3)
    parser.add_argument("--radius-x", type=float, default=0.14)
    parser.add_argument("--radius-y", type=float, default=0.055)
    parser.add_argument("--radius-z", type=float, default=0.12)
    parser.add_argument("--youngs-modulus", type=float, default=9000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.35)
    parser.add_argument("--viscosity", type=float, default=12.0)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--gravity", type=float, default=-1.2)
    parser.add_argument("--floor-y", type=float, default=0.20)
    parser.add_argument("--floor-friction", type=float, default=0.55)
    parser.add_argument("--velocity-damping", type=float, default=0.994)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--gui-fps-substeps", type=int, default=4)
    parser.add_argument("--free-camera", action="store_true")
    parser.add_argument("--camera-speed", type=float, default=0.03)
    parser.add_argument("--record-video", action="store_true", help="Record the live GUI frames to an MP4.")
    parser.add_argument("--video-path", type=Path, default=OUTPUT_DIR / "dough_fall_gui.mp4")
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
    args = parser.parse_args()

    ti.init(arch=ti.cpu if args.cpu else ti.gpu)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x, floor_x, initialize, substep = build_sim(args)
    initialize()

    metadata = {
        "scene": "Taichi 3D MLS-MPM dough falling onto a floor",
        "model": "compressible corotated/Neo-Hookean stress plus viscous velocity-gradient stress",
        "camera": CAMERA,
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "video": {},
        "frames": [],
    }

    frame_idx = 0
    step = 0

    if args.gui:
        window = ti.ui.Window("Taichi Dough Falling On Floor", (args.width, args.height), vsync=True)
        canvas = window.get_canvas()
        scene = window.get_scene()
        camera = ti.ui.Camera()
        camera.position(*CAMERA["position"])
        camera.lookat(*CAMERA["lookAt"])
        camera.up(0.0, 1.0, 0.0)
        video_writer = None
        video_frame_dir = args.output_dir / "video_frames"
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
                for _ in range(args.substeps_per_frame):
                    substep()
                step += 1
            frame_path = None
            if args.record_video:
                frame_path = video_frame_dir / f"gui_{gui_frame_idx:06d}.png"

            draw_gui(
                window,
                canvas,
                scene,
                camera,
                x,
                floor_x,
                frame_path=frame_path,
                free_camera=args.free_camera,
                movement_speed=args.camera_speed,
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
                points = x.to_numpy()
                np.save(args.output_dir / f"particles_{frame_idx:06d}.npy", points)
                metadata["frames"].append({"frame": frame_idx, "step": step, **render_depth(points, args.width, args.height, args.output_dir, frame_idx)})
                frame_idx += 1

        if video_writer is not None:
            video_writer.release()
            print(f"Wrote GUI video to {args.video_path}")
    else:
        for step in range(args.steps):
            for _ in range(args.substeps_per_frame):
                substep()
            if not args.no_save and (step % args.save_every == 0 or step == args.steps - 1):
                points = x.to_numpy()
                np.save(args.output_dir / f"particles_{frame_idx:06d}.npy", points)
                metadata["frames"].append({"frame": frame_idx, "step": step, **render_depth(points, args.width, args.height, args.output_dir, frame_idx)})
                frame_idx += 1

    metadata_path = args.output_dir / "camera_parameters.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {frame_idx} dough-fall frames to {args.output_dir}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
