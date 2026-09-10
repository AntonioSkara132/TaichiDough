#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import taichi as ti

try:
    from deformpath_topview import depth_to_pointcloud as shared_depth_to_pointcloud, rasterize_depth
except ImportError:
    from .deformpath_topview import depth_to_pointcloud as shared_depth_to_pointcloud, rasterize_depth

try:
    from taichi_viscoelastic_mpm_scene import (
        CAMERA_VIEWS,
        TOOL_INITIAL_POSES,
        UdpRigidBoxControl,
        build_sim,
    )
except ImportError:
    from .taichi_viscoelastic_mpm_scene import (
        CAMERA_VIEWS,
        TOOL_INITIAL_POSES,
        UdpRigidBoxControl,
        build_sim,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream Taichi virtual depth-camera pointcloud observations over UDP."
    )
    parser.add_argument("--send-host", default="127.0.0.1")
    parser.add_argument(
        "--send-port",
        type=int,
        default=6005,
        help="UDP destination for observation packets. 6005 avoids Taichi tool-control port 5005.",
    )
    parser.add_argument("--socket-timeout", type=float, default=0.2)
    parser.add_argument("--recv-buffer-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--max-packet-bytes", type=int, default=65000)
    parser.add_argument("--wait-for-response", action="store_true")
    parser.add_argument("--register-first", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Build packets and print summaries without sending UDP.")

    parser.add_argument("--steps", type=int, default=300, help="Number of observation packets to send.")
    parser.add_argument("--hz", type=float, default=30.0, help="Observation frequency in simulation time.")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Sleep so packets are sent at --hz in wall time. Otherwise send as fast as simulation runs.",
    )
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--view", choices=list(CAMERA_VIEWS), default="top_dough")
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=480)
    parser.add_argument("--depth-splat-radius", type=int, default=3)
    parser.add_argument(
        "--max-points",
        type=int,
        default=256,
        help="Maximum pointcloud points per packet. 0 sends all visible depth pixels, but that is usually too large for UDP JSON.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=5,
        help="Round point coordinates before JSON serialization to keep UDP packets small.",
    )
    parser.add_argument("--pointcloud-format", choices=("deformpath7", "xyz"), default="deformpath7")
    parser.add_argument(
        "--pointcloud-frame",
        choices=("world", "camera"),
        default="camera",
        help="Coordinate frame for sent points. camera sends [camera_x, camera_y, depth].",
    )
    parser.add_argument("--include-history", action="store_true")
    parser.add_argument("--history-limit", type=int, default=16)
    parser.add_argument("--history-stride", type=int, default=1)

    parser.add_argument("--particles", type=int, default=24000)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--dt", type=float, default=2e-4)
    parser.add_argument("--density", type=float, default=1100.0)
    parser.add_argument("--youngs-modulus", type=float, default=2000.0)
    parser.add_argument("--poisson-ratio", type=float, default=0.35)
    parser.add_argument("--viscosity", type=float, default=2.5)
    parser.add_argument("--gravity", type=float, default=-9.81)
    parser.add_argument("--floor-y", type=float, default=0.20)
    parser.add_argument("--floor-friction", type=float, default=0.7)
    parser.add_argument("--floor-absorption", type=float, default=0.0)
    parser.add_argument("--floor-stickiness", type=float, default=0.0)
    parser.add_argument("--floor-plastic-damping-band", type=float, default=0.02)
    parser.add_argument("--tool-close-time", type=float, default=0.04)
    parser.add_argument("--tool-motion-start", type=float, default=1.0)
    parser.add_argument("--tool-contact-padding", type=float, default=0.035)
    parser.add_argument("--tool-contact-friction", type=float, default=0.75)
    parser.add_argument("--tool-contact-absorption", type=float, default=0.0)
    parser.add_argument("--tool-stickiness", type=float, default=0.0)
    parser.add_argument("--velocity-damping", type=float, default=0.998)
    parser.add_argument("--pure-viscoelastic", action="store_true")
    parser.add_argument("--plastic-min", type=float, default=0.88)
    parser.add_argument("--plastic-max", type=float, default=1.08)
    parser.add_argument("--plastic-velocity-damping", type=float, default=0.92)
    parser.add_argument("--plastic-affine-damping", type=float, default=0.80)
    parser.add_argument("--use-jp", action="store_true")
    parser.add_argument("--jp-hardening", type=float, default=0.0)
    parser.add_argument("--jp-min", type=float, default=0.6)
    parser.add_argument("--jp-max", type=float, default=2.0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--ros-control",
        action="store_true",
        help="Also listen for Taichi tool velocity/pose control on UDP 5005/5007.",
    )
    parser.add_argument("--ros-tool-max-vel", type=float, default=1.0)
    return parser.parse_args()


def make_sim_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        particles=args.particles,
        grid=args.grid,
        dt=args.dt,
        density=args.density,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
        viscosity=args.viscosity,
        gravity=args.gravity,
        floor_y=args.floor_y,
        floor_friction=args.floor_friction,
        floor_absorption=args.floor_absorption,
        floor_stickiness=args.floor_stickiness,
        floor_plastic_damping_band=args.floor_plastic_damping_band,
        tool_close_time=args.tool_close_time,
        tool_motion_start=args.tool_motion_start,
        tool_contact_padding=args.tool_contact_padding,
        tool_contact_friction=args.tool_contact_friction,
        tool_contact_absorption=args.tool_contact_absorption,
        tool_stickiness=args.tool_stickiness,
        tool_collision="box",
        velocity_damping=args.velocity_damping,
        pure_viscoelastic=args.pure_viscoelastic,
        plastic_min=args.plastic_min,
        plastic_max=args.plastic_max,
        plastic_velocity_damping=args.plastic_velocity_damping,
        plastic_affine_damping=args.plastic_affine_damping,
        use_jp=args.use_jp,
        jp_hardening=args.jp_hardening,
        jp_min=args.jp_min,
        jp_max=args.jp_max,
        ros_control=args.ros_control,
    )


def rasterize_virtual_depth(
    points: np.ndarray,
    width: int,
    height: int,
    view_config: dict[str, Any],
    splat_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return rasterize_depth(points, width, height, view_config, splat_radius=splat_radius)


def make_depth_pointcloud(points: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    view_config = CAMERA_VIEWS[args.view]
    depth, nearest_indices, right, up, forward, camera_pos = rasterize_virtual_depth(
        points,
        args.depth_width,
        args.depth_height,
        view_config,
        args.depth_splat_radius,
    )
    pointcloud = shared_depth_to_pointcloud(
        depth,
        nearest_indices,
        right,
        up,
        forward,
        camera_pos,
        view_config,
        max_points=args.max_points,
        frame=args.pointcloud_frame,
    )
    if args.pointcloud_format == "xyz":
        return pointcloud
    extra = np.zeros((len(pointcloud), 4), dtype=np.float32)
    extra[:, 2] = 1.0
    return np.concatenate([pointcloud, extra], axis=1)


def compact_frame(frame: np.ndarray, decimals: int) -> list[list[float]]:
    if decimals >= 0:
        frame = np.round(frame, decimals=decimals)
    return frame.tolist()


def build_observation_packet(
    seq: int,
    timestamp: float,
    current_frame: np.ndarray,
    history: list[np.ndarray],
    decimals: int,
) -> dict[str, Any]:
    packet: dict[str, Any] = {
        "packet_type": "observation",
        "seq": seq,
        "timestamp": timestamp,
        "current_frame": compact_frame(current_frame, decimals),
    }
    if history:
        packet["history_frames"] = [compact_frame(frame, decimals) for frame in history]
    return packet


def build_registration_packet() -> dict[str, Any]:
    return {
        "packet_type": "register_observation_sender",
        "timestamp": time.time(),
    }


def response_summary(packet: dict[str, Any]) -> str:
    return (
        f"seq={packet.get('seq')} | "
        f"chunk_size={packet.get('chunk_size')} | "
        f"path_dim={packet.get('path_dim')} | "
        f"inference_ms={packet.get('inference_ms')} | "
        f"first_action={packet.get('first_action')}"
    )


def main() -> None:
    args = parse_args()
    if args.hz <= 0.0:
        raise ValueError("--hz must be positive")
    if args.max_points < 0:
        raise ValueError("--max-points must be non-negative")

    ti.init(arch=ti.cpu if args.cpu else ti.gpu)
    x, tool_x, initialize, substep, update_tool_visuals, set_tool_state = build_sim(make_sim_args(args))
    del tool_x, update_tool_visuals
    initialize()
    set_tool_state(TOOL_INITIAL_POSES, np.zeros((2, 3), dtype=np.float32))

    ros_control = UdpRigidBoxControl(max_vel=args.ros_tool_max_vel) if args.ros_control else None
    sock = None if args.dry_run else socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if sock is not None:
        sock.settimeout(args.socket_timeout)

    substeps_per_packet = max(1, int(round(1.0 / (args.hz * args.dt))))
    period_s = 1.0 / args.hz
    history: list[np.ndarray] = []
    start_wall = time.perf_counter()

    print(
        f"Streaming Taichi {args.view} depth pointclouds to udp://{args.send_host}:{args.send_port} | "
        f"hz={args.hz:.3f} | sim_substeps_per_packet={substeps_per_packet} | "
        f"max_points={args.max_points if args.max_points > 0 else 'uncapped'} | "
        f"pointcloud_frame={args.pointcloud_frame} | dry_run={args.dry_run}"
    )

    try:
        if sock is not None and args.register_first:
            payload = json.dumps(build_registration_packet(), ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            sock.sendto(payload, (args.send_host, args.send_port))

        sim_time = 0.0
        for seq in range(args.steps):
            frame_wall_start = time.perf_counter()
            for _ in range(substeps_per_packet):
                if ros_control is not None:
                    poses, linear_velocities = ros_control.poll_and_integrate(args.dt)
                    set_tool_state(poses, linear_velocities)
                    ros_control.transmit_poses()
                substep(sim_time)
                sim_time += args.dt

            pointcloud = make_depth_pointcloud(x.to_numpy(), args)
            history_packet = []
            if args.include_history and history:
                history_packet = history[:: max(1, args.history_stride)][-max(1, args.history_limit) :]
            packet = build_observation_packet(seq, sim_time, pointcloud, history_packet, args.round_decimals)
            payload = json.dumps(packet, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
            if len(payload) > args.max_packet_bytes:
                raise ValueError(
                    f"UDP payload for seq {seq} is {len(payload)} bytes, above --max-packet-bytes={args.max_packet_bytes}. "
                    "Use a smaller --max-points or larger --max-packet-bytes."
                )

            response_text = ""
            if sock is not None:
                sock.sendto(payload, (args.send_host, args.send_port))
                if args.wait_for_response:
                    try:
                        response_payload, sender_addr = sock.recvfrom(args.recv_buffer_bytes)
                        response_packet = json.loads(response_payload.decode("utf-8"))
                        response_text = f" | response_from={sender_addr[0]}:{sender_addr[1]} | {response_summary(response_packet)}"
                    except socket.timeout:
                        response_text = " | response_timeout"

            if args.include_history:
                history.append(pointcloud)
                history = history[-max(1, args.history_limit * max(1, args.history_stride)) :]

            if seq % max(1, args.print_every) == 0:
                elapsed = time.perf_counter() - start_wall
                print(
                    f"message={seq + 1} | seq={seq} | sim_time={sim_time:.4f}s | "
                    f"points={pointcloud.shape[0]} | bytes={len(payload)} | elapsed={elapsed:.2f}s"
                    f"{response_text}",
                    flush=True,
                )

            if args.realtime:
                sleep_s = period_s - (time.perf_counter() - frame_wall_start)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
    finally:
        if sock is not None:
            sock.close()
        if ros_control is not None:
            ros_control.close()


if __name__ == "__main__":
    main()
