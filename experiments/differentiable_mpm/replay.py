"""Timestamp-faithful tool controls and observation-state indices."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import operator

import numpy as np

from .state import ToolControl


@dataclass(frozen=True)
class ReplayFrame:
    source_frame: int
    original_source_frame: int
    source_timestamp_s: float
    target_time_s: float
    completed_substeps: int
    sim_time_s: float

    @property
    def pairing_error_s(self):
        return self.sim_time_s - self.target_time_s


def observation_schedule(times, original_indices, end_frame: int, dt: float) -> tuple[ReplayFrame, ...]:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Observation timestamps must be finite and strictly increasing")
    if len(original_indices) != len(times):
        raise ValueError("Original source-frame mapping has the wrong length")
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in original_indices):
        raise ValueError("Original source-frame indices must be integers")
    if len(set(int(v) for v in original_indices)) != len(times):
        raise ValueError("Original source-frame indices must be unique")
    if isinstance(end_frame, (bool, np.bool_)) or not isinstance(end_frame, (int, np.integer)):
        raise ValueError("Replay endpoint must be an integer")
    if not np.isfinite(dt) or dt <= 0 or not 0 <= end_frame < len(times):
        raise ValueError("Invalid replay endpoint or time step")
    records = []
    for index in range(end_frame + 1):
        target = float(times[index] - times[0])
        completed = int(np.ceil(target / dt - 1e-10))
        records.append(ReplayFrame(index, int(original_indices[index]), float(times[index]),
                                   target, completed, float(completed * dt)))
    return tuple(records)


class RecordedControls(Sequence):
    """Precomputed fixed inputs; index k is sampled at k*dt before step k."""
    def __init__(self, replay, total_steps: int, dt: float):
        if total_steps < 0 or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Invalid number of replay steps or dt")
        self.dt = float(dt)
        self.poses = np.empty((total_steps, 2, 7), dtype=np.float32)
        self.velocities = np.empty((total_steps, 2, 6), dtype=np.float32)
        self.last_recorded_time = float(replay.times[-1])
        self.replay = replay
        for step in range(total_steps):
            poses, velocities = replay.at(min(step * self.dt, self.last_recorded_time))
            control = ToolControl(poses, velocities, step * self.dt)
            control.validate()
            self.poses[step] = poses
            self.velocities[step] = velocities
        self.poses.flags.writeable = False
        self.velocities.flags.writeable = False

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return ToolControl(self.poses[index], self.velocities[index], index * self.dt)

    def at_completed_step(self, completed_substeps: int):
        """End poses match the reference's final-time clamping convention."""
        if not 0 <= completed_substeps <= len(self):
            raise ValueError("Completed step lies outside replay")
        time = completed_substeps * self.dt
        poses, velocities = self.replay.at(min(time, self.last_recorded_time))
        return ToolControl(poses, velocities, time)
