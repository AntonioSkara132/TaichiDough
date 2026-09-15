from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

subject = importlib.import_module(
    "experiments.differentiable_mpm.episode18_chained_comparison"
)

PARAMETERS = {
    "youngs_modulus": 8800.0,
    "poisson_ratio": 0.49,
    "viscosity": 5.0,
    "plastic_min": 0.86857,
    "plastic_max": 1.06446,
    "floor_retention": 0.99,
    "tool_friction_coefficient": 0.99,
    "tool_stickiness": 0.1,
}


class FakeState:
    def __init__(self, value: float, n_particles: int = 2):
        self.x = np.full((n_particles, 3), value, dtype=np.float32)
        self.v = np.zeros_like(self.x)
        self.C = np.zeros((n_particles, 3, 3), dtype=np.float32)
        self.F = np.repeat(np.eye(3, dtype=np.float32)[None], n_particles, axis=0)
        self.Jp = np.ones(n_particles, dtype=np.float32)

    def copy(self):
        result = FakeState(0.0, len(self.x))
        for name in ("x", "v", "C", "F", "Jp"):
            setattr(result, name, getattr(self, name).copy())
        return result

    def validate(self):
        assert np.isfinite(self.x).all()

    def arrays(self):
        return {name: getattr(self, name) for name in ("x", "v", "C", "F", "Jp")}


class FakeStepper:
    instances = []

    def __init__(self, simulation_config, parameters, capacity, sdf):
        self.simulation_config = simulation_config
        self.parameters = parameters
        self.capacity = capacity
        self.sdf = sdf
        self.states = {}
        self.loads = []
        FakeStepper.instances.append(self)

    def load_state(self, slot, state):
        self.states[slot] = state.copy()
        self.loads.append(state.copy())

    def advance(self, slot, control):
        state = self.states[slot].copy()
        increment = float(control)
        state.x += increment
        state.v += 2.0 * increment
        state.C += 3.0 * increment
        state.F += 4.0 * increment
        state.Jp += 5.0 * increment
        self.states[slot + 1] = state

    def state(self, slot):
        return self.states[slot].copy()


def fake_episode(index: int, end_frame: int = 2):
    return SimpleNamespace(
        id=f"episode18_kugla_chunk{index:02d}",
        scored_window=SimpleNamespace(end_frame=end_frame),
    )


def fake_prepared(initial: float, control: float = 1.0):
    return SimpleNamespace(
        sdf=object(),
        simulation_config=SimpleNamespace(n_particles=2),
        parameters={},
        initial_state=FakeState(initial),
        frames=[SimpleNamespace(source_frame=2, completed_substeps=2, sim_time_s=0.2)],
        controls=[control, control],
        summary=lambda: {"prepared": True},
    )


def image_records(episode_ids):
    return [
        {"episode_id": episode_id, "role": role, "path": f"/{episode_id}_{role}.png"}
        for episode_id in episode_ids
        for role in ("start", "end")
    ]


def grouped_records(episode_ids):
    return [
        {
            "episode_id": episode_id,
            "frames": [
                {"role": role, "path": f"/{episode_id}_{role}.png"}
                for role in ("start", "end")
            ],
        }
        for episode_id in episode_ids
    ]


def test_requested_parameters_accept_forward_floor_retention():
    args = SimpleNamespace(**PARAMETERS)
    assert subject.parameters_from(args) == PARAMETERS


def test_rgb_requests_include_frame_zero_and_chunk_end():
    episodes = [fake_episode(index, end_frame=3 + index) for index in range(1, 5)]
    ranges = [
        {
            "start_pointcloud_ordinal": 10 * index,
            "end_pointcloud_ordinal_exclusive": 10 * index + 10,
        }
        for index in range(1, 5)
    ]
    requests = subject.requested_rgb_frames(episodes, ranges)
    assert [(row["role"], row["local_frame"]) for row in requests] == [
        ("start", 0), ("end", 4),
        ("start", 0), ("end", 5),
        ("start", 0), ("end", 6),
        ("start", 0), ("end", 7),
    ]


def test_chain_carries_full_terminal_state_and_saves_start_end(monkeypatch, tmp_path: Path):
    FakeStepper.instances.clear()
    monkeypatch.setattr(subject, "Stepper", FakeStepper)
    prepared = [fake_prepared(10.0 * index) for index in range(1, 5)]
    episodes = [fake_episode(index) for index in range(1, 5)]

    records, terminal_states = subject.run_chain(
        "from_chunk1", 0, prepared, episodes, tmp_path,
    )

    assert len(records) == 4
    assert set(terminal_states) == {1, 2, 3, 4}
    assert [[frame["role"] for frame in record["frames"]] for record in records] == [
        ["start", "end"], ["start", "end"], ["start", "end"], ["start", "end"],
    ]
    starts = [np.load(record["frames"][0]["snapshot"]) for record in records]
    ends = [np.load(record["frames"][1]["snapshot"]) for record in records]
    assert np.all(starts[0] == 10.0)
    for index in range(1, 4):
        assert np.array_equal(starts[index], ends[index - 1])
    assert [float(end[0, 0]) for end in ends] == [12.0, 14.0, 16.0, 18.0]
    terminal = np.load(tmp_path / "chains/from_chunk1/chunk04/simulation/terminal_state.npz")
    assert set(terminal.files) == {"x", "v", "C", "F", "Jp"}


def test_second_chain_starts_from_chain_a_chunk2_complete_end_state(monkeypatch, tmp_path: Path):
    FakeStepper.instances.clear()
    monkeypatch.setattr(subject, "Stepper", FakeStepper)
    prepared = [fake_prepared(10.0 * index) for index in range(1, 5)]
    episodes = [fake_episode(index) for index in range(1, 5)]
    first, first_terminal_states = subject.run_chain(
        "from_chunk1", 0, prepared, episodes, tmp_path,
    )
    source = first[1]["terminal_state"]
    provenance = {
        "type": "chain_terminal_full_state",
        "source_chain": "from_chunk1",
        "source_chunk": 2,
        "source_role": "end",
        "path": source["path"],
        "sha256": source["sha256"],
        "fields": ["x", "v", "C", "F", "Jp"],
        "next_controls": list(subject.EPISODE_IDS[1:]),
    }

    second, _ = subject.run_chain(
        "from_chunk2_replayed", 1, prepared, episodes, tmp_path,
        initial_state=first_terminal_states[2],
        initial_state_provenance=provenance,
    )

    first_end = np.load(source["path"])
    second_start = np.load(second[0]["initial_state"]["path"])
    assert set(first_end.files) == {"x", "v", "C", "F", "Jp"}
    assert set(second_start.files) == set(first_end.files)
    for field in ("x", "v", "C", "F", "Jp"):
        assert np.array_equal(second_start[field], first_end[field])
    assert second[0]["state_source"] == provenance
    second_chunk2_end = np.load(second[0]["terminal_state"]["path"])
    assert np.all(second_chunk2_end["x"] == first_end["x"] + 2.0)
    manifest = json.loads(
        (tmp_path / "chains/from_chunk2_replayed/chain_manifest.json").read_text()
    )
    assert manifest["initial_state"] == provenance
    assert manifest["controls_applied_in_order"] == list(subject.EPISODE_IDS[1:])


def test_particle_renderer_applies_clockwise_minus_ninety_rotation():
    points = np.array([[0.2, 0.0, 0.8]], dtype=np.float64)
    image = subject.render_particle_image(
        points,
        (np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        size=101,
        point_radius=0,
    )
    pixels = np.asarray(image)
    colored = np.argwhere(np.any(pixels != 255, axis=2))
    mean_y, mean_x = colored.mean(axis=0)
    assert mean_x > 50
    assert mean_y < 50


def test_topview_renderer_is_particle_only_and_records_rotation(tmp_path: Path):
    frames = []
    for role, offset in (("start", 0.0), ("end", 0.1)):
        path = tmp_path / f"{role}.npy"
        np.save(path, np.array([[offset, 0.0, 0.0], [0.2 + offset, 0.0, 0.3]]))
        frames.append({
            "role": role,
            "local_frame": 0 if role == "start" else 2,
            "snapshot": str(path),
            "snapshot_sha256": subject.sha256(path),
            "tool_poses": [[1.0] * 7],
        })
    chains = {
        "from_chunk1": [{"episode_id": subject.EPISODE_IDS[0], "chunk": 1, "frames": frames}],
        "from_chunk2_replayed": [],
    }

    rendered, camera = subject.render_topviews(tmp_path / "render", chains)

    assert len(rendered["from_chunk1"][0]["frames"]) == 2
    assert camera["tool_geometry_rendered"] is False
    assert camera["post_render_rotation_deg"] == -90
    assert camera["rotation_interpretation"] == "clockwise in image coordinates"
    assert all(Path(row["path"]).is_file()
               for row in rendered["from_chunk1"][0]["frames"])


def test_four_row_layout_has_start_end_pairs_and_required_empty_cells():
    rgb = image_records(subject.EPISODE_IDS)
    rendered = {
        "from_chunk1": grouped_records(subject.EPISODE_IDS),
        "from_chunk2_replayed": grouped_records(subject.EPISODE_IDS[1:]),
    }
    cells = subject.figure_cells(rgb, rendered)
    assert len(cells) == 4
    assert all(len(row) == 4 for row in cells)
    assert cells[2][0] is None
    assert cells[3][0] is None
    assert all(cell is not None for row in cells[:2] for cell in row)
    assert all(cell is not None for row in cells[2:] for cell in row[1:])
    for row in cells:
        for cell in row:
            if cell is not None:
                assert set(cell) == {"start", "end"}
