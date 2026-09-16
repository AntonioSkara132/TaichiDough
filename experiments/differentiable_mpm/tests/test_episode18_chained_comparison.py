from __future__ import annotations

import colorsys
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageOps

from experiments.differentiable_mpm.state import ToolControl

subject = importlib.import_module(
    "experiments.differentiable_mpm.episode18_chained_comparison"
)

PARAMETERS = {
    "youngs_modulus": 9780.0,
    "poisson_ratio": 0.49,
    "viscosity": 10.0,
    "plastic_min": 0.89,
    "plastic_max": 1.09,
    "floor_retention": 0.9,
    "tool_friction_coefficient": 0.9,
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
        for name in subject.STATE_FIELDS:
            setattr(result, name, getattr(self, name).copy())
        return result

    def validate(self):
        assert all(np.isfinite(getattr(self, name)).all() for name in subject.STATE_FIELDS)

    def arrays(self):
        return {name: getattr(self, name) for name in subject.STATE_FIELDS}


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


def fake_prepared(initial: float, control: float = 1.0, context: int = 0):
    return SimpleNamespace(
        sdf=f"sdf-{context}",
        simulation_config=SimpleNamespace(n_particles=2, context=context),
        parameters={"context": context},
        initial_state=FakeState(initial),
        frames=[SimpleNamespace(source_frame=2, completed_substeps=2, sim_time_s=0.2)],
        controls=[control, control],
        summary=lambda: {"prepared": True, "context": context},
    )


def image_records(episode_ids, prefix=""):
    return [
        {"episode_id": episode_id, "role": role, "path": f"{prefix}{episode_id}_{role}.png"}
        for episode_id in episode_ids
        for role in ("start", "end")
    ]


def grouped_records(episode_ids, prefix=""):
    return [
        {
            "episode_id": episode_id,
            "chunk": int(episode_id[-2:]),
            "frames": [
                {"role": role, "path": f"{prefix}{episode_id}_{role}.png"}
                for role in ("start", "end")
            ],
        }
        for episode_id in episode_ids
    ]


def load_run_path(root: Path, value: str) -> Path:
    return root / value


def pale_dough_rgb() -> tuple[int, int, int]:
    rgb = colorsys.hsv_to_rgb(140.0 / 360.0, 40.0 / 255.0, 220.0 / 255.0)
    red, green, blue = (round(value * 255) for value in rgb)
    return red, green, blue


def test_requested_parameters_are_corrected_article_values():
    args = SimpleNamespace(**PARAMETERS)
    assert subject.parameters_from(args) == PARAMETERS
    assert subject.SCHEMA.endswith("/v3")


def test_rgb_requests_include_frame_zero_and_chunk_end():
    episodes = [fake_episode(index, end_frame=3 + index) for index in range(1, 5)]
    ranges = [
        {"start_pointcloud_ordinal": 10 * index,
         "end_pointcloud_ordinal_exclusive": 10 * index + 10}
        for index in range(1, 5)
    ]
    requests = subject.requested_rgb_frames(episodes, ranges)
    assert [(row["role"], row["local_frame"]) for row in requests] == [
        ("start", 0), ("end", 4), ("start", 0), ("end", 5),
        ("start", 0), ("end", 6), ("start", 0), ("end", 7),
    ]


def test_chain_carries_complete_state_and_uses_selected_dynamics(monkeypatch, tmp_path: Path):
    FakeStepper.instances.clear()
    monkeypatch.setattr(subject, "Stepper", FakeStepper)
    prepared = [fake_prepared(10.0 * index, context=index) for index in range(1, 5)]
    episodes = [fake_episode(index) for index in range(1, 5)]

    records, terminal_states = subject.run_chain(
        subject.CHAIN_A, 0, prepared, episodes, tmp_path, dynamics_index=0,
    )

    assert len(records) == 4
    assert set(terminal_states) == {1, 2, 3, 4}
    assert FakeStepper.instances[-1].simulation_config is prepared[0].simulation_config
    starts = [np.load(load_run_path(tmp_path, record["frames"][0]["snapshot"]))
              for record in records]
    ends = [np.load(load_run_path(tmp_path, record["frames"][1]["snapshot"]))
            for record in records]
    assert np.all(starts[0] == 10.0)
    for index in range(1, 4):
        assert np.array_equal(starts[index], ends[index - 1])
    terminal = np.load(tmp_path / "chains/chain_a/chunk04/simulation/terminal_state.npz")
    assert set(terminal.files) == set(subject.STATE_FIELDS)


def test_independent_c2_uses_c2_reconstruction_and_context(monkeypatch, tmp_path: Path):
    FakeStepper.instances.clear()
    monkeypatch.setattr(subject, "Stepper", FakeStepper)
    prepared = [fake_prepared(10.0 * index, context=index) for index in range(1, 5)]
    episodes = [fake_episode(index) for index in range(1, 5)]

    records, _ = subject.run_chain(
        subject.INDEPENDENT_C2, 1, prepared, episodes, tmp_path, dynamics_index=1,
    )

    initial = np.load(load_run_path(tmp_path, records[0]["initial_state"]["path"]))
    assert np.all(initial["x"] == 20.0)
    assert FakeStepper.instances[-1].simulation_config is prepared[1].simulation_config
    manifest = json.loads((tmp_path / "chains/independent_from_chunk2/chain_manifest.json").read_text())
    assert manifest["control_start_episode"] == subject.EPISODE_IDS[1]
    assert manifest["dynamics_context_episode"] == subject.EPISODE_IDS[1]


def test_c1_terminal_replay_uses_c1_context_and_reproduces_chain_a(monkeypatch, tmp_path: Path):
    FakeStepper.instances.clear()
    monkeypatch.setattr(subject, "Stepper", FakeStepper)
    prepared = [fake_prepared(10.0 * index, context=index) for index in range(1, 5)]
    episodes = [fake_episode(index) for index in range(1, 5)]
    chain_a, terminal_states = subject.run_chain(
        subject.CHAIN_A, 0, prepared, episodes, tmp_path, dynamics_index=0,
    )
    source = chain_a[0]["terminal_state"]
    provenance = {
        "type": "chain_terminal_full_state", "source_chain": subject.CHAIN_A,
        "source_chunk": 1, "source_role": "end", "path": source["path"],
        "sha256": source["sha256"], "fields": list(subject.STATE_FIELDS),
    }
    replay, _ = subject.run_chain(
        subject.REPLAY_C1_END, 1, prepared, episodes, tmp_path,
        dynamics_index=0, initial_state=terminal_states[1],
        initial_state_provenance=provenance, initial_time_s=0.2,
    )

    assert FakeStepper.instances[-1].simulation_config is prepared[0].simulation_config
    replay_start = np.load(load_run_path(tmp_path, replay[0]["initial_state"]["path"]))
    chain_a_c1_end = np.load(load_run_path(tmp_path, source["path"]))
    for field in subject.STATE_FIELDS:
        assert np.array_equal(replay_start[field], chain_a_c1_end[field])
    metrics = subject.replay_comparison(tmp_path, chain_a, replay)
    assert metrics["all_exact"] is True
    assert metrics["all_within_tolerance"] is True
    assert all(
        values["max_abs_difference"] == 0.0
        for comparison in metrics["comparisons"]
        for values in comparison["fields"].values()
    )


def test_episode18_context_requires_mass_contact_and_padding():
    config = SimpleNamespace(
        n_particles=24000, particle_mass=subject.EXPECTED_TOTAL_MASS_KG / 24000,
        particle_volume=1e-8, grid=48, plasticity="stretch-clamp", use_jp=False,
        jp_hardening=0.0, tool_collision="sdf",
        tool_contact_padding=1.0 / 48.0 / 16.0,
        tool_contact_model="coulomb-adhesive-v1", tool_contact_absorption=0.0,
        p2g_mode="atomic",
    )
    subject.validate_episode18_context([SimpleNamespace(simulation_config=config, parameters={"tool_retention": 1.0})])
    config.tool_contact_padding = 0.0
    with pytest.raises(ValueError, match="padding"):
        subject.validate_episode18_context([SimpleNamespace(simulation_config=config, parameters={"tool_retention": 1.0})])


def test_tool_controls_are_lowered_two_millimeters_without_mutating_source():
    poses = np.zeros((2, 7), dtype=np.float64)
    poses[:, :3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    poses[:, 6] = 1.0
    original = ToolControl(poses, np.ones((2, 6)), time=0.1)

    shifted = subject.offset_tool_control(original)

    assert np.allclose(shifted.poses[:, 1], original.poses[:, 1] - 0.002)
    assert np.array_equal(shifted.poses[:, (0, 2, 3, 4, 5, 6)],
                          original.poses[:, (0, 2, 3, 4, 5, 6)])
    assert np.array_equal(shifted.velocities, original.velocities)
    assert np.allclose(original.poses[:, 1], [0.2, 0.5])


def test_compare_arrays_reports_exact_tolerant_and_incompatible_cases():
    base = np.array([1.0, 2.0], dtype=np.float32)
    exact = subject.compare_arrays(base, base.copy())
    close = subject.compare_arrays(base, base + np.float32(1e-6))
    incompatible = subject.compare_arrays(base, base.astype(np.float64))
    assert exact["exact_equal"] and exact["within_tolerance"]
    assert not close["exact_equal"] and close["within_tolerance"]
    assert not incompatible["compatible"] and not incompatible["within_tolerance"]


def test_largest_component_rejects_threshold_matching_distractors():
    rgb = np.zeros((80, 100, 3), dtype=np.uint8)
    color = pale_dough_rgb()
    rgb[20:60, 30:75] = color
    rgb[2:7, 2:7] = color
    selected, area, bounds = subject.largest_component(subject.hsv_mask(rgb), minimum_area=50)
    assert area == 40 * 45
    assert bounds == [30, 20, 75, 60]
    assert selected[3, 3] == 0


def test_shared_rgb_crop_unions_frames_and_preserves_originals(tmp_path: Path):
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir()
    color = pale_dough_rgb()
    records = []
    original_hashes = []
    for index, (x0, y0) in enumerate(((30, 25), (40, 30))):
        array = np.zeros((100, 120, 3), dtype=np.uint8)
        array[y0:y0 + 30, x0:x0 + 35] = color
        path = rgb_dir / f"frame{index}.png"
        Image.fromarray(array).save(path)
        digest = subject.sha256(path)
        records.append({"episode_id": subject.EPISODE_IDS[index], "role": "start",
                        "path": str(path.relative_to(tmp_path)), "sha256": digest})
        original_hashes.append(digest)

    cropped, manifest = subject.crop_rgb_frames(tmp_path, records, margin=10, minimum_area=50)

    assert manifest["union_bounds_xyxy"] == [30, 25, 75, 60]
    crop = manifest["square_crop_xyxy"]
    assert crop[2] - crop[0] == crop[3] - crop[1]
    assert all(row["crop_xyxy"] == crop for row in cropped)
    assert [subject.sha256(rgb_dir / f"frame{index}.png") for index in range(2)] == original_hashes


def test_square_crop_translates_inside_image_without_shrinking():
    crop = subject.square_crop([0, 0, 40, 60], 100, 80, margin=18)
    assert crop[0] == 0 and crop[1] == 0
    assert crop[2] - crop[0] == crop[3] - crop[1] == 80


def test_particle_renderer_transform_is_rotate_then_mirror():
    points = np.array([[0.2, 0.0, 0.8], [0.65, 0.0, 0.3]], dtype=np.float64)
    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    actual = subject.render_particle_image(points, bounds, size=101, point_radius=0)

    mask = np.zeros((101, 101), dtype=np.uint8)
    pixels = np.rint(points[:, (0, 2)] * 100).astype(int)
    mask[100 - pixels[:, 1], pixels[:, 0]] = 255
    reference = Image.new("RGB", (101, 101), subject.BACKGROUND_COLOR)
    reference.paste(subject.PARTICLE_COLOR, mask=Image.fromarray(mask, mode="L"))
    reference = ImageOps.mirror(reference.rotate(-90, resample=Image.Resampling.NEAREST))
    assert np.array_equal(np.asarray(actual), np.asarray(reference))


def test_topview_renderer_records_mirror_and_omits_tools(tmp_path: Path):
    frames = []
    for role, offset in (("start", 0.0), ("end", 0.1)):
        path = tmp_path / f"{role}.npy"
        np.save(path, np.array([[offset, 0.0, 0.0], [0.2 + offset, 0.0, 0.3]]))
        frames.append({
            "role": role, "local_frame": 0 if role == "start" else 2,
            "snapshot": path.name, "snapshot_sha256": subject.sha256(path),
            "tool_poses": [[1.0] * 7],
        })
    chains = {subject.CHAIN_A: [{"episode_id": subject.EPISODE_IDS[0],
                                "chunk": 1, "frames": frames}],
              subject.INDEPENDENT_C2: [], subject.REPLAY_C1_END: []}

    rendered, camera = subject.render_topviews(tmp_path, chains)

    assert len(rendered[subject.CHAIN_A][0]["frames"]) == 2
    assert camera["tool_geometry_rendered"] is False
    assert camera["post_render_rotation_deg"] == -90
    assert camera["post_render_horizontal_mirror"] is True
    assert camera["transform_order"] == "rotate_-90_then_mirror_left_right"


def test_four_row_layout_marks_unavailable_c1_cells_explicitly():
    rgb = image_records(subject.EPISODE_IDS)
    rendered = {
        subject.CHAIN_A: grouped_records(subject.EPISODE_IDS),
        subject.INDEPENDENT_C2: grouped_records(subject.EPISODE_IDS[1:]),
        subject.REPLAY_C1_END: grouped_records(subject.EPISODE_IDS[1:]),
    }
    cells = subject.figure_cells(rgb, rendered)
    assert len(cells) == 4 and all(len(row) == 4 for row in cells)
    assert [(row, column) for row in range(4) for column in range(4)
            if subject.is_not_applicable(cells[row][column])] == [(2, 0), (3, 0)]
    assert cells[2][0]["label"] == "N/A — begins at C2"
    assert cells[3][0]["label"] == "N/A — begins at C2"
    assert cells[2][1]["start"].endswith("chunk02_start.png")
    assert cells[3][1]["start"].endswith("chunk02_start.png")


def test_article_export_writes_pdf_png_and_finite_manifest(tmp_path: Path):
    try:
        importlib.import_module("matplotlib")
    except ImportError:
        pytest.skip("local Matplotlib binary is incompatible with installed NumPy")
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for episode in subject.EPISODE_IDS:
        for role in ("start", "end"):
            path = image_dir / f"{episode}_{role}.png"
            image = Image.new("RGB", (40, 40), "white")
            ImageDraw.Draw(image).ellipse((8, 8, 32, 32), fill=subject.PARTICLE_COLOR)
            image.save(path)
    rgb = image_records(subject.EPISODE_IDS, "images/")
    rendered = {
        subject.CHAIN_A: grouped_records(subject.EPISODE_IDS, "images/"),
        subject.INDEPENDENT_C2: grouped_records(subject.EPISODE_IDS[1:], "images/"),
        subject.REPLAY_C1_END: grouped_records(subject.EPISODE_IDS[1:], "images/"),
    }
    chains = {name: [] for name in rendered}
    result = subject.make_figure(
        tmp_path, rgb, chains, rendered,
        {"tool_geometry_rendered": False}, {"square_crop_xyxy": [0, 0, 40, 40]},
        {"all_within_tolerance": True}, PARAMETERS,
    )

    manifest_path = tmp_path / result["path"]
    manifest = json.loads(manifest_path.read_text(), parse_constant=lambda value: pytest.fail(value))
    for extension in ("pdf", "png"):
        metadata = manifest["outputs"][extension]
        path = tmp_path / metadata["path"]
        assert path.is_file() and subject.sha256(path) == metadata["sha256"]
    assert manifest["layout"]["png_dpi"] == 600
    assert manifest["layout"]["not_applicable_cells_zero_based"] == [[2, 0], [3, 0]]
    assert manifest["layout"]["state_transfer_annotations"] == [
        "carry {x, v, C, F, Jp}", "independent reconstruction", "C1 terminal state",
    ]
    with Image.open(tmp_path / manifest["outputs"]["png"]["path"]) as image:
        assert image.size == tuple(manifest["layout"]["png_dimensions_pixels"])


def test_stage_validation_allows_simulation_without_bag_and_requires_figure_inputs(tmp_path: Path):
    simulate = SimpleNamespace(
        stage="simulate", output_dir=subject.ROOT / "runs" / "new-test-run",
        dataset=tmp_path / "dataset.json", range_selection=tmp_path / "ranges.json",
        bag_dir=None, conversion_metadata=None,
    )
    subject.validate_inputs(simulate)
    figure = SimpleNamespace(
        stage="figure", output_dir=subject.ROOT / "runs" / "missing-test-run", dataset=None,
        range_selection=None, bag_dir=None, conversion_metadata=None,
    )
    with pytest.raises(ValueError, match="requires --bag-dir"):
        subject.validate_inputs(figure)


def test_generated_state_paths_are_relative_and_resolve_after_move(monkeypatch, tmp_path: Path):
    FakeStepper.instances.clear()
    monkeypatch.setattr(subject, "Stepper", FakeStepper)
    source = tmp_path / "source"
    source.mkdir()
    prepared = [fake_prepared(10.0 * index, context=index) for index in range(1, 5)]
    episodes = [fake_episode(index) for index in range(1, 5)]
    records, _ = subject.run_chain(subject.CHAIN_A, 0, prepared, episodes, source)
    assert not Path(records[0]["initial_state"]["path"]).is_absolute()
    moved = tmp_path / "moved"
    source.rename(moved)
    assert (moved / records[0]["initial_state"]["path"]).is_file()
