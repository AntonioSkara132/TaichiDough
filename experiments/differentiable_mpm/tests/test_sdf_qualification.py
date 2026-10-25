"""Independent analytic geometry checks and parity with the actual Taichi sampler."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.differentiable_mpm.qualify_tool_sdf import (
    interpolate_field, qualify_field, qualify_resolution, qualification_identity,
)


def box_triangles(half, center=(0., 0., 0.), rotation=None):
    vertices = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                         [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float) * half
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
                      [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5]])
    rotation = np.eye(3) if rotation is None else rotation
    return (vertices @ rotation.T + center)[faces]


class AnalyticBox:
    def __init__(self, half, center=(0., 0., 0.), rotation=None):
        self.half = np.asarray(half)
        self.center = np.asarray(center)
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation)

    def query(self, points):
        local = (np.asarray(points) - self.center) @ self.rotation
        q = np.abs(local) - self.half
        outside = np.maximum(q, 0)
        length = np.linalg.norm(outside, axis=1)
        distance = length + np.minimum(np.max(q, axis=1), 0)
        normals = np.zeros_like(local)
        exterior = length > 0
        normals[exterior] = outside[exterior] * np.sign(local[exterior]) / length[exterior, None]
        index = np.argmax(q[~exterior], axis=1)
        rows = np.arange(np.count_nonzero(~exterior))
        interior_normals = np.zeros((len(rows), 3))
        interior_normals[rows, index] = np.where(local[~exterior][rows, index] >= 0, 1., -1.)
        normals[~exterior] = interior_normals
        return distance, normals @ self.rotation.T


def sampled_box(query, resolution=33, minimum=(-.012, -.012, -.012), spacing=(.00075,) * 3):
    minimum, spacing = np.array(minimum, dtype=np.float32), np.array(spacing, dtype=np.float32)
    indices = np.indices((resolution,) * 3).reshape(3, -1).T
    distance, normal = query.query(minimum + indices * spacing)
    return (distance.astype(np.float32).reshape((resolution,) * 3),
            normal.astype(np.float32).reshape((resolution,) * 3 + (3,)), minimum, spacing)


def extruded_polygon(polygon, cap_faces, half_z):
    polygon = np.asarray(polygon, dtype=float)
    lower = np.column_stack((polygon, np.full(len(polygon), -half_z)))
    upper = np.column_stack((polygon, np.full(len(polygon), half_z)))
    triangles = [lower[list(face)[::-1]] for face in cap_faces]
    triangles.extend(upper[list(face)] for face in cap_faces)
    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        triangles.extend(([lower[i], lower[j], upper[j]], [lower[i], upper[j], upper[i]]))
    return np.asarray(triangles)


class AnalyticPrism:
    """Exact polygon-distance oracle, extruded as a Cartesian product with z."""
    def __init__(self, polygon, half_z):
        self.polygon = np.asarray(polygon, dtype=float)
        self.half_z = half_z

    def query(self, points):
        points = np.asarray(points, dtype=float)
        xy = points[:, :2]
        a = self.polygon
        b = np.roll(a, -1, axis=0)
        edge = b - a
        fraction = np.clip(np.sum((xy[:, None] - a) * edge, axis=2) / np.sum(edge * edge, axis=1), 0, 1)
        nearest = a + fraction[:, :, None] * edge
        difference = xy[:, None] - nearest
        distances = np.linalg.norm(difference, axis=2)
        indices = np.argmin(distances, axis=1)
        rows = np.arange(len(points))
        distance = distances[rows, indices]
        inside = np.zeros(len(points), dtype=bool)
        for start, end in zip(a, b):
            if start[1] == end[1]:
                continue
            intersects = ((start[1] > xy[:, 1]) != (end[1] > xy[:, 1]))
            ray_x = (end[0] - start[0]) * (xy[:, 1] - start[1]) / (end[1] - start[1]) + start[0]
            inside ^= intersects & (xy[:, 0] < ray_x)
        sign = np.where(inside, -1., 1.)
        planar_normal = np.zeros((len(points), 2))
        nonzero = distance > 1e-15
        planar_normal[nonzero] = (sign[nonzero, None] * difference[rows[nonzero], indices[nonzero]]
                                  / distance[nonzero, None])
        outward = np.column_stack((edge[:, 1], -edge[:, 0])) / np.linalg.norm(edge, axis=1)[:, None]
        planar_normal[~nonzero] = outward[indices[~nonzero]]
        distance *= sign
        z_distance = np.abs(points[:, 2]) - self.half_z
        components = np.column_stack((distance, z_distance))
        positive = np.maximum(components, 0)
        length = np.linalg.norm(positive, axis=1)
        signed = length + np.minimum(np.max(components, axis=1), 0)
        normal = np.zeros((len(points), 3))
        exterior = length > 0
        normal[exterior, :2] = planar_normal[exterior] * positive[exterior, 0, None] / length[exterior, None]
        normal[exterior, 2] = np.sign(points[exterior, 2]) * positive[exterior, 1] / length[exterior]
        side = (~exterior) & (distance >= z_distance)
        cap = (~exterior) & ~side
        normal[side, :2] = planar_normal[side]
        normal[cap, 2] = np.where(points[cap, 2] >= 0, 1., -1.)
        return signed, normal


class OracleSamples:
    """Bypass interpolation to test qualification policy against exact geometry."""
    arrays = {}
    resolution = 1024
    minimum = np.full(3, -.05)
    spacing = np.full(3, .00025)

    def __init__(self, query, shift=0.):
        self.query = query
        self.shift = shift

    def sample(self, points, dtype):
        distance, normals = self.query.query(points)
        return np.ones(len(points), dtype=bool), distance + self.shift, normals


class QualificationTests(unittest.TestCase):
    def test_exact_concave_solid_accepts_true_extra_crossings(self):
        from experiments.differentiable_mpm.qualify_tool_sdf import _qualify_samples
        from experiments.differentiable_mpm.triangle_sdf import TriangleDistanceQuery

        polygon = [(-.02, -.02), (.02, -.02), (.02, .02), (.001, .02),
                   (.001, -.01), (-.001, -.01), (-.001, .02), (-.02, .02)]
        caps = [(1, 2, 3), (1, 3, 4), (0, 1, 4), (0, 4, 5), (7, 0, 5), (5, 6, 7)]
        triangles = extruded_polygon(polygon, caps, .005)
        TriangleDistanceQuery(triangles)  # The oracle fixture satisfies production mesh checks.
        query = AnalyticPrism(polygon, .005)
        report = _qualify_samples(triangles, OracleSamples(query), .004, query=query)
        self.assertTrue(report["passed"], report["failures"])
        extra = [row for row in report["profiles"] if row["expected_crossings"] == 3]
        self.assertTrue(extra)
        self.assertTrue(all(row["zero_crossings"] == 3 for row in extra))
        self.assertEqual(report["worst"]["distance_error_m"], 0.)

    def test_oblique_ray_displacement_is_not_geometric_boundary_error(self):
        from experiments.differentiable_mpm.qualify_tool_sdf import _qualify_samples
        from experiments.differentiable_mpm.triangle_sdf import TriangleDistanceQuery

        polygon = [(0., 0.), (.1, 0.), (.1, .001)]
        triangles = extruded_polygon(polygon, [(0, 1, 2)], .01)
        TriangleDistanceQuery(triangles)
        query = AnalyticPrism(polygon, .01)
        report = _qualify_samples(triangles, OracleSamples(query, shift=2e-6), .004, query=query)
        self.assertTrue(report["passed"], report["failures"])
        self.assertLess(report["worst"]["boundary_distance_error_m"], 2.1e-6)
        self.assertGreater(report["worst"]["ray_crossing_displacement_m"], .00019)
        self.assertTrue(any(not row["thickness_eligible"] for row in report["profiles"]))
        self.assertTrue(any(row["thickness_eligible"] for row in report["profiles"]))

    def test_varying_support_normal_errors_are_not_excused(self):
        from experiments.differentiable_mpm.qualify_tool_sdf import _qualify_samples, _normal_support_spread

        class VaryingNormalSamples(OracleSamples):
            def sample(self, points, dtype):
                valid, distance, normals = super().sample(points, dtype)
                varying = _normal_support_spread(self, points, normals, self.query, dtype) > 1.
                normals[varying] = 0.
                return valid, distance, normals

        polygon = [(0., 0.), (.1, 0.), (.1, .001)]
        triangles = extruded_polygon(polygon, [(0, 1, 2)], .01)
        query = AnalyticPrism(polygon, .01)
        report = _qualify_samples(triangles, VaryingNormalSamples(query), .004, query=query)
        self.assertTrue(report["regular_support_normal_passed"])
        self.assertFalse(report["all_probe_normal_passed"])
        self.assertFalse(report["passed"])
        self.assertGreater(report["coverage"]["varying_support_normal_probes"], 0)
        self.assertGreater(report["coverage"]["regular_support_normal_probes"], 0)

    def test_stable_support_profile_excuses_only_varying_support(self):
        from experiments.differentiable_mpm.qualify_tool_sdf import _qualify_samples, _normal_support_spread

        class NormalSamples(OracleSamples):
            def __init__(self, query, rotate_stable):
                super().__init__(query)
                self.rotate_stable = rotate_stable

            def sample(self, points, dtype):
                valid, distance, normals = super().sample(points, dtype)
                varying = _normal_support_spread(self, points, normals, self.query, dtype) > 1.
                normals[varying] = 0.
                if self.rotate_stable:
                    normals[~varying] = np.roll(normals[~varying], 1, axis=1)
                return valid, distance, normals

        polygon = [(0., 0.), (.1, 0.), (.1, .001)]
        triangles = extruded_polygon(polygon, [(0, 1, 2)], .01)
        query = AnalyticPrism(polygon, .01)
        report = _qualify_samples(triangles, NormalSamples(query, False), .004, query=query,
                                  normal_acceptance="stable-support")
        self.assertTrue(report["passed"], report["failures"])
        self.assertFalse(report["all_probe_normal_passed"])
        self.assertEqual(report["tolerances"]["normal_acceptance"], "stable-support")
        report = _qualify_samples(triangles, NormalSamples(query, True), .004, query=query,
                                  normal_acceptance="stable-support")
        self.assertFalse(report["passed"])
        self.assertFalse(report["regular_support_normal_passed"])

    def test_shallow_material_rule_accepts_wall_hugging_profiles_but_not_lost_plates(self):
        from experiments.differentiable_mpm.qualify_tool_sdf import _qualify_samples

        class ShallowOutside(OracleSamples):
            def __init__(self, query, depth):
                super().__init__(query)
                self.depth = depth

            def sample(self, points, dtype):
                valid, distance, normals = super().sample(points, dtype)
                return valid, np.where((distance < 0) & (distance > -self.depth), 1e-6, distance), normals

        slope, length = np.tan(np.deg2rad(17.)), .01
        polygon = [(0., 0.), *[(x, 0.) for x in (.0002, .0004, .0006, .001, .002, .004, .007)],
                   (length, 0.), (length, length * slope)]
        top = len(polygon) - 1
        wedge = extruded_polygon(polygon, [(top, i, i + 1) for i in range(top - 1)], .01)
        query = AnalyticPrism(polygon, .01)
        strict = _qualify_samples(wedge, ShallowOutside(query, 3e-5), .004, query=query,
                                  normal_acceptance="stable-support")
        self.assertFalse(strict["passed"])
        relaxed = _qualify_samples(wedge, ShallowOutside(query, 3e-5), .004, query=query,
                                   normal_acceptance="stable-support", crossing_acceptance="below-tolerance")
        self.assertTrue(relaxed["passed"], relaxed["failures"])
        self.assertGreater(relaxed["coverage"]["accepted_shallow_profiles"], 0)
        too_deep = _qualify_samples(wedge, ShallowOutside(query, 2e-4), .004, query=query,
                                    normal_acceptance="stable-support", crossing_acceptance="below-tolerance")
        self.assertFalse(too_deep["passed"])

        plate = [(0., 0.), (.02, 0.), (.02, .0004), (0., .0004)]
        query = AnalyticPrism(plate, .01)
        lost = _qualify_samples(extruded_polygon(plate, [(0, 1, 2), (0, 2, 3)], .01), ShallowOutside(query, 3e-4),
                                .004, query=query, distance_tolerance=1e-3, normal_acceptance="stable-support",
                                crossing_acceptance="below-tolerance")
        self.assertFalse(lost["passed"])
        self.assertTrue(any("negative interior is missing" in row["reasons"] for row in lost["failures"]))

    def test_qualification_profiles(self):
        from experiments.differentiable_mpm.qualify_tool_sdf import qualification_identity, qualification_settings

        self.assertEqual(qualification_settings(), {"distance_tolerance": 1e-4, "normal_acceptance": "all-evaluated",
                                                    "crossing_acceptance": "exact"})
        self.assertEqual(qualification_settings("grid-relative-v1", 1 / 48),
                         {"distance_tolerance": .05 / 48, "normal_acceptance": "stable-support",
                          "crossing_acceptance": "below-tolerance"})
        for profile, spacing in (("grid-relative-v1", None), ("grid-relative-v1", 0.), ("unknown", .02)):
            with self.subTest(profile=profile, spacing=spacing), self.assertRaises(ValueError):
                qualification_settings(profile, spacing)
        self.assertNotEqual(qualification_identity(), qualification_identity("grid-relative-v1", 1 / 48))
        self.assertNotEqual(qualification_identity("grid-relative-v1", 1 / 48), qualification_identity("grid-relative-v1", 1 / 64))

    def test_refinement_queries_actual_sampler(self):
        from experiments.differentiable_mpm.qualify_tool_sdf import _refined_crossings

        class QuadraticSamples:
            def sample(self, points, dtype):
                return np.ones(len(points), dtype=bool), points[:, 0] ** 2 - .04, np.zeros_like(points)

        roots = _refined_crossings(np.array([0., .5]), np.array([-.04, .21]), QuadraticSamples(),
                                   np.zeros(3), np.array([1., 0., 0.]), np.float64, 1e-9)
        np.testing.assert_allclose(roots, [.2], rtol=0., atol=1e-9)

    def test_resolved_box_has_correct_crossings_and_normals(self):
        query = AnalyticBox([.006, .006, .006])
        volume = sampled_box(query, resolution=97, spacing=(.00025,) * 3)
        report = qualify_field(box_triangles(query.half), volume, .001, query=query)
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["coverage"]["profiles"], 12)
        self.assertGreater(report["coverage"]["regular_normal_probes"], 0)
        self.assertLess(report["worst"]["zero_crossing_error_m"], 1e-8)
        self.assertEqual(len(report["arrays"]["distances"]["sha256"]), 64)

    def test_real_triangle_builder_and_command_outputs(self):
        from experiments.differentiable_mpm.triangle_sdf import build_tool_sdf
        from experiments.differentiable_mpm.qualify_tool_sdf import main
        from scripts.build_tool_collision_meshes import write_binary_stl

        triangles = box_triangles([.006] * 3)
        volume = build_tool_sdf(triangles, 64, .001)
        report = qualify_field(triangles, volume, .001)
        self.assertTrue(report["passed"], report["failures"])
        self.assertEqual(report["precisions"], ["f32", "f64"])
        scratch = Path("/home/antonio/.claude/jobs/1f2c3881/tmp")
        with tempfile.TemporaryDirectory(dir=scratch, prefix="sdf-qualification-test-") as directory:
            root = Path(directory)
            mesh = root / "box.stl"
            write_binary_stl(mesh, triangles.reshape(-1, 3), np.arange(36).reshape(-1, 3))
            args = ["--mesh", str(mesh), "--scale", "1", "--padding", ".001", "--resolution", "64"]
            passed = root / "passed"
            self.assertEqual(main([*args, "--output-dir", str(passed)]), 0)
            self.assertTrue((passed / "qualified_field.npz").is_file())
            self.assertTrue(json.loads((passed / "qualification.json").read_text())["passed"])
            with self.assertRaises(FileExistsError):
                main([*args, "--output-dir", str(passed)])
            invalid = root / "invalid.stl"
            write_binary_stl(invalid, triangles[:-1].reshape(-1, 3), np.arange(33).reshape(-1, 3))
            failed = root / "failed"
            self.assertEqual(main([*args, "--mesh", str(invalid), "--output-dir", str(failed)]), 2)
            self.assertFalse((failed / "qualified_field.npz").exists())
            self.assertFalse(json.loads((failed / "qualification.json").read_text())["passed"])

    def test_sparse_preflight_matches_actual_dense_profiles(self):
        from experiments.differentiable_mpm.triangle_sdf import build_tool_sdf, TriangleDistanceQuery

        angle = .37
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        cases = [(box_triangles([.006] * 3), 64),
                 (box_triangles([.006, .006, .0002], center=[.0013, -.0021, .0007], rotation=rotation), 24)]
        outcomes = []
        for triangles, resolution in cases:
            with self.subTest(resolution=resolution):
                query = TriangleDistanceQuery(triangles)
                volume = build_tool_sdf(triangles, resolution, .001)
                dense = qualify_field(triangles, volume, .001, query=query)
                sparse = qualify_resolution(triangles, resolution, .001, query=query)
                self.assertTrue(sparse["preflight"])
                self.assertTrue(sparse["dense_qualification_required"])
                self.assertEqual(sparse["arrays"], {})
                self.assertEqual(sparse["passed"], dense["passed"])
                self.assertEqual(sparse["precisions"], ["f32", "f64"])
                self.assertEqual(sparse["coverage"], dense["coverage"])
                self.assertEqual(sparse["failures"], dense["failures"])
                self.assertEqual(sparse["profiles"], dense["profiles"])
                self.assertEqual(sparse["worst"], dense["worst"])
                self.assertEqual(sparse["memory"]["dense_array_bytes"], 16 * resolution ** 3)
                self.assertLessEqual(sparse["memory"]["largest_unique_corner_batch"], 8 * 87)
                outcomes.append(sparse["passed"])
        self.assertEqual(outcomes, [True, False])

    def test_sparse_preflight_does_not_allocate_dense_resolution(self):
        from unittest.mock import patch
        from experiments.differentiable_mpm.triangle_sdf import TriangleDistanceQuery

        triangles = box_triangles([.006] * 3)
        query = TriangleDistanceQuery(triangles)
        allocate = np.empty

        def bounded_empty(shape, *args, **kwargs):
            self.assertLess(np.prod(shape), 1000000, "preflight attempted a large dense allocation")
            return allocate(shape, *args, **kwargs)

        with patch("numpy.empty", side_effect=bounded_empty):
            report = qualify_resolution(triangles, 4096, .001, query=query)
        self.assertTrue(report["preflight"])
        self.assertEqual(report["memory"]["dense_array_bytes"], 16 * 4096 ** 3)
        self.assertLessEqual(report["memory"]["largest_unique_corner_batch"], 8 * 87)

    def test_exact_samples_can_lose_subvoxel_slab(self):
        query = AnalyticBox([.006, .006, .0001], center=[0., 0., .000375])
        volume = sampled_box(query)
        self.assertGreaterEqual(volume[0].min(), 0.)
        report = qualify_field(box_triangles(query.half, query.center), volume, .001, query=query)
        self.assertFalse(report["passed"])
        self.assertTrue(any("negative interior is missing" in row.get("reasons", []) for row in report["failures"]))

    def test_rotated_thin_slab_is_not_accepted_only_for_exact_samples(self):
        angle = .37
        rotation = np.array([[np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
                             [-np.sin(angle), 0., np.cos(angle)]])
        query = AnalyticBox([.006, .006, .0001], center=[0., 0., .0003], rotation=rotation)
        report = qualify_field(box_triangles(query.half, query.center, rotation), sampled_box(query), .001, query=query)
        self.assertFalse(report["passed"])

    def test_reversed_and_zero_normals_are_rejected(self):
        query = AnalyticBox([.006] * 3)
        volume = sampled_box(query)
        for factor in (0., -1.):
            with self.subTest(factor=factor):
                bad = (volume[0], volume[1] * factor, volume[2], volume[3])
                report = qualify_field(box_triangles(query.half), bad, .001, query=query)
                self.assertFalse(report["passed"])
                self.assertGreaterEqual(report["worst"]["normal_error_degrees"], 89.)

    def test_missing_bounds_and_bad_settings_fail(self):
        query = AnalyticBox([.006] * 3)
        volume = sampled_box(query)
        report = qualify_field(box_triangles(query.half), volume, .02, query=query)
        self.assertFalse(report["passed"])
        self.assertTrue(any("profile extends outside field" in row.get("reasons", []) for row in report["failures"]))
        for options in ({"distance_tolerance": 0}, {"max_faces": 0}, {"normal_tolerance_degrees": np.nan}):
            with self.assertRaises(ValueError):
                qualify_field(box_triangles(query.half), volume, .001, query=query, **options)

    def test_interpolation_domain_threshold_and_zero_normal(self):
        query = AnalyticBox([.006] * 3)
        volume = sampled_box(query)
        points = np.array([volume[2], volume[2] + 32 * volume[3], [.03, 0., 0.]])
        valid, distance, normal = interpolate_field(volume, points)
        np.testing.assert_array_equal(valid, [True, True, False])
        self.assertEqual(distance[-1], 1e6)
        np.testing.assert_array_equal(normal[-1], [0., 0., 0.])
        zero = (volume[0], np.zeros_like(volume[1]), *volume[2:])
        self.assertTrue(np.isfinite(interpolate_field(zero, points)[2]).all())
        self.assertEqual(qualification_identity()["distance_tolerance_m"], 1e-4)


class SamplerParityTests(unittest.TestCase):
    def test_actual_sampler_values_and_fixed_cell_derivative(self):
        import taichi as ti
        from experiments.differentiable_mpm.solver import Stepper
        from experiments.differentiable_mpm.state import SDFData, SimulationConfig

        @ti.data_oriented
        class Probe:
            def __init__(self, dtype):
                self.point = ti.Vector.field(3, dtype, shape=(), needs_grad=True)
                self.distance = ti.field(dtype, shape=())
                self.normal = ti.Vector.field(3, dtype, shape=())
                self.valid = ti.field(ti.i32, shape=())
                self.loss = ti.field(dtype, shape=(), needs_grad=True)

            # Taichi 1.7.4 f32 adjoints change when this diagnostic vector write
            # shares the differentiated kernel. Test primal output separately.
            @ti.kernel
            def diagnostics(self, solver: ti.template()):
                valid, distance, normal = solver._sdf_at(0, self.point[None])
                self.valid[None] = valid
                self.distance[None] = distance
                self.normal[None] = normal

            @ti.kernel
            def evaluate(self, solver: ti.template()):
                valid, distance, normal = solver._sdf_at(0, self.point[None])
                self.loss[None] = distance + .4 * normal[0] - .2 * normal[2]

        for precision, dtype in (("f32", np.float32), ("f64", np.float64)):
            with self.subTest(precision=precision):
                ti.reset()
                ti.init(arch=ti.cpu, default_fp=ti.f64 if precision == "f64" else ti.f32,
                        cpu_max_num_threads=1, offline_cache=False)
                grid = np.indices((4,) * 3).astype(np.float32) * .1
                distances = grid[0] + .2 * grid[1] - .1 * grid[2]
                normals = np.stack((np.ones_like(grid[0]), .2 * grid[0], .1 * grid[2]), axis=-1)
                volume = (distances, normals, np.zeros(3, np.float32), np.full(3, .1, np.float32))
                sdf = SDFData(*(np.stack([value, value]) for value in volume))
                solver = Stepper(SimulationConfig(n_particles=1, grid=8, precision=precision, tool_collision="sdf"),
                                 capacity=2, sdf=sdf)
                probe = Probe(ti.f64 if precision == "f64" else ti.f32)
                point = np.array([.137, .146, .173], dtype=dtype)
                probe.point[None] = point
                probe.diagnostics(solver)
                probe.evaluate(solver)
                expected = interpolate_field(volume, point[None], dtype=dtype)
                self.assertEqual(bool(probe.valid[None]), bool(expected[0][0]))
                np.testing.assert_allclose(probe.distance[None], expected[1][0], rtol=1e-6, atol=1e-8)
                np.testing.assert_allclose(probe.normal[None], expected[2][0], rtol=1e-6, atol=1e-8)
                probe.point.grad[None] = [0., 0., 0.]
                probe.loss.grad[None] = 1.
                probe.evaluate.grad(solver)
                gradient = np.asarray(probe.point.grad[None])
                h = 1e-5 if precision == "f32" else 1e-6
                finite = []
                for axis in range(3):
                    values = []
                    for sign in (1., -1.):
                        changed = point.astype(float).copy()
                        changed[axis] += sign * h
                        _, d, n = interpolate_field(volume, changed[None])
                        values.append(d[0] + .4 * n[0, 0] - .2 * n[0, 2])
                    finite.append((values[0] - values[1]) / (2 * h))
                runtime_finite = []
                for step in (1e-3, 3e-4, 1e-4):
                    estimate = []
                    for axis in range(3):
                        values, positions = [], []
                        for sign in (1., -1.):
                            changed = point.copy()
                            changed[axis] = dtype(float(point[axis]) + sign * step)
                            positions.append(float(changed[axis]))
                            probe.point[None] = changed
                            probe.evaluate(solver)
                            values.append(probe.loss[None])
                        estimate.append((values[0] - values[1]) / (positions[0] - positions[1]))
                    runtime_finite.append(estimate)
                np.testing.assert_allclose(gradient, finite, rtol=2e-5, atol=2e-6,
                                           err_msg=f"runtime finite differences: {runtime_finite}")
                best_runtime_error = min(np.max(np.abs(np.asarray(value) - finite)) for value in runtime_finite)
                self.assertLessEqual(best_runtime_error, 5e-5 if precision == "f32" else 1e-6,
                                     f"runtime finite differences: {runtime_finite}")
        ti.reset()


if __name__ == "__main__":
    unittest.main()
