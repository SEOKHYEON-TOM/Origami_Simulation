"""Independent frame, finite joint, trajectory gap and snapshot contracts."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/origami-tess-mpl")
import sys, unittest, tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import origami_forward_kinematics as fk
from origami_tesellator_1D import OrigamiTesellator1D


class StripFrames(unittest.TestCase):
    def make(self, alphas=(70, 20, 110, 160), *, n=3, iout=2, sigma=1, **kwargs):
        kwargs.setdefault("contact", False)
        return OrigamiTesellator1D(
            [dict(alphas=alphas, iout=iout, sigma=sigma)], n, verbose=False, **kwargs
        )

    def test_fk_half_fold_corrections_use_actual_input_and_output(self):
        s = self.make(n=4, iout=3)
        seen = []
        original = fk.get_se3_frame

        def record(origin, axis, adj, rho, is_base=True):
            seen.append((rho, is_base))
            return original(origin, axis, adj, rho, is_base)

        with patch.object(fk, "get_se3_frame", side_effect=record):
            r = fk.extract_relative_se3(s, 45)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(seen[0][0], np.pi / 4)
        self.assertAlmostEqual(seen[-1][0], s.get_fold_angles(3)[3])
        self.assertNotAlmostEqual(s.rhos[0][0], seen[0][0])
        np.testing.assert_allclose(r["R_rel"].T @ r["R_rel"], np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r["R_rel"]), 1.0, places=12)
        a, b = r["output_line"]
        axis = (b - a) / np.linalg.norm(b - a)
        np.testing.assert_allclose(axis, r["R_rel"][:, 0], atol=1e-12)

    def test_relative_se3_is_invariant_under_global_rigid_motion(self):
        for a, d in [
            ([70, 20, 110, 160], 45),
            ([61, 91, 136, 76], 65),
            ([70, 80, 70, 100], 80),
        ]:
            with self.subTest(alphas=a):
                s = self.make(a, n=2)
                r = fk.extract_relative_se3(s, d)
                self.assertIsNotNone(r, s.last_error)
                Q, _ = np.linalg.qr(np.array([[1.0, 2, 4], [2, 1, 7], [3, 5, 1]]))
                if np.linalg.det(Q) < 0:
                    Q[:, 0] *= -1
                s.global_vertices[:] = [v @ Q.T + [1, -4, 2] for v in s.global_vertices]
                moved = fk.extract_relative_se3(s, d, compute=False)
                np.testing.assert_allclose(moved["P_rel"], r["P_rel"], atol=1e-11)
                np.testing.assert_allclose(moved["R_rel"], r["R_rel"], atol=1e-11)

    def test_frame_degeneracy_is_explicit_and_frames_are_proper(self):
        for base in (True, False):
            R, _ = fk.get_se3_frame([0, 0, 0], [2, 1, 4], [1, 3, -2], 0.8, base)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)
        with self.assertRaises(ValueError):
            fk.get_se3_frame([0, 0, 0], [0, 0, 0], [1, 0, 0], 0.1)
        with self.assertRaises(ValueError):
            fk.get_se3_frame([0, 0, 0], [1, 0, 0], [2, 0, 0], 0.1)

    def test_frames_bisect_the_actual_neighboring_panel_normals(self):
        for a in ([70, 20, 110, 160], [61, 91, 136, 76], [70, 80, 70, 100]):
            for iout in (1, 2, 3):
                s = self.make(a, n=1, iout=iout)
                self.assertEqual(s.compute_strip_kinematics(1.4), 1, s.last_error)
                gv = s.global_vertices[0]
                mapping = s.cells[0]["CLV"]
                q = s.get_fold_angles(0)
                rays = [gv[mapping[k]] - gv[0] for k in range(4)]

                def unit(x):
                    return x / np.linalg.norm(x)

                for k, base in ((0, True), (iout, False)):
                    before = unit(np.cross(rays[(k - 1) % 4], rays[k]))
                    after = unit(np.cross(rays[k], rays[(k + 1) % 4]))
                    expected = unit(before + after)
                    adjacent = rays[1] if base else rays[(k - 1) % 4]
                    R, _ = fk.get_se3_frame(gv[0], rays[k], adjacent, q[k], base)
                    np.testing.assert_allclose(R[:, 2], expected, atol=1e-11)

    def test_axis_policy_reports_mismatched_finite_edge_without_changing_lengths(self):
        s = self.make(n=2, iout=3, lengths=(1, 1, 2))
        self.assertEqual(s.compute_strip_kinematics(0.7), 2, s.last_error)
        j = s.last_result.joints[0]
        self.assertAlmostEqual(j["edge_error"], 1.0)
        self.assertFalse(j["edge_shared"])
        self.assertEqual(j["policy"], "axis")
        self.assertLess(j["axis_error"], 1e-12)
        self.assertLess(j["normal_error"], 1e-12)
        self.assertLess(j["origin_error"], 1e-12)
        cell = s.cells[0]
        self.assertAlmostEqual(
            np.linalg.norm(s.global_vertices[0][cell["CLV"][3]]), 2.0
        )

    def test_shared_edge_policy_rejects_mismatch_and_accepts_compatible_scale(self):
        bad = self.make(n=2, iout=3, lengths=(1, 1, 2), joint_policy="shared_edge")
        self.assertEqual(bad.compute_strip_kinematics(0.7), 1)
        self.assertEqual(bad.last_result.status, "JOINT_INCOMPATIBLE")
        self.assertEqual(bad.last_result.termination["unit_index"], 1)
        good = self.make(
            n=4, lengths=(1, 2, 1), scale_factor=2, joint_policy="shared_edge"
        )
        self.assertEqual(good.compute_strip_kinematics(0.7), 4, good.last_error)
        self.assertTrue(all(j["edge_shared"] for j in good.last_result.joints))
        for j in good.last_result.joints:
            np.testing.assert_allclose(
                j["rotation"].T @ j["rotation"], np.eye(3), atol=1e-12
            )
            self.assertAlmostEqual(j["determinant"], 1.0, places=12)

    def test_antiparallel_normals_keep_the_input_axis_reversed(self):
        s = self.make(n=2)
        local = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
        previous = np.array([[0.0, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0], [-1, 0, 0]])
        mapping = {i: i + 1 for i in range(4)}
        gv = s._place_unit_global(local, previous, mapping, 1, mapping)
        np.testing.assert_allclose(gv[1] - gv[0], [-1, 0, 0], atol=1e-12)
        normal = np.cross(gv[1] - gv[0], gv[2] - gv[0])
        np.testing.assert_allclose(normal, [0, 0, -1], atol=1e-12)

    def test_failed_samples_keep_nan_breaks_and_segment_identity(self):
        s = self.make([40, 80, 50, 110], n=1)
        rows = fk.run_trajectory_sweep(s, -100, 100, steps=31, track=False)
        self.assertEqual(len(rows), 31)
        self.assertTrue(any(not r["valid"] for r in rows))
        self.assertTrue(any(r["valid"] and r["rho0_deg"] > 0 for r in rows))
        self.assertTrue(any(r["valid"] and r["rho0_deg"] < 0 for r in rows))
        self.assertTrue(all(np.isnan(r["P_rel"]).all() for r in rows if not r["valid"]))
        self.assertEqual({r["segment_id"] for r in rows if r["valid"]}, {1, 2})
        compact = fk.run_trajectory_sweep(
            s, -100, 100, steps=31, track=False, include_failures=False
        )
        self.assertTrue(all(r["valid"] for r in compact))
        self.assertEqual({r["segment_id"] for r in compact}, {1, 2})

    def test_temporal_sweep_cannot_jump_to_a_disconnected_domain(self):
        s = self.make([40, 80, 50, 110], n=1)
        rows = fk.run_trajectory_sweep(s, -100, 100, steps=31, track=True)
        self.assertTrue(any(r["valid"] for r in rows))
        self.assertFalse(any(r["valid"] and r["rho0_deg"] > 0 for r in rows))
        self.assertTrue(any(r["status"] == "DRIVE_EXTREMUM" for r in rows))

    def test_partial_fk_and_insufficient_planarity_are_not_reported_as_success(self):
        s = self.make([40, 80, 50, 110])
        self.assertIsNone(fk.extract_relative_se3(s, 0.0))
        ok, normal, error = fk.check_planarity([[0, 0, 0], [1, 0, 0], [np.nan] * 3])
        self.assertFalse(ok)
        self.assertTrue(np.isnan(normal).all())
        self.assertTrue(np.isnan(error))

    def test_frame_failure_stays_a_numerical_gap_in_the_trajectory(self):
        s = self.make(n=1)
        original = fk.get_se3_frame

        def frame(origin, axis, adj, rho, is_base=True):
            if is_base and abs(rho - np.pi / 6) < 1e-9:
                raise ValueError("injected frame degeneracy")
            return original(origin, axis, adj, rho, is_base)

        with patch.object(fk, "get_se3_frame", side_effect=frame):
            rows = fk.run_trajectory_sweep(s, 20, 40, steps=3)
        self.assertEqual([r["status"] for r in rows], ["OK", "NUMERICAL_FAILURE", "OK"])
        self.assertTrue(np.isnan(rows[1]["P_rel"]).all())

    def test_mat_export_uses_common_selector_and_preserves_accepted_state(self):
        s = self.make(n=2)
        s.compute_strip_kinematics(0.6)
        s.update(0.8)
        old = s.last_result
        folds = s.rhos.copy()
        vertices = np.array(s.global_vertices)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "strip.mat"
            data = s.export_to_mat(path, 45)
            mat = loadmat(path, simplify_cells=True)
            self.assertEqual(mat["meta"]["valid_units"], 2)
            self.assertEqual(mat["meta"]["rho0_deg"], 45.0)
            self.assertEqual(mat["meta"]["global_contact_checked"], 0)
            self.assertGreaterEqual(mat["faces"].min(), 1)
            self.assertLessEqual(mat["faces"].max(), len(mat["vertices"]))
            self.assertAlmostEqual(data["fold_angles"][0, 0], np.pi / 4)
        self.assertIs(s.last_result, old)
        np.testing.assert_array_equal(folds, s.rhos)
        np.testing.assert_array_equal(vertices, s.global_vertices)


if __name__ == "__main__":
    unittest.main()
