"""Coordinate invariants and rendered geometry for backbone/trajectory plots."""
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/origami-fk-motion")
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import origami_forward_kinematics as fk
from origami_tesellator_1D import OrigamiTesellator1D


class MotionPlots(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def strip(self, alphas=(70, 60, 110, 120), n=3):
        return OrigamiTesellator1D([dict(alphas=alphas, iout=2, sigma=1)],
                                  n, contact=False, verbose=False)

    def test_fixed_reference_centers_frames_and_tip_correspond(self):
        rows = fk.sample_backbone_motion(self.strip(), [30, 60, 100])
        first = rows[0]
        R0, P0 = first["R_base_world"], first["P_base_world"]
        for row in rows:
            self.assertTrue(row["valid"])
            self.assertEqual(row["unit_centers"].shape, (3, 3))
            self.assertEqual(row["backbone"].shape, (4, 3))
            np.testing.assert_allclose(row["unit_centers"] @ R0.T + P0,
                                       row["unit_centers_world"], atol=1e-12)
            np.testing.assert_allclose(row["backbone"][:-1], row["unit_centers"])
            np.testing.assert_allclose(row["backbone"][-1], row["output_line"][1])
            np.testing.assert_allclose(row["P_rel"], row["unit_centers"][-1])
            np.testing.assert_allclose(row["P_input"], row["unit_centers"][0])
            for line, rotation in ((row["input_line"], row["R_input"]),
                                    (row["output_line"], row["R_rel"])):
                axis = line[1] - line[0]
                np.testing.assert_allclose(axis / np.linalg.norm(axis), rotation[:, 0],
                                           atol=1e-12)
                np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
            self.assertEqual(row["reference_rho0_deg"], 30)
        self.assertGreater(np.linalg.norm(rows[-1]["R_input"] - np.eye(3)), 0.1)

    def test_current_reference_preserves_legacy_sweep(self):
        legacy = fk.run_trajectory_sweep(self.strip(), 30, 90, steps=3)
        current = fk.sample_backbone_motion(self.strip(), [30, 60, 90], reference="current")
        for old, new in zip(legacy, current):
            for key in ("P_rel", "R_rel", "input_line", "output_line"):
                np.testing.assert_allclose(old[key], new[key], atol=1e-12)
            np.testing.assert_allclose(new["R_input"], np.eye(3), atol=1e-12)

    def test_world_reference_uses_assembly_geometry(self):
        for row in fk.sample_backbone_motion(self.strip(), [30, 90], reference="world"):
            np.testing.assert_allclose(row["unit_centers"], row["unit_centers_world"])
            np.testing.assert_allclose(row["R_rel"], row["R_end_world"])

    def test_fixed_reference_invariant_under_whole_strip_rigid_transform(self):
        strip = self.strip()
        before = fk.sample_backbone_motion(strip, [30, 70], track=False)
        compute = strip.compute_strip_kinematics
        angle = 0.71
        Q = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                      [-np.sin(angle), 0, np.cos(angle)]])

        def transformed(*args, **kwargs):
            count = compute(*args, **kwargs)
            strip.global_vertices[:] = [v @ Q.T + [2, -3, 5]
                                        for v in strip.global_vertices]
            return count

        with patch.object(strip, "compute_strip_kinematics", side_effect=transformed):
            after = fk.sample_backbone_motion(strip, [30, 70], track=False)
        for a, b in zip(before, after):
            for key in ("unit_centers", "backbone", "R_rel", "R_input"):
                np.testing.assert_allclose(a[key], b[key], atol=1e-11)

    def test_real_gap_is_retained_in_full_and_compact_plots(self):
        strip = self.strip([40, 80, 50, 110], n=1)
        rows = fk.sample_backbone_motion(strip, [-100, 0, 100], track=False)
        self.assertEqual([r["valid"] for r in rows], [True, False, True])
        self.assertTrue(np.isnan(rows[1]["unit_centers"]).all())
        self.assertEqual(rows[0]["reference_rho0_deg"], -100)
        for data in (rows, [r for r in rows if r["valid"]]):
            ax = fk.plot_end_trajectory(data, show=False)
            self.assertTrue(np.isnan(ax.lines[0].get_data_3d()[0]).any())
            ax.figure.canvas.draw()
        ax = fk.plot_backbone_motion(rows, show=False)
        self.assertEqual(len(ax.origami_failures), 1)
        self.assertTrue(np.isnan(ax.lines[-1].get_data_3d()[0]).any())

    def test_initial_reference_starts_at_first_valid_not_failed_sample(self):
        rows = fk.sample_backbone_motion(self.strip([40, 80, 50, 110], n=1),
                                         [0, 100], track=False)
        self.assertFalse(rows[0]["valid"])
        self.assertTrue(rows[1]["valid"])
        self.assertEqual(rows[1]["reference_rho0_deg"], 100)
        np.testing.assert_allclose(rows[1]["R_input"], np.eye(3), atol=1e-12)

    def test_all_failed_samples_and_empty_plot_do_not_invent_poses(self):
        rows = fk.sample_backbone_motion(self.strip([40, 80, 50, 110], n=1), [0])
        self.assertFalse(rows[0]["valid"])
        for plot in (fk.plot_backbone_motion, fk.plot_end_trajectory):
            for data in (rows, []):
                ax = plot(data, show=False)
                self.assertEqual(len(ax.origami_frames), 0)
                ax.figure.canvas.draw()

    def test_plot_arrows_at_samples_and_bounds_include_arrow_tips(self):
        rows = fk.sample_backbone_motion(self.strip(), [30, 60, 90])
        ax = fk.plot_end_trajectory(rows, frame_indices=[0, 2], frame_length=5, show=False)
        self.assertEqual(len(ax.origami_frames), 3)  # base + two end frames
        for frame, row in zip(ax.origami_frames[1:], (rows[0], rows[2])):
            np.testing.assert_allclose(frame["origin"], row["P_rel"])
            np.testing.assert_allclose(frame["rotation"], row["R_rel"])
            np.testing.assert_allclose(np.linalg.norm(frame["tips"] - frame["origin"], axis=1), 5)
        ax.figure.canvas.draw()
        limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
        for frame in ax.origami_frames:
            self.assertTrue(np.all(frame["tips"] >= limits[:, 0]))
            self.assertTrue(np.all(frame["tips"] <= limits[:, 1]))
            self.assertEqual(len(frame["arrows"]), 3)
            self.assertEqual(len(frame["texts"]), 3)

    def test_four_cell_twenty_unit_round_trip_and_rendered_polylines(self):
        cfg = [dict(alphas=[120, 60, 60, 120], iout=1, sigma=-1),
               dict(alphas=[60, 60, 120, 120], iout=2, sigma=1),
               dict(alphas=[120, 60, 60, 120], iout=3, sigma=-1),
               dict(alphas=[120, 120, 60, 60], iout=2, sigma=1)]
        strip = OrigamiTesellator1D(cfg, 5, contact=True, verbose=False)
        rows = fk.sample_backbone_motion(strip, [15, 45, 90, 135, 170, 90, 15])
        self.assertTrue(all(r["valid"] for r in rows))
        np.testing.assert_allclose(rows[0]["unit_centers"], rows[-1]["unit_centers"], atol=1e-10)
        ax = fk.plot_backbone_motion(rows, frame_indices=[0, 4], show=False)
        self.assertEqual(len(ax.origami_frames), 4)
        for line, row in zip(ax.lines[:len(rows)], rows):
            np.testing.assert_allclose(np.column_stack(line.get_data_3d()), row["backbone"])
        ax.figure.canvas.draw()

    def test_bad_requests_and_rotations_are_rejected(self):
        for angles in ([], [[30]], [np.nan]):
            with self.assertRaises(ValueError):
                fk.sample_backbone_motion(self.strip(), angles)
        with self.assertRaises(ValueError):
            fk.sample_backbone_motion(self.strip(), [30], reference="unknown")
        rows = fk.sample_backbone_motion(self.strip(), [30])
        for kwargs in (dict(frame_count=-1), dict(frame_indices=[2]), dict(frame_length=0)):
            with self.assertRaises(ValueError):
                fk.plot_end_trajectory(rows, show=False, **kwargs)
        ax = plt.figure().add_subplot(projection="3d")
        with self.assertRaises(ValueError):
            fk.draw_coordinate_frame(ax, [0, 0, 0], np.diag([1, 1, -1]))


if __name__ == "__main__":
    unittest.main()
