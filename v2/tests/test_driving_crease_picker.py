"""Screen-space crease clicks and shape-preserving viewer reparameterization."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/origami-test-mpl")
import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from matplotlib.backend_bases import MouseEvent
from mpl_toolkits.mplot3d import proj3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from origami_simulator import OrigamiDegree4Simulator


class DrivingCreasePicker(unittest.TestCase):
    def viewer(self, alphas=(70, 20, 110, 160), branch=None, contact=False):
        source = OrigamiDegree4Simulator(
            alphas, driving_crease=4, contact=contact, verbose=False
        ).run_simulation(81)
        viewer = source.show_3d_interactive(branch=branch, show=False)
        self.addCleanup(viewer.close)
        viewer.slider.set_val(0.7)
        return source, viewer

    def click(self, v, xy, button=1, offset=(0, 0)):
        canvas = v.fig.canvas
        for name, point in [
            ("button_press_event", xy),
            ("button_release_event", np.asarray(xy) + offset),
        ]:
            event = MouseEvent(
                name, canvas, float(point[0]), float(point[1]), button=button
            )
            canvas.callbacks.process(name, event)

    def project(self, v, p):
        x, y, _ = proj3d.proj_transform(*p, v.ax_3d.get_proj())
        return v.ax_3d.transData.transform((x, y))

    def test_highlight_and_all_driver_changes_preserve_shape_and_source(self):
        original, v = self.viewer()
        for drive in [1, 2, 3, 4]:
            before = v.session.state.rhos.copy()
            self.assertTrue(v.set_driving_crease(drive), v.last_error)
            self.assertEqual(original.driving_crease, 4)
            self.assertEqual(v.sim.driving_crease, drive)
            self.assertEqual(v.sim.resolution, 81)
            np.testing.assert_array_equal(v.session.state.rhos, before)
            self.assertLess(v.session.state.residual, 1e-8)
            active = v.crease_lines[drive - 1]
            self.assertGreater(
                active.get_linewidth(),
                max(
                    x.get_linewidth()
                    for i, x in enumerate(v.crease_lines)
                    if i != drive - 1
                ),
            )
            self.assertIn(f"e{drive}", v.ax_2d.get_xlabel())
            self.assertEqual(v.slider.label.get_text(), f"Fold e{drive}")
            self.assertEqual(v._other, [i for i in range(4) if i != drive - 1])
            self.assertEqual(v.radio.value_selected, v.session.branch_id)
            self.assertAlmostEqual(v.slider.val, v.session.state.drive_wrapped)
            v.slider.set_val(v.slider.val + 0.005)
            self.assertIsNone(v.last_error)

    def test_actual_mouse_click_on_projected_crease_updates_graph(self):
        _, v = self.viewer()
        v.fig.canvas.draw()
        vertices = v.sim.get_3d_geometry(v.session.state.rhos)[1]
        choices = [self.project(v, f * vertices[1]) for f in [0.45, 0.6, 0.75, 0.9]]
        point = next(p for p in choices if v._crease_at(*p) == 1)
        before = v.session.state.rhos.copy()
        self.click(v, point)
        self.assertEqual(v.sim.driving_crease, 1, v.last_error)
        np.testing.assert_array_equal(before, v.session.state.rhos)
        self.assertIn("e1", v.ax_2d.get_xlabel())
        graph = np.vstack(v.session.plot_data())
        np.testing.assert_allclose(
            v._graph_lines[0].get_xdata(), v.session.plot_data()[0][:, 0]
        )
        self.assertGreater(len(graph), 1)

    def test_label_click_uses_projected_text_position(self):
        _, v = self.viewer()
        v.fig.canvas.draw()
        label = v.crease_labels[1]
        point = self.project(v, label.get_position_3d())
        renderer = v.fig.canvas.get_renderer()
        w, h, d = renderer.get_text_width_height_descent(
            label.get_text(), label.get_fontproperties(), ismath=False
        )
        point = point + np.array([w / 2, h / 2 - d])
        self.assertEqual(v._crease_at(*point), 2)
        self.click(v, point)
        self.assertEqual(v.sim.driving_crease, 2, v.last_error)

    def test_camera_drag_right_click_and_origin_do_not_change_driver(self):
        _, v = self.viewer()
        v.fig.canvas.draw()
        q = v.session.state.rhos.copy()
        p = self.project(v, 0.7 * v.sim.get_3d_geometry(q)[1][1])
        self.click(v, p, offset=(30, 10))
        self.click(v, p, button=3)
        self.click(v, self.project(v, np.zeros(3)))
        self.assertEqual(v.sim.driving_crease, 4)
        np.testing.assert_array_equal(q, v.session.state.rhos)

    def test_cached_models_reused_and_failed_selection_is_atomic(self):
        original, v = self.viewer()
        self.assertTrue(v.set_driving_crease(1), v.last_error)
        first = v.sim
        self.assertTrue(v.set_driving_crease(4), v.last_error)
        self.assertIs(v.sim, original)
        self.assertTrue(v.set_driving_crease(1), v.last_error)
        self.assertIs(v.sim, first)
        old = v.session
        with patch.object(
            v, "_session_at_shape", side_effect=ValueError("ambiguous current path")
        ):
            self.assertFalse(v.set_driving_crease(2))
        self.assertIs(v.session, old)
        self.assertIs(v.sim, first)
        self.assertIn("ambiguous", v.last_error)
        self.assertEqual(v.radio.value_selected, old.branch_id)

    def test_switch_after_wrap_preserves_lift_then_contact_reanchor_works(self):
        _, v = self.viewer()
        v.session.move_to(np.pi + 0.2)
        before = v.session.state.rhos.copy()
        self.assertTrue(v.set_driving_crease(1), v.last_error)
        np.testing.assert_array_equal(before, v.session.state.rhos)
        self.assertAlmostEqual(v.session.state.drive_unwrapped, before[0])
        self.assertEqual(v.session.state.winding_count, 1)
        self.assertTrue(v.set_contact(True), v.last_error)
        self.assertTrue(v.session.contact)
        self.assertEqual(v.session.state.winding_count, 0)
        v.slider.set_val(v.slider.val + 0.005)
        self.assertIsNone(v.last_error)

    def test_stationary_crease_selects_family_and_free_crease_restores_actuation(self):
        _, v = self.viewer((60, 60, 120, 120), branch="ff_minus_c0")
        before = v.session.state.rhos.copy()
        self.assertTrue(v.set_driving_crease(1), v.last_error)
        self.assertTrue(v.session.is_family)
        self.assertFalse(v.slider.active)
        self.assertTrue(v.free_slider.ax.get_visible())
        self.assertEqual(v.free_slider.label.get_text(), "Free e2")
        np.testing.assert_array_equal(before, v.session.state.rhos)
        v.free_slider.set_val(0.8)
        self.assertAlmostEqual(v.session.state.rhos[1], 0.8)
        before = v.session.state.rhos.copy()
        self.assertTrue(v.set_driving_crease(2), v.last_error)
        self.assertFalse(v.session.is_family)
        self.assertTrue(v.slider.active)
        self.assertFalse(v.free_slider.ax.get_visible())
        np.testing.assert_array_equal(before, v.session.state.rhos)

    def test_generic_developable_and_hyperbolic_axis_changes(self):
        for alphas in [(60, 90, 135, 75), (61, 91, 136, 76)]:
            _, v = self.viewer(alphas, contact=True)
            for drive in [1, 2, 3, 4]:
                with self.subTest(alphas=alphas, drive=drive):
                    before = v.session.state.rhos.copy()
                    self.assertTrue(v.set_driving_crease(drive), v.last_error)
                    self.assertTrue(v.session.contact)
                    np.testing.assert_array_equal(before, v.session.state.rhos)
                    self.assertLess(v.session.state.residual, 1e-8)
            v.close()

    def test_flat_intersection_keeps_incoming_mode_direction(self):
        _, v = self.viewer()
        v.slider.set_val(0.0)
        self.assertTrue(v.set_driving_crease(1), v.last_error)
        v.slider.set_val(0.03)
        self.assertIsNone(v.last_error)
        q = v.session.state.rhos
        self.assertAlmostEqual(q[0], q[2], places=8)
        self.assertAlmostEqual(q[1], -q[3], places=8)


if __name__ == "__main__":
    unittest.main()
