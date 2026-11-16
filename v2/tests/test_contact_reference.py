"""Bounds can start a new reference at a periodic configuration, explicitly."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/origami-test-mpl")
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from origami_simulator import OrigamiDegree4Simulator, PathError


class ContactReference(unittest.TestCase):
    def sim(self, alphas=(70, 20, 110, 160), drive=4):
        return OrigamiDegree4Simulator(
            alphas, driving_crease=drive, contact=False, verbose=False
        ).run_simulation(81)

    def same_shape(self, sim, before, after):
        self.assertEqual(before.branch_id, after.branch_id)
        np.testing.assert_allclose(
            sim.get_3d_geometry(before.rhos)[1],
            sim.get_3d_geometry(after.rhos)[1],
            atol=1e-12,
        )
        self.assertLess(after.residual, 1e-8)

    def test_both_ff_modes_and_wrap_directions_reanchor_without_shape_jump(self):
        sim = self.sim()
        for branch in ("ff_plus_c0", "ff_minus_c0"):
            for direction in (-1, 1):
                with self.subTest(branch=branch, direction=direction):
                    session = sim.create_session(branch, contact=False)
                    session.move_to(direction * (2 * np.pi + 0.23))
                    before = session.state
                    self.assertFalse(session.set_contact(True))  # legacy strict policy
                    self.assertTrue(session.set_contact(True, reanchor=True))
                    self.same_shape(sim, before, session.state)
                    self.assertEqual(session.state.winding_count, 0)
                    self.assertLessEqual(np.max(abs(session.state.rhos)), np.pi + 1e-10)
                    self.assertIn("CONTACT_REFERENCE_RESET", session.state.events)
                    self.assertEqual(len(session.contact_reference_history), 1)
                    np.testing.assert_array_equal(
                        session.contact_reference_history[0].rhos, before.rhos
                    )
                    session.move_to(session.state.drive_unwrapped + 0.01)

    def test_real_checkbox_reanchors_and_syncs_slider_and_graph(self):
        sim = self.sim()
        viewer = sim.show_3d_interactive(show=False)
        try:
            viewer.session.move_to(np.pi - 0.02)
            viewer.slider.set_val(np.pi + 0.01)
            old = viewer.session.state
            viewer.check.set_active(0)
            self.assertTrue(viewer.check.get_status()[0])
            self.assertTrue(viewer.session.contact)
            self.assertIsNone(viewer.last_error)
            self.same_shape(sim, old, viewer.session.state)
            self.assertEqual(viewer.session.state.winding_count, 0)
            self.assertAlmostEqual(viewer.slider.val, -np.pi + 0.01)
            self.assertAlmostEqual(viewer.slider.valmin, -np.pi)
            self.assertAlmostEqual(viewer.slider.valmax, np.pi)
            self.assertIn("reference reset", viewer.status.get_text())
            graph = np.vstack(viewer.session.plot_data())
            self.assertLess(
                np.min(np.linalg.norm(graph - viewer.session.state.rhos, axis=1)), 1e-8
            )
            viewer.check.set_active(0)
            viewer.slider.set_val(-np.pi - 0.01)
            self.assertEqual(viewer.session.state.winding_count, -1)
            viewer.check.set_active(0)
            self.assertTrue(viewer.session.contact)
            self.assertEqual(viewer.session.state.winding_count, 0)
            self.assertEqual(len(viewer.session.contact_reference_history), 2)
        finally:
            viewer.close()

    def test_reanchored_bounds_still_block_future_domain_crossing(self):
        sim = self.sim()
        session = sim.create_session(contact=False)
        session.move_to(2 * np.pi + 0.2)
        session.set_contact(True, reanchor=True)
        old = session.state
        with self.assertRaises(PathError):
            session.move_to(np.pi + 0.01)
        self.assertIs(session.state, old)
        session.move_to(0.1)
        self.assertAlmostEqual(session.state.drive_unwrapped, 0.1)

    def test_non_grashof_plot_uses_current_principal_component(self):
        sim = self.sim((40, 80, 50, 110), drive=1)
        branch = next(
            b.id
            for b in sim.branches.values()
            if b.family is None and b.seed_state[0] > 0
        )
        session = sim.create_session(branch, contact=False)
        session.move_to(np.pi + 0.2)
        old = session.state
        self.assertTrue(session.set_contact(True, reanchor=True))
        self.same_shape(sim, old, session.state)
        pieces = session.plot_data()
        self.assertTrue(pieces)
        graph = np.vstack(pieces)
        self.assertTrue(np.all(graph[:, 0] < 0))
        self.assertLess(np.min(abs(graph[:, 0] - session.state.drive_wrapped)), 1e-9)
        old = session.state
        with self.assertRaises(PathError):
            session.move_to(0.0)
        self.assertIs(old, session.state)

    def test_family_free_fold_is_also_reanchored(self):
        sim = self.sim((60, 60, 120, 120))
        bid = next(b.id for b in sim.branches.values() if b.family is not None)
        session = sim.create_session(bid, contact=False)
        session.move_family(2 * np.pi + 0.3)
        old = session.state
        self.assertTrue(session.set_contact(True, reanchor=True))
        self.same_shape(sim, old, session.state)
        self.assertAlmostEqual(session.free_angle, 0.3)
        self.assertTrue(session.plot_data())
        with self.assertRaises(PathError):
            session.move_family(np.pi + 0.1)


if __name__ == "__main__":
    unittest.main()
