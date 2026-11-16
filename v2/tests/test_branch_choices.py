"""Contact options, seam-connected paths and validated random viewer starts."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/origami-test-mpl")
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from origami_simulator import OrigamiDegree4Simulator


class BranchChoices(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = OrigamiDegree4Simulator(
            [70, 80, 70, 100], driving_crease=4, contact=False, verbose=False
        ).run_simulation(81)

    def viewer(self, sim=None, **kwargs):
        kwargs.setdefault("random_seed", 42)
        viewer = (sim or self.sim).show_3d_interactive(show=False, **kwargs)
        self.addCleanup(viewer.close)
        return viewer

    def assert_valid(self, viewer):
        state = viewer.session.state
        self.assertTrue(np.isfinite(state.rhos).all())
        self.assertLess(state.residual, 1e-8)
        self.assertAlmostEqual(state.rhos[3], state.drive_unwrapped)
        # Check against the actual sector geometry, independently of UI groups.
        vertices = viewer.sim.get_3d_geometry(state.rhos)[1][1:]
        cosines = np.sum(vertices * np.roll(vertices, -1, axis=0), axis=1)
        np.testing.assert_allclose(
            cosines, np.cos(viewer.sim.original_alphas), atol=1e-8
        )

    def test_four_contact_options_and_two_verified_periodic_paths(self):
        v = self.viewer(contact=True)
        self.assertEqual(len(v.branch_options), 4)
        self.assertTrue(all(len(ids) == 1 for ids in v.branch_options.values()))
        self.assertEqual(len(v.random_initializations), 0)
        before = v.session.state.rhos.copy()
        v.check.set_active(0)  # Real checkbox callback.
        self.assertEqual(set(v.branch_options), {"NE_mode1", "NE_mode2"})
        self.assertTrue(all(len(ids) == 2 for ids in v.branch_options.values()))
        np.testing.assert_array_equal(v.session.state.rhos, before)
        self.assertEqual(len(v.random_initializations), 0)
        for members in v.branch_options.values():
            first, second = (v.sim.branches[bid] for bid in members)
            self.assertTrue(
                second.id in first.seam_connections.values()
                or first.id in second.seam_connections.values()
            )
        v.check.set_active(0)
        self.assertEqual(len(v.branch_options), 4)
        np.testing.assert_allclose(v.session.state.rhos, before, atol=1e-14)
        self.assertEqual(len(v.random_initializations), 0)

    def test_random_start_is_continuous_valid_and_reproducible(self):
        first = self.viewer(random_seed=7)
        second = self.viewer(random_seed=7)
        third = self.viewer(random_seed=8)
        np.testing.assert_array_equal(
            first.session.state.rhos, second.session.state.rhos
        )
        self.assertNotAlmostEqual(first.slider.val, third.slider.val)
        for v in [first, second, third]:
            self.assert_valid(v)
            self.assertIn("RANDOM_START", v.session.state.events)
            self.assertAlmostEqual(v.slider.val, v.session.state.drive_wrapped)
            # The elliptic example has a genuine inaccessible interval around 0.
            self.assertGreater(abs(v.slider.val), np.deg2rad(29.349698))

    def test_random_starts_can_use_both_valid_intervals_and_are_not_grid_samples(self):
        v = self.viewer()
        signs = set()
        members = v.branch_options["NE_mode1"]
        for seed in range(12):
            v._rng = np.random.default_rng(seed)
            session = v._random_session(members)
            t = session.state.drive_unwrapped
            signs.add(int(np.sign(t)))
            self.assertGreater(abs(t), np.deg2rad(29.349698))
            grid = self.sim.branches[session.branch_id].states[:, 3]
            self.assertGreater(np.min(abs(grid - t)), 1e-10)
            self.assertLess(session.state.residual, 1e-8)
        self.assertEqual(signs, {-1, 1})

    def test_actual_radio_selects_path_and_randomizes_once(self):
        v = self.viewer(branch="NE_mode1")
        initial = len(v.random_initializations)
        v.radio.set_active(v._branch_ids.index("NE_mode2"))
        self.assertIsNone(v.last_error)
        self.assertIn(v.session.branch_id, v.branch_options["NE_mode2"])
        self.assertEqual(v.radio.value_selected, "NE_mode2")
        self.assertIn("NE_mode2", v.ax_2d.get_title())
        self.assertEqual(len(v.random_initializations), initial + 1)
        self.assert_valid(v)
        before = v.session.state.rhos.copy()
        v.set_branch(v.session.branch_id)  # Legacy raw segment alias, same path.
        np.testing.assert_array_equal(v.session.state.rhos, before)
        self.assertEqual(len(v.random_initializations), initial + 1)

    def test_contact_branch_selection_uses_its_own_canonical_start(self):
        v = self.viewer(contact=True)
        for index, bid in enumerate(v._branch_ids):
            v.radio.set_active(index)
            self.assertEqual(v.session.branch_id, bid)
            np.testing.assert_array_equal(
                v.session.state.rhos, self.sim.branches[bid].seed_state
            )
        self.assertEqual(len(v.random_initializations), 0)

    def test_wrap_keeps_selected_path_and_toggle_keeps_shape(self):
        v = self.viewer(branch="NE_mode1", random_seed=42)
        # Start on the positive interval, then cross its allowed pi seam.
        self.assertGreater(v.slider.val, 0)
        v.session.move_to(np.pi - 0.02)
        v.slider.set_val(np.pi + 0.01)
        self.assertIsNone(v.last_error)
        self.assertEqual(v.session.state.winding_count, 1)
        self.assertEqual(v.radio.value_selected, "NE_mode1")
        before = v.sim.get_3d_geometry(v.session.state.rhos)[1]
        starts = len(v.random_initializations)
        v.check.set_active(0)
        self.assertTrue(v.session.contact, v.last_error)
        self.assertEqual(len(v.branch_options), 4)
        np.testing.assert_allclose(
            v.sim.get_3d_geometry(v.session.state.rhos)[1], before, atol=1e-12
        )
        v.check.set_active(0)
        self.assertEqual(len(v.branch_options), 2)
        self.assertEqual(v.radio.value_selected, "NE_mode1")
        self.assertEqual(len(v.random_initializations), starts)

    def test_drive_change_keeps_state_and_does_not_randomize(self):
        v = self.viewer()
        before = v.session.state.rhos.copy()
        count = len(v.random_initializations)
        self.assertTrue(v.set_driving_crease(2), v.last_error)
        np.testing.assert_array_equal(v.session.state.rhos, before)
        self.assertEqual(len(v.random_initializations), count)
        self.assertIn(v.session.branch_id, v.branch_options[v.radio.value_selected])
        self.assertEqual(len(v.branch_options), 2)

    def test_group_counts_across_resolution_and_hyperbolic_case(self):
        for alphas, drive, n in [
            ([70, 80, 70, 100], 4, 41),
            ([70, 80, 70, 100], 4, 181),
            ([40, 80, 50, 110], 1, 81),
        ]:
            with self.subTest(alphas=alphas, resolution=n):
                sim = OrigamiDegree4Simulator(
                    alphas, driving_crease=drive, contact=True, verbose=False
                ).run_simulation(n)
                v = self.viewer(sim)
                self.assertEqual(len(v.branch_options), 4)
                self.assertTrue(v.set_contact(False), v.last_error)
                self.assertEqual(set(v.branch_options), {"NE_mode1", "NE_mode2"})
                v.set_branch("NE_mode2")
                self.assertLess(v.session.state.residual, 1e-8)

    def test_no_contact_still_rejects_inaccessible_drive_gap(self):
        v = self.viewer()
        before = v.session.state
        v.slider.set_val(0.0)
        self.assertIs(v.session.state, before)
        self.assertIsNotNone(v.last_error)
        self.assertEqual(v.radio.value_selected, "NE_mode1")

    def test_special_families_are_preserved(self):
        sim = OrigamiDegree4Simulator(
            [90] * 4, driving_crease=4, contact=False, verbose=False
        ).run_simulation(81)
        family = next(b for b in sim.branches.values() if b.family is not None)
        v = self.viewer(sim, branch=family.id)
        self.assertGreater(len(v.branch_options), 2)
        for b in sim.branches.values():
            if b.family is not None:
                self.assertEqual(v.branch_options[b.id], (b.id,))
        self.assertTrue(v.session.is_family)
        self.assertFalse(v.slider.active)
        self.assertAlmostEqual(v.free_slider.val, v.session.free_angle)
        v.free_slider.set_val(0.4)
        self.assertIsNone(v.last_error)
        self.assert_valid(v)

    def test_no_random_start_when_validation_fails_and_selection_is_atomic(self):
        v = self.viewer()
        old = v.session
        with patch.object(
            v.sim, "_advance", side_effect=ValueError("injected failure")
        ):
            v.set_branch("NE_mode2")
        self.assertIs(v.session, old)
        self.assertIn("injected failure", v.last_error)
        self.assertEqual(v.radio.value_selected, "NE_mode1")
        v.set_branch("unknown")
        self.assertIs(v.session, old)
        self.assertIn("Unknown branch", v.last_error)

    def test_viewer_leaves_deterministic_model_data_intact(self):
        before = {
            bid: (b.states.copy(), b.seed_state.copy())
            for bid, b in self.sim.branches.items()
        }
        session = self.sim.create_session()
        self.viewer()
        for bid, (states, seed) in before.items():
            np.testing.assert_array_equal(self.sim.branches[bid].states, states)
            np.testing.assert_array_equal(self.sim.branches[bid].seed_state, seed)
        np.testing.assert_array_equal(
            session.state.rhos, self.sim.create_session().state.rhos
        )


if __name__ == "__main__":
    unittest.main()
