"""Panel interiors, isolated collinear contact states and actual UI callbacks."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/origami-test-mpl")
import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from origami_simulator import OrigamiDegree4Simulator, panel_penetrations, PathError


def independent_intersections(faces):
    """SVD cone feasibility, independent of production's signed cofactors."""
    hits = []
    for i, j in ((0, 2), (1, 3)):
        a, b = np.asarray(faces[i])[1:]
        c, d = np.asarray(faces[j])[1:]
        _, singular, vh = np.linalg.svd(np.column_stack((a, b, -c, -d)))
        k = vh[-1]
        if singular[-1] > 1e-9 and (np.min(k) > 1e-9 or np.max(k) < -1e-9):
            hits.append((i + 1, j + 1))
    return hits


class PanelContact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = OrigamiDegree4Simulator(
            [70, 110, 70, 110], driving_crease=1, contact=True, verbose=False
        ).run_simulation(81)

    def viewer(self, **kwargs):
        v = self.sim.show_3d_interactive(show=False, random_seed=42, **kwargs)
        self.addCleanup(v.close)
        return v

    def test_report_reproduces_penetration_inside_old_angle_bounds(self):
        session = self.sim.create_session("collinear_other_c0", contact=False)
        state = session.move_to(np.deg2rad(56.706))
        report = self.sim.contact_report(state.rhos)
        self.assertTrue(report["angle_bounds"])
        self.assertFalse(report["admissible"])
        self.assertEqual([h["panels"] for h in report["penetrations"]], [(1, 3)])
        faces, _ = self.sim.get_3d_geometry(state.rhos)
        self.assertEqual(independent_intersections(faces), [(1, 3)])
        # Exhibit the SAME point strictly inside each finite triangle.
        point = report["penetrations"][0]["point"]
        for i in (0, 2):
            triangle = np.asarray(faces[i])
            ab, *_ = np.linalg.lstsq(triangle[1:].T, point, rcond=None)
            self.assertTrue(np.all(ab > 1e-8))
            self.assertLess(ab.sum(), 1 - 1e-8)
            np.testing.assert_allclose(ab @ triangle[1:], point, atol=1e-12)

    def test_collinear_mode_relation_and_both_signed_interiors(self):
        session = self.sim.create_session("collinear_other_c0", contact=False)
        for d in np.r_[
            np.linspace(1e-4, np.pi - 1e-4, 31), -np.linspace(1e-4, np.pi - 1e-4, 31)
        ]:
            q = session.move_to(float(d)).rhos
            # Homogeneous relation remains finite at the tan-half-angle poles.
            self.assertAlmostEqual(
                np.cos(np.deg2rad(70)) * np.sin(q[0] / 2) * np.sin(q[1] / 2),
                np.cos(q[0] / 2) * np.cos(q[1] / 2),
                places=10,
            )
            self.assertAlmostEqual(q[2], -q[0], places=9)
            self.assertAlmostEqual(q[3], -q[1], places=9)
            faces, vertices = self.sim.get_3d_geometry(q)
            self.assertEqual(independent_intersections(faces), [(1, 3)])
            self.assertEqual(
                [h["panels"] for h in panel_penetrations(vertices)], [(1, 3)]
            )

    def test_contact_initialization_selects_exact_zero_not_the_invalid_math_seed(self):
        b = self.sim.branches["collinear_other_c0"]
        self.assertGreater(b.seed_state[0], 0)
        self.assertFalse(self.sim.contact_report(b.seed_state)["admissible"])
        session = self.sim.create_session(b.id, contact=True)
        np.testing.assert_allclose(
            session.state.rhos, [0, np.pi, 0, -np.pi], atol=1e-12
        )
        self.assertTrue(session.drive_locked)
        self.assertLess(session.state.residual, 1e-12)
        self.assertTrue(self.sim.contact_report(session.state.rhos)["admissible"])
        self.assertTrue(all(np.all(abs(p[:, 0]) < 1e-12) for p in session.plot_data()))
        old = session.state
        for target in (1e-6, -1e-6, 0.5, np.pi):
            with self.assertRaisesRegex(PathError, "CONTACT_LIMIT"):
                session.move_to(target)
            self.assertIs(session.state, old)

    def test_contact_graph_keeps_the_disconnected_pi_state_separate(self):
        branch = self.sim.branches["collinear_other_c0"]
        self.assertEqual(len(branch.physical_segments), 2)
        self.assertTrue(all(len(p) == 1 for p in branch.physical_segments))
        self.assertAlmostEqual(branch.physical_segments[0][0, 0], 0)
        self.assertAlmostEqual(branch.physical_segments[1][0, 0], np.pi)
        self.assertTrue(
            any(
                e.reason == "CONTACT_LIMIT" and abs(e.angle) < 1e-9
                for e in branch.endpoints
            )
        )

    def test_actual_radio_and_slider_lock_then_no_contact_motion(self):
        v = self.viewer()
        v.radio.set_active(v._branch_ids.index("collinear_other_c0"))
        self.assertEqual(v.session.branch_id, "collinear_other_c0")
        self.assertAlmostEqual(v.slider.val, 0)
        self.assertFalse(v.slider.active)
        self.assertIn("fixed", v.status.get_text())
        self.assertTrue(all(len(line.get_xdata()) == 1 for line in v._graph_lines))
        q = v.session.state.rhos.copy()
        v.slider.set_val(0.5)  # Programmatic calls must also respect the lock.
        np.testing.assert_array_equal(v.session.state.rhos, q)
        self.assertIn("CONTACT_LIMIT", v.last_error)
        v.check.set_active(0)
        self.assertFalse(v.session.contact)
        self.assertTrue(v.slider.active)
        v.slider.set_val(0.5)
        self.assertIsNone(v.last_error)
        self.assertTrue(self.sim.contact_report(v.session.state.rhos)["penetrations"])
        old = v.session
        v.check.set_active(0)
        self.assertIs(v.session, old)
        self.assertFalse(v.session.contact)
        self.assertFalse(v.check.get_status()[0])
        self.assertIn("penetration", v.last_error)

    def test_reanchor_cannot_hide_penetration_and_is_atomic(self):
        session = self.sim.create_session("collinear_other_c0", contact=False)
        session.move_to(2 * np.pi + 0.5)
        old = session.state
        for reanchor in (False, True):
            self.assertFalse(session.set_contact(True, reanchor=reanchor))
            self.assertIs(session.state, old)
            self.assertFalse(session.contact)
            self.assertEqual(session.contact_reference_history, [])
            self.assertIn("penetration", session.last_error)

    def test_null_mode_and_fixed_drive_family_remain_mobile(self):
        for bid in ("collinear_null_c0", "family_drive_e1_0_free_e2"):
            session = self.sim.create_session(bid, contact=True)
            for angle in (-2.0, -0.1, 0.0, 0.1, 2.0):
                state = (
                    session.move_family(angle)
                    if session.is_family
                    else session.move_to(angle)
                )
                self.assertTrue(self.sim.contact_report(state.rhos)["admissible"])

    def test_coplanar_overlap_and_touching_are_allowed(self):
        for q in ([0, 0, 0, 0], [0, np.pi, 0, -np.pi], [np.pi, 0, -np.pi, 0]):
            self.assertEqual(self.sim.contact_report(q)["penetrations"], [])

    def test_point_normal_reversal_and_rigid_motion_do_not_change_intersections(self):
        session = self.sim.create_session("collinear_other_c0", contact=False)
        q = session.move_to(0.5).rhos
        for folds in (q, -q):
            faces, vertices = self.sim.get_3d_geometry(folds)
            self.assertEqual(independent_intersections(faces), [(1, 3)])
            # Proper rotation plus translation of the same finite triangles.
            rotation, _ = np.linalg.qr(np.array([[1.0, 3, 2], [5, 2, 1], [1, 4, 7]]))
            moved = vertices @ rotation + [1, 2, -3]
            self.assertEqual([h["panels"] for h in panel_penetrations(moved)], [(1, 3)])

    def test_even_and_odd_sample_counts_preserve_isolated_start(self):
        for n in (40, 81, 182):
            sim = OrigamiDegree4Simulator(
                [70, 110, 70, 110], driving_crease=1, verbose=False
            ).run_simulation(n)
            session = sim.create_session("collinear_other_c0", contact=True)
            self.assertAlmostEqual(session.state.drive_unwrapped, 0, places=12)
            self.assertTrue(session.drive_locked)

    def test_predicate_matches_independent_svd_across_cases_and_drivers(self):
        for alphas in (
            [70, 20, 110, 160],
            [60, 90, 135, 75],
            [70, 70, 120, 110],
            [60, 60, 120, 120],
            [70, 110, 70, 110],
            [90] * 4,
            [60, 90, 120, 150],
            [40, 80, 50, 110],
            [70, 80, 70, 100],
        ):
            for drive in (1, 2, 3, 4):
                sim = OrigamiDegree4Simulator(
                    alphas, driving_crease=drive, verbose=False
                )
                for d in (-np.pi, -2.0, -0.1, 0.0, 0.1, 2.0, np.pi):
                    for candidate in sim.solve_states(d):
                        faces, vertices = sim.get_3d_geometry(candidate.rhos)
                        self.assertEqual(
                            [h["panels"] for h in panel_penetrations(vertices)],
                            independent_intersections(faces),
                            (alphas, drive, d),
                        )

    def test_contact_cannot_start_advancing_from_penetrated_state(self):
        seed = self.sim.branches["collinear_other_c0"].seed_state
        with self.assertRaisesRegex(PathError, "CONTACT_LIMIT.*penetration"):
            self.sim._advance(seed, float(seed[0]), float(seed[0]), contact=True)


if __name__ == "__main__":
    unittest.main()
