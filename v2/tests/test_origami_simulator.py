"""Independent kinematic invariants and public simulator regressions.

Run: python -m unittest discover -s v2/tests -v
No GUI display or third-party test runner is required.
"""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/origami-test-mpl")
import sys
import unittest
import contextlib
import io
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = [
    [70, 20, 110, 160],
    [60, 90, 135, 75],
    [70, 70, 120, 110],
    [60, 60, 120, 120],
    [70, 110, 70, 110],
    [90, 90, 90, 90],
    [60, 90, 120, 150],
    [40, 80, 50, 110],
]


def rx(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def closure_error(alphas, rhos):
    """Independent matrix-loop oracle, without production helper calls."""
    m = np.eye(3)
    for a, r in zip(np.deg2rad(alphas), rhos):
        m = m @ rx(r) @ rz(a)
    return np.linalg.norm(m - np.eye(3))


def angle_delta(a, b):
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


def contains(states, target, tolerance=2e-7):
    return any(np.linalg.norm(angle_delta(s, target)) < tolerance for s in states)


class SolverInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from origami_kinematics import Kinematics

        cls.Kinematics = Kinematics

    def test_geometry_symmetry_maps_preserve_sectors_and_incidence(self):
        for a in ([90] * 4, [60, 60, 120, 120], [70, 110, 70, 110], [60, 90, 135, 75]):
            g = self.Kinematics(a).geometry
            for item in g["rotations"] + g["reflections"]:
                sm = np.array(item["sector_map"]) - 1
                cm = np.array(item["crease_map"]) - 1
                np.testing.assert_allclose(np.array(a), np.array(a)[sm], atol=1e-10)
                for i in range(4):
                    self.assertEqual(
                        {int(sm[(i - 1) % 4]), int(sm[i])},
                        {(int(cm[i]) - 1) % 4, int(cm[i])},
                    )
        self.assertEqual(len(self.Kinematics([90] * 4).geometry["rotations"]), 4)
        self.assertEqual(len(self.Kinematics([90] * 4).geometry["reflections"]), 4)

    def test_near_geometry_does_not_silently_change_case(self):
        g = self.Kinematics([70, 20, 110, 160.00001]).geometry
        self.assertEqual(g["case"], "hyperbolic")
        self.assertTrue(g["near"]["euclidean"])
        self.assertFalse(g["euclidean_flat_foldable"])

    def test_invalid_and_explicitly_unsupported_input(self):
        for a in (
            [70, 20, 110],
            [70, 20, np.nan, 160],
            [0, 90, 90, 180],
            [-1, 90, 90, 181],
        ):
            with self.assertRaises(ValueError):
                self.Kinematics(a)
        with self.assertRaises(ValueError):
            self.Kinematics([90] * 4, driving_crease=True)
        k = self.Kinematics([180, 60, 60, 60])
        self.assertEqual(k.solve(0.4), [])
        self.assertTrue(k.last_status.startswith("UNSUPPORTED"))

    def test_all_cases_and_original_drive_indices_close(self):
        for a in CASES:
            total = 0
            for d in range(1, 5):
                kin = self.Kinematics(a, driving_crease=d)
                for deg in [-179.999, -90, -45, 0, 45, 90, 179.999]:
                    with self.subTest(alphas=a, drive=d, angle=deg):
                        for state in kin.solve(np.deg2rad(deg)):
                            q = np.asarray(state.rhos)
                            total += 1
                            self.assertEqual(q.shape, (4,))
                            self.assertTrue(np.isfinite(q).all())
                            self.assertLess(closure_error(a, q), 2e-7)
                            self.assertLess(
                                abs(angle_delta(q[d - 1], np.deg2rad(deg))), 1e-9
                            )
            self.assertGreater(total, 0)

    def test_zero_drive_reconstruction_uses_consistent_signs(self):
        a = [70, 70, 120, 110]
        states = self.Kinematics(a, driving_crease=4).solve(0.0)
        self.assertEqual(len(states), 2)
        for s in states:
            self.assertLess(closure_error(a, s.rhos), 1e-10)
        self.assertTrue(
            contains(
                [s.rhos for s in states],
                np.deg2rad(
                    [-38.690259815447575, 30.63681144691559, -38.69025981544758, 0]
                ),
            )
        )

    def test_projective_root_at_infinity_is_not_lost(self):
        a = [60, 90, 120, 150]
        states = self.Kinematics(a, driving_crease=4).solve(np.pi / 2)
        self.assertTrue(
            contains([s.rhos for s in states], np.deg2rad([180, -120, 90, 90]))
        )

    def test_flat_foldable_near_zero_matches_analytic_slopes(self):
        kin = self.Kinematics([70, 20, 110, 160], driving_crease=4)
        drive = np.deg2rad(1e-6)
        t0, t1 = np.deg2rad([70, 20])
        k = (np.sin(t0) + np.array([-1, 1]) * np.sin(t1)) / np.sin(t0 + t1)
        expected = np.sort(2 * np.arctan(k * np.tan(drive / 2)))
        actual = np.sort([s.rhos[0] for s in kin.solve(drive)])
        self.assertEqual(len(actual), 2)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-5)

    def test_null_fold_modes_remain_present(self):
        for a in ([60, 60, 120, 120], [90, 90, 90, 90]):
            states = self.Kinematics(a, driving_crease=4).solve(np.pi / 4)
            self.assertTrue(
                contains([s.rhos for s in states], np.deg2rad([0, 45, 0, 45]))
            )

    def test_fixed_drive_continuous_family_is_explicit(self):
        kin = self.Kinematics([90, 90, 90, 90], driving_crease=4)
        kin.solve(0.0)
        self.assertTrue(kin.last_families)
        for family in kin.last_families:
            for t in [-1.0, 0.2, 1.0]:
                candidate = family["evaluate"](t)
                self.assertIsNotNone(candidate)
                self.assertLess(closure_error([90] * 4, candidate.rhos), 1e-9)
                self.assertLess(abs(angle_delta(candidate.rhos[3], 0)), 1e-9)

    def test_developable_origin_exists_even_when_modes_meet(self):
        for a in ([70, 20, 110, 160], [60, 90, 135, 75], [70, 110, 70, 110], [90] * 4):
            kin = self.Kinematics(a, driving_crease=4)
            for drive in [0.0, -4.440892098500626e-16]:
                isolated = kin.solve(drive)
                states = [s.rhos for s in isolated]
                for family in kin.last_families:
                    c = family["evaluate"](0.0)
                    if c is not None:
                        states.append(c.rhos)
                self.assertTrue(
                    contains(states, np.zeros(4)), (a, drive, kin.last_status)
                )

    def test_boundary_families_match_independent_exact_rules(self):
        for a, drive, target in [
            ([60, 60, 120, 120], np.pi, lambda t: [t, np.pi, -t, np.pi]),
            ([70, 110, 70, 110], 0.0, lambda t: [t, 0.0, t, 0.0]),
            ([90] * 4, np.pi, lambda t: [t, np.pi, -t, np.pi]),
            ([110, 110, 120, 120], np.pi, lambda t: [t, np.pi, -t, np.pi]),
            ([70, 70, 60, 60], np.pi, lambda t: [t, np.pi, -t, np.pi]),
        ]:
            kin = self.Kinematics(a, driving_crease=4)
            kin.solve(drive)
            self.assertTrue(kin.last_families)
            for t in [-2.0, -0.4, 0.3, 1.7]:
                wanted = np.array(target(t))
                self.assertLess(closure_error(a, wanted), 1e-10)
                states = []
                for family in kin.last_families:
                    c = family["evaluate"](t)
                    if c is not None:
                        states.append(c.rhos)
                self.assertTrue(contains(states, wanted))

    def test_cyclic_reindexing_preserves_solution_sets(self):
        a = np.array([61.0, 91.0, 136.0, 76.0])
        drive = 0.8
        d = 4
        reference = [s.rhos for s in self.Kinematics(a, driving_crease=d).solve(drive)]
        self.assertTrue(reference)
        for shift in range(4):
            moved = self.Kinematics(
                np.roll(a, shift), driving_crease=(d - 1 + shift) % 4 + 1
            )
            states = [s.rhos for s in moved.solve(drive)]
            self.assertEqual(len(states), len(reference))
            for r in reference:
                self.assertTrue(contains(states, np.roll(r, shift)))

    def test_fold_inversion(self):
        for a in CASES:
            kin = self.Kinematics(a, driving_crease=4)
            positive = [s.rhos for s in kin.solve(0.9)]
            negative = [s.rhos for s in kin.solve(-0.9)]
            self.assertEqual(len(positive), len(negative))
            for r in positive:
                self.assertTrue(contains(negative, -r))

    def test_elliptic_hyperbolic_duality(self):
        for a in ([61, 91, 136, 76], [70, 70, 120, 110], [40, 80, 50, 110]):
            kin = self.Kinematics(a, driving_crease=4)
            dual = self.Kinematics(180 - np.array(a), driving_crease=4)
            r = [s.rhos for s in kin.solve(0.9)]
            t = [s.rhos for s in dual.solve(-0.9)]
            self.assertEqual(len(r), len(t))
            for q in r:
                mapped = np.array([1, -1, 1, -1]) * q
                self.assertLess(closure_error(180 - np.array(a), mapped), 1e-8)
                self.assertTrue(contains(t, mapped))


class SimulatorRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from origami_simulator import OrigamiDegree4Simulator

        cls.Simulator = OrigamiDegree4Simulator

    def make(self, a=(70, 20, 110, 160), n=81, contact=False, drive=4):
        sim = self.Simulator(a, contact=contact, verbose=False, driving_crease=drive)
        with contextlib.redirect_stdout(io.StringIO()):
            sim.run_simulation(n)
        return sim

    def test_all_case_runs_keep_valid_math_states(self):
        for a in CASES:
            with self.subTest(alphas=a):
                sim = self.make(a)
                self.assertGreater(len(sim.full_math_data), 0)
                for q in sim.full_math_data:
                    self.assertLess(closure_error(a, q), 2e-7)
                for name in ["branch1", "branch2", "math_branch1", "math_branch2"]:
                    self.assertEqual(np.asarray(getattr(sim, name)).shape[1:], (4,))

    def test_query_returns_actual_solutions_not_masked_nan(self):
        sim = self.Simulator([70, 20, 110, 160], verbose=False)
        out = sim.compute_folding_angles_from_input_output([45], 0, sign=1)
        self.assertEqual(int(out["num_solutions"][0]), 2)
        self.assertGreaterEqual(np.count_nonzero(np.isfinite(out["output_angles"])), 2)
        self.assertEqual(np.asarray(out["folding_angles_all"]).shape[-1], 4)

    def test_branch_ids_independent_of_display_resolution(self):
        for a in (
            [70, 20, 110, 160],
            [60, 90, 135, 75],
            [70, 70, 120, 110],
            [60, 60, 120, 120],
            [90] * 4,
        ):
            with self.subTest(alphas=a):
                low = self.make(a, n=81)
                high = self.make(a, n=121)
                self.assertEqual(set(low.branches), set(high.branches))

    def test_seed_work_is_conditional(self):
        direct = self.make()
        self.assertEqual(direct.metrics["seed_attempts"], 0)
        traced = self.make([70, 70, 120, 110])
        self.assertGreaterEqual(traced.metrics["seed_checks"], 10)

    def test_geometry_classification_unique_and_special(self):
        generic = self.Simulator([60, 90, 135, 75], verbose=False)
        self.assertEqual(set(generic.geometry["unique_folds"]), {1, 2})
        for a in ([60, 60, 120, 120], [90, 90, 90, 90]):
            geom = self.Simulator(a, verbose=False).geometry
            self.assertTrue(geom["flat_foldable"])
            self.assertTrue(geom["collinear_pairs"])

    def test_non_euclidean_flat_foldability_is_separate_from_factor_rule(self):
        a = [110, 110, 120, 120]
        sim = self.Simulator(a, verbose=False)
        self.assertEqual(sim.geometry["case"], "hyperbolic")
        self.assertTrue(sim.geometry["flat_foldable"])
        self.assertFalse(sim.geometry["euclidean_flat_foldable"])

    def test_tessellator_and_forward_kinematics_legacy_calls(self):
        from origami_tesellator_1D import OrigamiTesellator1D
        from origami_forward_kinematics import extract_relative_se3

        for a, rho in [
            ([70, 20, 110, 160], -70),
            ([70, 70, 120, 110], 45),
            ([80, 60, 100, 40], 45),
        ]:
            with self.subTest(alphas=a):
                tess = OrigamiTesellator1D(
                    [{"alphas": a, "sigma": 1, "iout": 2}], num_periods=5, verbose=False
                )
                self.assertEqual(tess.compute_strip_kinematics(np.deg2rad(rho)), 5)
                result = extract_relative_se3(tess, rho)
                self.assertIsNotNone(result)
                self.assertTrue(np.isfinite(result["R_rel"]).all())
                np.testing.assert_allclose(
                    result["R_rel"].T @ result["R_rel"], np.eye(3), atol=1e-7
                )

    def test_periodic_session_keeps_branch_and_overshoot(self):
        sim = self.make(n=121)
        session = sim.create_session(contact=False)
        bid = session.state.branch_id
        session.move_to(np.pi - 0.02)
        before = session.state.rhos.copy()
        state = session.move_slider(np.pi + 0.01)
        self.assertAlmostEqual(state.drive_unwrapped, np.pi + 0.01, places=9)
        self.assertAlmostEqual(state.drive_wrapped, -np.pi + 0.01, places=9)
        self.assertEqual(state.winding_count, 1)
        self.assertEqual(state.branch_id, bid)
        self.assertLess(np.linalg.norm(angle_delta(before, state.rhos)), 0.2)
        self.assertLess(closure_error(sim.original_alphas_deg, state.rhos), 1e-7)
        state = session.move_slider(-np.pi - 0.01)
        self.assertAlmostEqual(state.drive_unwrapped, np.pi - 0.01, places=9)
        self.assertEqual(state.winding_count, 0)
        state = session.move_to(-np.pi + 0.02)
        state = session.move_slider(-np.pi - 0.01)
        self.assertAlmostEqual(state.drive_unwrapped, -np.pi - 0.01, places=9)
        self.assertAlmostEqual(state.drive_wrapped, np.pi - 0.01, places=9)
        self.assertEqual(state.winding_count, -1)

    def test_contact_does_not_wrap_and_rechecks_unwrapped_state(self):
        sim = self.make(n=81)
        session = sim.create_session(contact=False)
        session.move_to(2 * np.pi + 0.2)
        self.assertFalse(session.set_contact(True))
        self.assertFalse(session.contact)
        bounded = sim.create_session(contact=True)
        with self.assertRaises(ValueError):
            bounded.move_to(np.pi + 0.01)

    def test_geometry_reproduces_all_four_signed_dihedrals(self):
        for a in CASES:
            sim = self.Simulator(a, contact=False, verbose=False, driving_crease=4)
            for c in sim.solve_states(1.2):
                faces, v = sim.get_3d_geometry(c.rhos)
                self.assertEqual(len(faces), 4)
                edges = np.asarray(v)[1:5]
                normals = []
                for i in range(4):
                    self.assertAlmostEqual(
                        float(edges[i] @ edges[(i + 1) % 4]),
                        float(np.cos(np.deg2rad(a[i]))),
                        places=8,
                    )
                    n = np.cross(edges[i], edges[(i + 1) % 4]) / np.sin(
                        np.deg2rad(a[i])
                    )
                    normals.append(n / np.linalg.norm(n))
                measured = []
                for i in range(4):
                    n, m = normals[(i - 1) % 4], normals[i]
                    measured.append(np.arctan2(edges[i] @ np.cross(n, m), n @ m))
                np.testing.assert_allclose(angle_delta(measured, c.rhos), 0, atol=2e-7)

    def test_non_grashof_gap_cannot_be_crossed_by_periodic_move(self):
        sim = self.make([40, 80, 50, 110], n=121, drive=1)
        positive = [
            b for b in sim.branches.values() if b.family is None and b.seed_state[0] > 0
        ]
        self.assertTrue(positive)
        session = sim.create_session(positive[0].id, contact=False)
        before = session.state.rhos.copy()
        with self.assertRaises(ValueError):
            session.move_to(0.0)
        np.testing.assert_allclose(session.state.rhos, before)
        with self.assertRaises(ValueError):
            session.move_to(-np.deg2rad(80))
        np.testing.assert_allclose(session.state.rhos, before)
        for piece in sim.plot_data(positive[0].id, False):
            self.assertTrue(np.all(piece[:, 0] > 0))

    def test_family_free_fold_slider_executes_real_model_callback(self):
        import matplotlib.pyplot as plt

        sim = self.make([90] * 4, n=81)
        family = next(b for b in sim.branches.values() if b.family is not None)
        viewer = sim.show_3d_interactive(branch=family.id, show=False)
        try:
            self.assertTrue(viewer.session.is_family)
            self.assertFalse(viewer.slider.active)
            self.assertTrue(viewer.free_slider.active)
            viewer.free_slider.set_val(0.4)
            self.assertTrue(viewer.last_movement_success, viewer.last_error)
            self.assertAlmostEqual(viewer.session.free_angle, 0.4)
            self.assertAlmostEqual(
                float(
                    angle_delta(
                        viewer.session.state.rhos[family.family["free_crease"] - 1], 0.4
                    )
                ),
                0,
                places=8,
            )
            self.assertLess(closure_error([90] * 4, viewer.session.state.rhos), 1e-8)
        finally:
            plt.close(viewer.fig)

    def test_full_space_includes_fixed_drive_family_states(self):
        sim = self.make([90] * 4, n=81)
        families = [b for b in sim.branches.values() if b.family is not None]
        self.assertTrue(families)
        for b in families:
            for q in b.states[::20]:
                self.assertTrue(contains(sim.full_math_data, q))

    def test_repeated_multi_cycle_wrapping_keeps_lift_and_identity(self):
        sim = self.make(n=81)
        session = sim.create_session(contact=False)
        bid = session.state.branch_id
        for target in [
            2 * np.pi + 0.3,
            4 * np.pi + 0.3,
            -2 * np.pi - 0.3,
            -4 * np.pi - 0.3,
            0.3,
        ]:
            state = session.move_to(target)
            self.assertAlmostEqual(state.drive_unwrapped, target, places=9)
            self.assertAlmostEqual(
                state.drive_wrapped,
                float((target + np.pi) % (2 * np.pi) - np.pi),
                places=9,
            )
            self.assertEqual(state.branch_id, bid)
            self.assertLess(closure_error(sim.original_alphas_deg, state.rhos), 1e-7)
            self.assertEqual(
                state.winding_count,
                int(round((target - state.drive_wrapped) / (2 * np.pi))),
            )

    def test_narrow_infeasible_gap_cannot_be_tunneled(self):
        a = (np.array([60, 90, 135, 75]) * 0.99999).tolist()
        sim = self.make(a, n=121, drive=3)
        positive = [
            b for b in sim.branches.values() if b.family is None and b.seed_state[2] > 0
        ]
        self.assertTrue(positive)
        session = sim.create_session(positive[0].id, contact=False)
        session.move_to(0.02)
        before = session.state.rhos.copy()
        with self.assertRaises(ValueError):
            session.move_to(-0.02)
        np.testing.assert_allclose(session.state.rhos, before)

    def test_contact_exact_positive_pi_can_retreat_without_wrap(self):
        sim = self.make(n=81)
        session = sim.create_session(contact=True)
        state = session.move_to(np.pi)
        self.assertAlmostEqual(state.drive_wrapped, np.pi, places=10)
        self.assertEqual(state.winding_count, 0)
        state = session.move_slider(np.pi - 0.01)
        self.assertAlmostEqual(state.drive_unwrapped, np.pi - 0.01, places=10)
        self.assertEqual(state.winding_count, 0)

    def test_family_slider_respects_requested_free_angle_not_nearest_lift(self):
        sim = self.make([90] * 4, n=81)
        branch = next(b for b in sim.branches.values() if b.family is not None)
        session = sim.create_session(branch.id, contact=False)
        session.move_family(-3.0)
        state = session.move_family(3.0)
        idx = branch.family["free_crease"] - 1
        self.assertAlmostEqual(state.rhos[idx], 3.0, places=9)
        self.assertAlmostEqual(session.free_angle, 3.0, places=9)
        self.assertLess(closure_error([90] * 4, state.rhos), 1e-8)

    def test_cached_empty_solution_keeps_its_failure_reason(self):
        sim=self.Simulator([40,80,50,110],contact=False,verbose=False,driving_crease=1)
        self.assertEqual(sim._solve(0.),[])
        reason=sim._core.last_status
        self.assertEqual(reason,'NO_SOLUTION')
        self.assertTrue(sim._solve(np.deg2rad(80)))
        self.assertEqual(sim._core.last_status,'OK')
        self.assertEqual(sim._solve(0.),[])
        self.assertEqual(sim._core.last_status,reason)

    def test_numerical_trace_failure_does_not_claim_unreached_interval(self):
        from origami_simulator import PathError
        base=self.Simulator
        class FailingContinuation(base):
            def _advance(self,q,t,target,*args,**kwargs):
                if abs(target)>.3:
                    raise PathError('NUMERICAL_FAILURE','injected continuation failure')
                return super()._advance(q,t,target,*args,**kwargs)
        sim=FailingContinuation([70,20,110,160],contact=False,verbose=False)
        sim.run_simulation(81)
        for branch in sim.branches.values():
            self.assertTrue(any(e.reason=='NUMERICAL_FAILURE' for e in branch.endpoints))
            for lo,hi in branch.reachable_intervals:
                self.assertGreaterEqual(lo,-.3-1e-9)
                self.assertLessEqual(hi,.3+1e-9)

    def test_seed_candidate_order_does_not_change_branch_identity(self):
        from unittest.mock import patch
        import origami_simulator as module
        a=[70,70,120,110]
        reference=self.make(a,n=81)
        original=np.linspace
        def reverse_seed_grid(start,stop,num=50,*args,**kwargs):
            values=original(start,stop,num,*args,**kwargs)
            return values[::-1] if num==50 else values
        with patch.object(module.np,'linspace',side_effect=reverse_seed_grid):
            reversed_seeds=self.make(a,n=81)
        self.assertEqual(set(reference.branches),set(reversed_seeds.branches))
        for bid,b in reference.branches.items():
            other=reversed_seeds.branches[bid]
            np.testing.assert_allclose(b.seed_state,other.seed_state,atol=1e-10)
            left=sorted((round(e.angle,7),e.reason) for e in b.endpoints)
            right=sorted((round(e.angle,7),e.reason) for e in other.endpoints)
            self.assertEqual(left,right)

    def test_forward_reverse_session_returns_to_same_state(self):
        sim=self.make(n=81)
        session=sim.create_session(contact=False)
        initial=session.state
        t=initial.drive_unwrapped
        session.move_to(t+.5)
        session.move_to(t-.5)
        returned=session.move_to(t)
        self.assertEqual(returned.branch_id,initial.branch_id)
        np.testing.assert_allclose(returned.rhos,initial.rhos,atol=2e-8)
        self.assertAlmostEqual(returned.drive_unwrapped,t,places=10)

    def test_dense_collinear_contact_boundary_keeps_incoming_path(self):
        sim=self.make([70,110,70,110],n=1000)
        self.assertGreater(len(sim.branches),1)
        for b in sim.branches.values():
            for endpoint in b.endpoints:
                if endpoint.reason=='CONTACT_LIMIT':
                    self.assertTrue(np.isfinite(endpoint.angle))
            for q in b.states[::100]:
                self.assertLess(closure_error([70,110,70,110],q),1e-8)

    def test_geometry_snapshot_and_invalid_render_guards(self):
        source=np.array([70.,70.,120.,110.])
        sim=self.Simulator(source,contact=False,verbose=False,driving_crease=4)
        expected=[c.rhos.copy() for c in sim.solve_states(.7)]
        source[:]=90.
        actual=[c.rhos for c in sim.solve_states(.7)]
        self.assertEqual(len(expected),len(actual))
        for q in expected: self.assertTrue(contains(actual,q))
        with self.assertRaises(ValueError): sim.original_alphas_deg[0]=80.
        with self.assertRaises(AttributeError): sim.driving_crease=2
        with self.assertRaises(ValueError): sim.get_3d_geometry(np.zeros(4))
        for tolerance in [-1.,0.,float('nan')]:
            with self.assertRaises(ValueError): self.Simulator([90]*4,residual_tol=tolerance)

    def test_slider_callback_wraps_without_recursion(self):
        import matplotlib.pyplot as plt

        sim = self.make(n=81)
        viewer = sim.show_3d_interactive(show=False)
        try:
            viewer.session.move_to(np.pi - 0.02)
            viewer.slider.set_val(np.pi + 0.01)
            self.assertAlmostEqual(viewer.slider.val, -np.pi + 0.01, places=8)
            self.assertAlmostEqual(
                viewer.session.state.drive_unwrapped, np.pi + 0.01, places=8
            )
            self.assertLess(
                closure_error(sim.original_alphas_deg, viewer.session.state.rhos), 1e-7
            )
        finally:
            plt.close(viewer.fig)


if __name__ == "__main__":
    unittest.main()
