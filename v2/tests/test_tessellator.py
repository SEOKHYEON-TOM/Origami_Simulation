"""Independent crease, mode, continuation, contact and strip geometry checks."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/origami-tess-mpl")
import sys, unittest, io, contextlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from origami_tesellator_1D import OrigamiTesellator1D

CASES = [
    [70, 20, 110, 160],
    [60, 90, 135, 75],
    [70, 80, 70, 100],
    [40, 80, 50, 110],
    [70, 70, 120, 110],
    [61, 91, 136, 76],
    [60, 60, 120, 120],
    [70, 110, 70, 110],
    [90] * 4,
    [70, 20, 110, 159.999],
]


def independent_closure(angles, folds):
    result = np.eye(3)
    for a, r in zip(angles, folds):
        cr, sr, ca, sa = np.cos(r), np.sin(r), np.cos(a), np.sin(a)
        result = (
            result
            @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
            @ np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
        )
    return np.linalg.norm(result - np.eye(3))


class StripSnapshots(unittest.TestCase):
    def make(self, alphas=(70, 20, 110, 160), *, iout=2, sigma=1, n=1, **kwargs):
        config = dict(alphas=alphas, iout=iout, sigma=sigma)
        for key in ("mode_id", "branch_id", "initial_state", "free_angle"):
            if key in kwargs:
                config[key] = kwargs.pop(key)
        kwargs.setdefault("contact", False)
        return OrigamiTesellator1D([config], num_periods=n, verbose=False, **kwargs)

    def assert_geometry(self, s):
        for state, vertices in zip(s.unit_states, s.global_vertices):
            cell = s.cells[state.cell_index]
            self.assertLess(independent_closure(cell["alphas"], state.folds), 2e-8)
            rays = np.array([vertices[cell["CLV"][k]] - vertices[0] for k in range(4)])
            rays /= np.linalg.norm(rays, axis=1)[:, None]
            np.testing.assert_allclose(
                np.sum(rays * np.roll(rays, -1, axis=0), axis=1),
                np.cos(cell["alphas"]),
                atol=2e-8,
            )

    def test_shared_classification_and_explicit_driver_near_euclidean(self):
        for alphas in (
            [70, 20, 110, 160],
            [70, 80, 70, 100],
            [61, 91, 136, 76],
            [70, 20, 110, 159.999],
        ):
            for iout in (1, 2, 3):
                s = self.make(alphas, iout=iout)
                self.assertEqual(s.compute_strip_kinematics(np.pi / 3), 1, s.last_error)
                self.assertAlmostEqual(s.get_fold_angles(0)[0], np.pi / 3, places=12)
                c = s.cells[0]
                self.assertEqual(c["fold_indices"][0], c["solver"].drive_index)
                self.assertEqual(c["geometry"]["case"], c["solver"].geometry["case"])
                self.assert_geometry(s)
        self.assertEqual(s.cells[0]["geometry"]["case"], "elliptic")

    def test_cyclic_reindex_preserves_physical_candidate(self):
        s = self.make([60, 90, 135, 75])
        self.assertEqual(s.compute_strip_kinematics(0.9), 1)
        folds = s.get_fold_angles(0)
        for shift in range(4):
            rotated = self.make(np.roll([60, 90, 135, 75], -shift))
            cell = rotated.cells[0]
            packet = cell["solver"].configuration_solutions(float(folds[shift]))
            expected = np.roll(folds, -shift)
            self.assertTrue(
                any(
                    np.linalg.norm(
                        (rotated._canonical(cell, c.rhos) - expected + np.pi)
                        % (2 * np.pi)
                        - np.pi
                    )
                    < 1e-7
                    for c in packet["candidates"]
                )
            )

    def test_mixed_cell_numbering_and_scalar_propagation(self):
        configs = [
            dict(alphas=a, iout=2, sigma=1)
            for a in ([70, 20, 110, 160], [70, 80, 70, 100], [61, 91, 136, 76])
        ]
        s = OrigamiTesellator1D(configs, 1, contact=False, verbose=False)
        self.assertEqual(s.compute_strip_kinematics(1.5), 3, s.last_error)
        for i in range(1, 3):
            self.assertAlmostEqual(
                s.get_fold_angles(i)[0], s.get_fold_angles(i - 1)[2], places=12
            )
        self.assert_geometry(s)

    def test_shared_main_cobweb_debug_and_order_invariance(self):
        s = self.make([60, 90, 135, 75], iout=1, sigma=-1)
        self.assertEqual(s.compute_strip_kinematics(0.6), 1)
        q = s.rhos.copy()
        before = s.global_vertices[0].copy()
        ax = s.plot_cobweb(np.rad2deg(0.6), resolution=43)
        self.addCleanup(plt.close, ax.figure)
        np.testing.assert_allclose(
            ax.origami_result.states[0].folds, s.get_fold_angles(0)
        )
        with contextlib.redirect_stdout(io.StringIO()):
            seq = s.debug_kinematics(np.rad2deg(0.6))
        self.assertAlmostEqual(seq[0][2], np.rad2deg(s.get_fold_angles(0)[1]))
        np.testing.assert_array_equal(q, s.rhos)
        np.testing.assert_array_equal(before, s.global_vertices[0])
        model = s.cells[0]["solver"]
        original = model.configuration_solutions

        def reverse(d):
            p = original(d)
            p["candidates"] = tuple(reversed(p["candidates"]))
            return p

        with patch.object(model, "configuration_solutions", side_effect=reverse):
            self.assertEqual(s.compute_strip_kinematics(0.6), 1)
        np.testing.assert_allclose(q, s.rhos, atol=1e-10)

    def test_legacy_output_hint_is_explicitly_ambiguous(self):
        s = self.make([60, 90, 135, 75], iout=1, sigma=-1, sigma_role="output")
        self.assertEqual(s.compute_strip_kinematics(0.6), 0)
        self.assertEqual(s.last_result.status, "AMBIGUOUS")
        s = self.make([60, 90, 135, 75], iout=1, sigma=-1, mode_id="unique_c1")
        self.assertEqual(s.compute_strip_kinematics(0.6), 1, s.last_error)

    def test_elliptic_gap_is_not_interpolated_and_positive_domain_not_lost(self):
        s = self.make([40, 80, 50, 110])
        self.assertEqual(s.compute_strip_kinematics(0.0), 0)
        self.assertEqual(s.last_result.status, "DRIVE_EXTREMUM")
        curves = s.cobweb_data(resolution=31)
        xs = [part["xy"][:, 0] for part in curves if not part["family"]]
        self.assertTrue(any(np.all(x > 0) for x in xs))
        self.assertTrue(any(np.all(x < 0) for x in xs))
        self.assertTrue(all(np.all(x > 0) or np.all(x < 0) for x in xs))
        self.assertEqual(s.compute_strip_kinematics(np.deg2rad(80)), 1, s.last_error)

    def test_family_needs_explicit_free_parameter(self):
        s = self.make([90] * 4, iout=3)
        self.assertEqual(s.compute_strip_kinematics(0.0), 0)
        self.assertEqual(s.last_result.status, "UNDERDETERMINED")
        s = self.make([90] * 4, iout=3, free_angle=0.4)
        self.assertEqual(s.compute_strip_kinematics(0.0), 1, s.last_error)
        self.assertEqual(s.unit_states[0].family_free_crease, 1)
        self.assertAlmostEqual(s.get_fold_angles(0)[1], 0.4)
        self.assert_geometry(s)
        self.assertTrue(any(part["family"] for part in s.cobweb_data()))

    def test_representative_matrix_has_closure_or_explicit_failure(self):
        accepted = 0
        for a in CASES:
            for iout in (1, 2, 3):
                for sigma in (-1, 1):
                    with self.subTest(alphas=a, iout=iout, sigma=sigma):
                        s = self.make(a, iout=iout, sigma=sigma)
                        count = s.compute_strip_kinematics(1.4)
                        if count:
                            accepted += 1
                            self.assert_geometry(s)
                            self.assertTrue(
                                s.validate_with_algebraic_solver(0, verbose=False)
                            )
                        else:
                            self.assertIn(
                                s.last_result.status,
                                (
                                    "DRIVE_EXTREMUM",
                                    "AMBIGUOUS",
                                    "UNDERDETERMINED",
                                    "MODE_UNAVAILABLE",
                                ),
                            )
        self.assertGreater(accepted, 40)

    def test_shared_models_have_independent_states(self):
        cfg = dict(alphas=[70, 20, 110, 160], iout=2, sigma=1)
        s = OrigamiTesellator1D([cfg, cfg], 2, contact=False, verbose=False)
        self.assertIs(s.cells[0]["solver"], s.cells[1]["solver"])
        self.assertEqual(s.compute_strip_kinematics(0.7), 4)
        self.assertEqual(len({id(x) for x in s.unit_states}), 4)
        self.assertFalse(
            np.shares_memory(s.unit_states[0].folds, s.unit_states[1].folds)
        )

    def test_invalid_input_and_legacy_array_order(self):
        for key, value in [
            ("scale_factor", 0),
            ("scale_factor", np.inf),
            ("lengths", (1, 0, 1)),
            ("num_periods", 1.5),
        ]:
            args = dict(
                cell_configs=[dict(alphas=[70, 20, 110, 160])],
                num_periods=1,
                verbose=False,
            )
            args[key] = value
            with self.assertRaises(ValueError):
                OrigamiTesellator1D(**args)
        s = self.make()
        s.compute_strip_kinematics(0.6)
        np.testing.assert_array_equal(s.rhos[0], s.get_fold_angles(0, order="solver"))
        np.testing.assert_array_equal(s.rhos[0, [3, 0, 1, 2]], s.get_fold_angles(0))
        with self.assertRaises(IndexError):
            s.get_fold_angles(1)

    def test_large_uncertain_cosine_is_not_clipped_into_a_solution(self):
        from origami_simulator import PathError

        with self.assertRaisesRegex(PathError, "NUMERICAL_FAILURE"):
            OrigamiTesellator1D._oracle(dict(A=1e14, B=-1e14 + 1.5), 0.0)


class StripDynamics(unittest.TestCase):
    make = StripSnapshots.make
    assert_geometry = StripSnapshots.assert_geometry

    def test_forward_reverse_crosses_zero_on_same_per_unit_paths(self):
        s = self.make(n=3)
        self.assertEqual(s.compute_strip_kinematics(0.6), 3)
        q = s.rhos.copy()
        ids = [x.branch_id for x in s.unit_states]
        for target in (1.2, -0.4, 0.6):
            result = s.update(target)
            self.assertTrue(result.complete, result.termination)
            self.assertEqual([x.branch_id for x in result.states], ids)
            self.assert_geometry(s)
        np.testing.assert_allclose(q, s.rhos, atol=2e-8)

    def test_frame_size_and_diagnostic_sampling_do_not_change_paths(self):
        a, b = self.make(n=2), self.make(n=2)
        a.compute_strip_kinematics(0.6)
        b.compute_strip_kinematics(0.6)
        self.assertTrue(a.update(1.1).complete)
        for t in np.linspace(0.6, 1.1, 24):
            self.assertTrue(b.update(float(t)).complete)
        np.testing.assert_allclose(a.rhos, b.rhos, atol=2e-8)
        q = a.rhos.copy()
        ax = a.plot_cobweb(np.rad2deg(1.1), resolution=11)
        plt.close(ax.figure)
        ax = a.plot_cobweb(np.rad2deg(1.1), resolution=97)
        plt.close(ax.figure)
        np.testing.assert_array_equal(q, a.rhos)

    def test_fixed_drive_free_parameter_and_underdetermined_exit(self):
        s = self.make([90] * 4, iout=3, free_angle=0.4)
        self.assertEqual(s.compute_strip_kinematics(0.0), 1)
        self.assertTrue(s.update(0.0, free_angles={0: 0.8}).complete)
        self.assertAlmostEqual(s.get_fold_angles(0)[1], 0.8)
        before = s.last_result
        q = s.rhos.copy()
        r = s.update(0.1)
        self.assertEqual(r.status, "UNDERDETERMINED")
        self.assertIs(s.last_result, before)
        np.testing.assert_array_equal(q, s.rhos)

    def test_contact_is_effective_and_enabling_penetration_is_atomic(self):
        blocked = self.make([70, 110, 70, 110], sigma=-1, contact=True, n=2)
        self.assertEqual(blocked.compute_strip_kinematics(0.6), 0)
        self.assertEqual(blocked.last_result.status, "CONTACT_LIMIT")
        s = self.make([70, 110, 70, 110], sigma=-1, n=2)
        self.assertEqual(s.compute_strip_kinematics(0.6), 2)
        q = s.rhos.copy()
        old = s.last_result
        self.assertFalse(s.set_contact(True))
        self.assertFalse(s.contact)
        self.assertIs(s.last_result, old)
        np.testing.assert_array_equal(q, s.rhos)
        self.assertIn("penetration", s.last_error)
        self.assertFalse(old.global_contact_checked)

    def test_continuation_failure_does_not_partially_replace_pose(self):
        from origami_simulator import PathError

        s = self.make(n=3)
        s.compute_strip_kinematics(0.6)
        q = s.rhos.copy()
        vertices = np.array(s.global_vertices)
        old = s.last_result
        with patch.object(
            s.cells[0]["solver"],
            "continue_configuration",
            side_effect=PathError("NUMERICAL_FAILURE", "injected"),
        ) as call:
            result = s.update(0.7)
            self.assertGreater(call.call_count, 1)
        self.assertEqual(result.status, "NUMERICAL_FAILURE")
        self.assertEqual(result.termination["unit_index"], 0)
        self.assertIs(s.last_result, old)
        np.testing.assert_array_equal(q, s.rhos)
        np.testing.assert_array_equal(vertices, s.global_vertices)

    def test_actual_slider_checkbox_callbacks_preserve_accepted_state(self):
        s = self.make(n=2, contact=True)
        fig = s.setup_interactive_viewer(show=False)
        self.addCleanup(plt.close, fig)
        slider = fig.origami_slider
        check = fig.origami_contact
        q = s.rhos.copy()
        value = slider.val
        before = fig.origami_guard["updates"]
        slider.set_val(200.0)
        self.assertEqual(fig.origami_guard["updates"], before + 1)
        self.assertAlmostEqual(slider.val, value)
        np.testing.assert_array_equal(q, s.rhos)
        self.assertIn("DOMAIN_END", s.last_error)
        check.set_active(0)
        self.assertFalse(s.contact)
        slider.set_val(200.0)
        self.assertIsNone(s.last_error)
        q = s.rhos.copy()
        check.set_active(0)
        self.assertFalse(s.contact)
        self.assertFalse(check.get_status()[0])
        np.testing.assert_array_equal(q, s.rhos)

    def test_actual_family_slider_keeps_driving_crease_fixed(self):
        s = self.make([90] * 4, iout=3, free_angle=0.4)
        fig = s.setup_interactive_viewer(show=False, initial_rho0_deg=0.0)
        self.addCleanup(plt.close, fig)
        self.assertFalse(fig.origami_slider.active)
        self.assertTrue(fig.origami_free_slider.active)
        fig.origami_free_slider.set_val(30.0)
        self.assertAlmostEqual(s.get_fold_angles(0)[0], 0.0)
        self.assertAlmostEqual(s.get_fold_angles(0)[1], np.pi / 6)
        fig.origami_slider.set_val(1.0)
        self.assertAlmostEqual(fig.origami_slider.val, 0.0)
        self.assertIn("UNDERDETERMINED", s.last_error)

    def test_public_candidate_packet_cache_preserves_failure_and_arrays(self):
        s = self.make([40, 80, 50, 110])
        model = s.cells[0]["solver"]
        p = model.configuration_solutions(0.0)
        self.assertEqual(p["status"], "NO_SOLUTION")
        a = model.configuration_solutions(1.4)
        original = a["candidates"][0].rhos.copy()
        a["candidates"][0].rhos[:] = 42
        b = model.configuration_solutions(1.4)
        np.testing.assert_array_equal(b["candidates"][0].rhos, original)
        self.assertEqual(model.configuration_solutions(0.0)["status"], "NO_SOLUTION")

    def test_regular_path_limits_keep_isolated_contact_at_exact_zero(self):
        s = self.make([70, 110, 70, 110], branch_id="collinear_other_c0", contact=True)
        self.assertEqual(s.compute_strip_kinematics(0.0), 1, s.last_error)
        np.testing.assert_allclose(
            s.get_fold_angles(0), [0, np.pi, 0, -np.pi], atol=1e-10
        )
        self.assertIsNone(s.unit_states[0].family_free_crease)
        q = s.rhos.copy()
        r = s.update(0.001)
        self.assertEqual(r.status, "CONTACT_LIMIT")
        self.assertEqual(r.termination["branch_id"], "collinear_other_c0")
        np.testing.assert_array_equal(q, s.rhos)

    def test_available_modes_and_family_parameters_use_tess_numbering(self):
        s = self.make([60, 90, 135, 75], iout=1)
        choices = s.available_configurations(0.6)
        self.assertEqual(
            {r["mode_id"] for r in choices["candidates"]}, {"unique_c0", "unique_c1"}
        )
        self.assertTrue(
            all(abs(r["folds"][0] - 0.6) < 1e-12 for r in choices["candidates"])
        )
        family = self.make([90] * 4).available_configurations(0.0)
        self.assertEqual(family["status"], "CONTINUUM")
        self.assertEqual(family["families"][0]["free_crease"], 1)

    def test_initial_states_need_resolved_paths_at_singularities(self):
        original = self.make()
        original.compute_strip_kinematics(0.6)
        seed = original.get_fold_angles(0)
        continued = self.make(initial_state=seed)
        self.assertEqual(
            continued.compute_strip_kinematics(0.7), 1, continued.last_error
        )
        family = self.make([90] * 4, initial_state=[0, 0.4, 0, 0.4])
        self.assertEqual(family.compute_strip_kinematics(0.0), 1)
        self.assertEqual(family.update(0.1).status, "UNDERDETERMINED")
        named = self.make(branch_id="ff_minus_c0", initial_state=[0, 0, 0, 0])
        self.assertEqual(named.compute_strip_kinematics(0.2), 1, named.last_error)
        self.assertEqual(named.unit_states[0].mode_id, "ff_minus")

    def test_atlas_resolution_and_general_candidate_order_keep_identity(self):
        results = []
        for resolution in (81, 129, 257):
            s = self.make([61, 91, 136, 76], n=2, atlas_resolution=resolution)
            original = s.cells[0]["solver"].configuration_solutions

            def reversed_packet(d):
                p = original(d)
                p["candidates"] = tuple(reversed(p["candidates"]))
                return p

            with patch.object(
                s.cells[0]["solver"],
                "configuration_solutions",
                side_effect=reversed_packet,
            ):
                self.assertEqual(s.compute_strip_kinematics(0.8), 2, s.last_error)
            self.assertTrue(s.update(1.0).complete)
            results.append((s.rhos.copy(), [u.branch_id for u in s.unit_states]))
        for q, ids in results[1:]:
            np.testing.assert_allclose(q, results[0][0], atol=1e-8)
            self.assertEqual(ids, results[0][1])

    def test_gui_can_start_a_family_at_exact_positive_pi(self):
        s = self.make([90] * 4, iout=3, free_angle=0.4, contact=True)
        fig = s.setup_interactive_viewer(show=False, initial_rho0_deg=180.0)
        self.addCleanup(plt.close, fig)
        self.assertIsNone(s.last_error)
        self.assertAlmostEqual(s.get_fold_angles(0)[0], np.pi)
        self.assertAlmostEqual(fig.origami_slider.val, 180.0)
        self.assertFalse(fig.origami_slider.active)
        self.assertTrue(fig.origami_free_slider.active)
        curves = s.cobweb_data()
        self.assertTrue(
            any(p["family"] and np.allclose(p["xy"][:, 0], np.pi) for p in curves)
        )


if __name__ == "__main__":
    unittest.main()
