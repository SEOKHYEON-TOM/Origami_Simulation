"""Independent substituted-equation checks; no extra symbolic dependency."""

import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from origami_kinematics import Kinematics


def delta(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def real_quadratic_roots(A, B, C):
    # Tests deliberately use well-conditioned finite roots, away from seams.
    return sorted(float(z.real) for z in np.roots([A, B, C]) if abs(z.imag) < 1e-9)


class SubstitutedRelations(unittest.TestCase):
    def test_repeated_root_uses_input_precision_but_preserves_infeasible_side(self):
        solver = Kinematics([80, 80, 110, 110], driving_crease=2)
        drive = 0.6077184810757758
        expected = np.array([-1.065056942892582, drive, -1.0650569428925818, 0.0])
        states = solver.solve(drive)
        self.assertTrue(states)
        self.assertTrue(
            any(np.linalg.norm(delta(c.rhos, expected)) < 1e-7 for c in states)
        )
        self.assertTrue(any("DISCRIMINANT_ROUNDOFF" in c.events for c in states))
        self.assertEqual(solver.solve(drive - 1e-12), [])
        self.assertEqual(solver.last_status, "NO_SOLUTION")
        self.assertEqual(len(solver.solve(drive + 1e-12)), 2)

    def assert_root_sets(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for wanted in expected:
            self.assertTrue(any(abs(delta(q, wanted)) < 1e-8 for q in actual))

    def test_paired_adjacent_sectors_use_the_substituted_quadratic(self):
        for adeg, bdeg in [(80, 110), (60, 70), (40, 70), (100, 130)]:
            a, b = np.deg2rad([adeg, bdeg])
            solver = Kinematics([adeg, adeg, bdeg, bdeg], driving_crease=4)
            for drive_deg in [-140, -60, -10, 10, 60, 140]:
                d = np.deg2rad(drive_deg)
                x = np.tan(d / 2)
                ys = real_quadratic_roots(
                    np.sin(a - b), 2 * np.sin(a) * x, -np.sin(a + b)
                )
                actual = [c.rhos[0] for c in solver.solve(d)]
                with self.subTest(a=adeg, b=bdeg, drive=drive_deg):
                    self.assert_root_sets(actual, [2 * np.arctan(y) for y in ys])

    def test_opposite_equal_sectors_have_a_symmetric_biquadratic(self):
        for adeg, bdeg, cdeg in [(70, 80, 100), (60, 90, 120), (100, 70, 130)]:
            a, b, c = np.deg2rad([adeg, bdeg, cdeg])
            u = np.cos(c) - np.cos(b)
            v = np.cos(2 * a - c) - np.cos(b)
            w = np.cos(2 * a + c) - np.cos(b)
            solver = Kinematics([adeg, bdeg, adeg, cdeg], driving_crease=4)
            for drive_deg in [-140, -60, -10, 10, 60, 140]:
                d = np.deg2rad(drive_deg)
                x = np.tan(d / 2)
                ys = real_quadratic_roots(
                    u + v * x * x, 4 * np.sin(a) ** 2 * x, w + u * x * x
                )
                states = solver.solve(d)
                with self.subTest(a=adeg, b=bdeg, c=cdeg, drive=drive_deg):
                    self.assert_root_sets(
                        [q.rhos[0] for q in states], [2 * np.arctan(y) for y in ys]
                    )
                    for q in states:
                        y = np.tan(q.rhos[0] / 2)
                        f = (
                            lambda x, y: v * x * x * y * y
                            + u * (x * x + y * y)
                            + 4 * np.sin(a) ** 2 * x * y
                            + w
                        )
                        self.assertAlmostEqual(f(x, y), 0, places=9)
                        self.assertAlmostEqual(f(x, y), f(y, x), places=9)

    def test_equal_adjacent_pair_does_not_imply_global_reflection(self):
        geometry = Kinematics([70, 70, 100, 120]).geometry
        self.assertTrue(geometry["special"])
        self.assertTrue(geometry["equal_angle_pairs"])
        self.assertFalse(geometry["reflection_symmetry"])
        self.assertFalse(geometry["rotational_symmetry"])

    def test_bird_foot_pi_family_survives_finite_coordinate_cancellation(self):
        solver = Kinematics([80, 80, 110, 110], driving_crease=4)
        self.assertEqual(solver.solve(np.pi), [])
        self.assertEqual(solver.last_status, "CONTINUUM")
        self.assertTrue(solver.last_families)
        for free in np.linspace(-np.pi, np.pi, 31):
            c = solver.last_families[0]["evaluate"](float(free))
            self.assertIsNotNone(c)
            self.assertLess(c.residual, 1e-9)
            self.assertLess(abs(delta(c.rhos[0], free)), 1e-9)
            self.assertLess(abs(delta(c.rhos[3], np.pi)), 1e-9)


if __name__ == "__main__":
    unittest.main()
