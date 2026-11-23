import unittest

import numpy as np

from cspca import fit_cspca


class ProjectionTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(17).normal(size=(18, 6))
        self.y = np.repeat([2, 5, 7], 6)

    def test_matches_explicit_label_kernel(self):
        mean, axes = fit_cspca(self.x, self.y, 3, 4.0)
        z = self.x - self.x.mean(axis=0)
        delta = (self.y[:, None] == self.y[None, :]).astype(float)
        direct = z.T @ delta @ z + 4.0 * (z.T @ z)
        _, ref = np.linalg.eigh(direct)
        ref = ref[:, -3:]
        np.testing.assert_allclose(mean, self.x.mean(axis=0))
        np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(axes @ axes.T, ref @ ref.T, atol=1e-10)

    def test_class_names_and_pixel_offsets_do_not_change_the_subspace(self):
        mean, axes = fit_cspca(self.x, self.y, 2, 0.1)
        shift = np.arange(6)
        mean2, axes2 = fit_cspca(self.x + shift, -self.y, 2, 0.1)
        np.testing.assert_allclose(mean2, mean + shift, atol=1e-10)
        np.testing.assert_allclose(axes @ axes.T, axes2 @ axes2.T, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
