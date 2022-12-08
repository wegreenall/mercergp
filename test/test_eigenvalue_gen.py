import torch
import torch.distributions as D
from ortho.orthopoly import (
    OrthogonalBasisFunction,
    SymmetricOrthonormalPolynomial,
)
from mercergp.eigenvalue_gen import (
    PolynomialEigenvalues,
    SmoothExponentialFasshauer,
    FavardEigenvalues,
)
from ortho.measure import MaximalEntropyDensity
from mercergp.likelihood import MercerLikelihood, FavardLikelihood

import unittest


class TestPolynomialEigenvalueGenerator(unittest.TestCase):
    def setUp(self):
        self.order = 10
        self.scale = 1.0
        self.shape = torch.linspace(1.0, 0.0, self.order)
        self.degree = 4.0

        self.eigenvalue_generator = PolynomialEigenvalues(self.order)
        pass

    def test_shape(self):
        params = {
            "scale": torch.Tensor([self.scale]),
            "shape": self.shape,
            "degree": torch.Tensor([self.degree]),
        }
        eigens = self.eigenvalue_generator(params)
        self.assertEqual(eigens.shape, torch.Size([self.order]))


class TestSmoothExponentialFasshauerEigenvalueGenerator(unittest.TestCase):
    def setUp(self):
        self.order = 10
        self.precision_parameter = 1.0
        self.ard_parameter = 1.0

        self.eigenvalue_generator = SmoothExponentialFasshauer(self.order)
        pass

    def test_shape(self):
        params = {
            "precision_parameter": torch.Tensor([self.precision_parameter]),
            "ard_parameter": torch.Tensor([self.ard_parameter]),
        }
        eigens = self.eigenvalue_generator(params)
        self.assertEqual(eigens.shape, torch.Size([self.order]))


class TestFavardEigenvalueGenerator(unittest.TestCase):
    def setUp(self):
        self.order = 10
        self.precision_parameter = 1.0
        self.ard_parameter = 1.0

        self.eigenvalue_generator = FavardEigenvalues(
            self.order,
            torch.Tensor([1.0]),
            torch.Tensor([1.0]),
            torch.Tensor([1.0]),
        )
        pass

    def test_shape(self):
        params = {
            "precision_parameter": torch.Tensor([self.precision_parameter]),
            "ard_parameter": torch.Tensor([self.ard_parameter]),
            "degree": 6,
        }
        eigens = self.eigenvalue_generator(params)
        self.assertEqual(eigens.shape, torch.Size([self.order]))


if __name__ == "__main__":
    unittest.main()
