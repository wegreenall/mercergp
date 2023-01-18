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
    eigenvalue_reshape,
)
from ortho.measure import MaximalEntropyDensity
from mercergp.likelihood import MercerLikelihood, FavardLikelihood
import math

import unittest


class TestEinsumMaker(unittest.TestCase):
    def setUp(self):
        self.order = 10
        self.dimension = 3
        self.x = torch.ones(self.order)
        self.y = 2 * torch.ones(self.order)
        self.z = 3 * torch.ones(self.order)
        self.data = torch.stack((self.x, self.y, self.z), dim=-1)
        # self.data = [self.x, self.y, self.z]

    def test_einsum_shape(self):
        result = eigenvalue_reshape(self.data)
        self.assertTrue(
            result.shape == torch.Size([self.order, self.order, self.order])
        )

    def test_einsum(self):
        result = eigenvalue_reshape(self.data)
        self.assertTrue(
            torch.allclose(
                result[1, 2, 3],
                torch.tensor(6.0),
            )
        )


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
        self.dimension = 1
        self.eigenvalue_generator = SmoothExponentialFasshauer(
            self.order, self.dimension
        )
        self.params = {
            "precision_parameter": torch.Tensor([[self.precision_parameter]]),
            "ard_parameter": torch.Tensor([[self.ard_parameter]]),
        }
        pass

    def test_shape(self):
        params = {
            "precision_parameter": torch.Tensor([self.precision_parameter]),
            "ard_parameter": torch.Tensor([self.ard_parameter]),
        }
        # breakpoint()
        eigens = self.eigenvalue_generator(self.params)
        self.assertEqual(eigens.shape, torch.Size([self.order]))

    def test_values(self):
        eigens = self.eigenvalue_generator(self.params)
        left_term = math.sqrt(2 / (2 + math.sqrt(3)))
        right_term = 1 / (2 + math.sqrt(3))
        true_eigens = left_term * (
            right_term ** torch.linspace(0, self.order - 1, self.order)
        )
        self.assertTrue(torch.allclose(eigens, true_eigens))

    # @unittest.skip("weird")
    def test_differential(self):
        # eigens = self.eigenvalue_generator(self.params)
        # true_eigens = torch.Tensor([])
        altered_params = {
            "precision_parameter": torch.Tensor(
                [self.precision_parameter + 0.5]
            ),
            "ard_parameter": torch.Tensor([self.ard_parameter + 0.5]),
        }
        left_term = math.sqrt(3 / (3 + 1.5 * math.sqrt(3)))
        right_term = 1.5 / (3 + 1.5 * math.sqrt(3))
        true_eigens_2 = left_term * (
            right_term ** torch.linspace(0, self.order - 1, self.order)
        )
        eigens_2 = self.eigenvalue_generator(altered_params)
        self.assertTrue(torch.allclose(eigens_2, true_eigens_2))


class TestMultivariateSmoothExponentialFasshauerEigenvalueGenerator(
    unittest.TestCase
):
    def setUp(self):
        self.order = 10
        self.dimension = 2
        self.precision_parameter = torch.eye(self.dimension)
        self.ard_parameter = torch.eye(self.dimension)
        self.eigenvalue_generator = SmoothExponentialFasshauer(
            self.order, self.dimension
        )
        self.params = {
            "precision_parameter": self.precision_parameter,
            "ard_parameter": self.ard_parameter,
        }
        pass

    def test_shape(self):
        # params = {
        # "precision_parameter": torch.Tensor([self.precision_parameter]),
        # "ard_parameter": torch.Tensor([self.ard_parameter]),
        # }
        eigens = self.eigenvalue_generator(self.params)
        self.assertEqual(
            eigens.shape, torch.Size([self.order ** self.dimension])
        )

    @unittest.skip("Not Implemented for multi dimensional")
    def test_values(self):
        eigens = self.eigenvalue_generator(self.params)
        left_term = math.sqrt(2 / (2 + math.sqrt(3)))
        right_term = 1 / (2 + math.sqrt(3))
        true_eigens = left_term * (
            right_term ** torch.linspace(0, self.order - 1, self.order)
        )
        self.assertTrue(torch.allclose(eigens, true_eigens))

    # @unittest.skip("weird")
    @unittest.skip("Not Implemented for multi dimensional")
    def test_differential(self):
        # eigens = self.eigenvalue_generator(self.params)
        # true_eigens = torch.Tensor([])
        altered_params = {
            "precision_parameter": torch.Tensor(
                [self.precision_parameter + 0.5]
            ),
            "ard_parameter": torch.Tensor([self.ard_parameter + 0.5]),
        }
        left_term = math.sqrt(3 / (3 + 1.5 * math.sqrt(3)))
        right_term = 1.5 / (3 + 1.5 * math.sqrt(3))
        true_eigens_2 = left_term * (
            right_term ** torch.linspace(0, self.order - 1, self.order)
        )
        eigens_2 = self.eigenvalue_generator(altered_params)
        self.assertTrue(torch.allclose(eigens_2, true_eigens_2))


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
