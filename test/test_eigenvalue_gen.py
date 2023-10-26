import torch
import torch.distributions as D
from ortho.orthopoly import (
    OrthogonalBasisFunction,
    SymmetricOrthonormalPolynomial,
)
from ortho.basis_functions import Basis
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
        self.params = {
            "scale": torch.Tensor([self.scale]),
            "shape": self.shape,
            "degree": torch.Tensor([self.degree]),
            "variance_parameter": torch.Tensor([1.0]),
        }

        self.eigenvalue_generator = PolynomialEigenvalues(self.order)
        pass

    def test_shape(self):
        eigens = self.eigenvalue_generator(self.params)
        self.assertEqual(eigens.shape, torch.Size([self.order]))

    def test_scale_derivatives_shape(self):
        eigens = self.eigenvalue_generator._scale_derivatives(self.params)
        self.assertEqual(eigens.shape, torch.Size([self.order]))

    def test_shape_derivatives_shape(self):
        eigens = self.eigenvalue_generator._shape_derivatives(self.params)
        self.assertEqual(eigens.shape, torch.Size([self.order]))

    def test_degree_derivatives_shape(self):
        eigens = self.eigenvalue_generator._degree_derivatives(self.params)
        self.assertEqual(eigens.shape, torch.Size([self.order]))


class TestSmoothExponentialFasshauerEigenvalueGenerator(unittest.TestCase):
    def setUp(self):
        self.order = 10
        self.precision_parameter = 1.0
        self.ard_parameter = 1.0
        self.variance_parameter = 1.0
        self.dimension = 1
        self.eigenvalue_generator = SmoothExponentialFasshauer(
            self.order, self.dimension
        )
        self.params = {
            "precision_parameter": torch.Tensor([[self.precision_parameter]]),
            "ard_parameter": torch.Tensor([[self.ard_parameter]]),
            "variance_parameter": torch.Tensor([[self.variance_parameter]]),
        }
        pass

    def test_shape(self):
        params = {
            "precision_parameter": torch.Tensor([self.precision_parameter]),
            "ard_parameter": torch.Tensor([self.ard_parameter]),
            "variance_parameter": torch.Tensor([1.0]),
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
        altered_params = {
            "precision_parameter": torch.Tensor(
                [self.precision_parameter + 0.5]
            ),
            "ard_parameter": torch.Tensor([self.ard_parameter + 0.5]),
            "variance_parameter": torch.Tensor([1.0]),
        }
        left_term = math.sqrt(3 / (3 + 1.5 * math.sqrt(3)))
        right_term = 1.5 / (3 + 1.5 * math.sqrt(3))
        true_eigens_2 = left_term * (
            right_term ** torch.linspace(0, self.order - 1, self.order)
        )
        eigens_2 = self.eigenvalue_generator(altered_params)
        self.assertTrue(torch.allclose(eigens_2, true_eigens_2))

    def test_derivatives_ard(self):
        derivatives = self.eigenvalue_generator._ard_parameter_derivative(
            self.params
        )
        self.assertTrue(derivatives.shape, torch.Size([self.order]))

    def test_derivatives_precision(self):
        derivatives = (
            self.eigenvalue_generator._precision_parameter_derivative(
                self.params
            )
        )
        self.assertTrue(derivatives.shape, torch.Size([self.order]))

    def test_derivatives(self):
        # a test when it's one dict;
        derivatives = self.eigenvalue_generator.derivatives(self.params)
        self.assertTrue(isinstance(derivatives, list))
        self.assertTrue(len(derivatives) == 1)

    def test_inverse_variance(self):
        eigens = self.eigenvalue_generator(self.params)
        initial_params = {
            "precision_parameter": torch.Tensor([1.0]),
            "ard_parameter": torch.Tensor([5.0]),
            "variance_parameter": torch.Tensor([5.0]),
        }
        result_params = self.eigenvalue_generator.inverse(
            eigens, initial_params
        )
        self.assertTrue(
            torch.allclose(
                result_params["variance_parameter"],
                torch.Tensor([[self.variance_parameter]]),
                rtol=1e-3,
            )
        )

    def test_inverse_ard(self):
        eigens = self.eigenvalue_generator(self.params)
        initial_params = {
            "precision_parameter": torch.Tensor([1.0]),
            "ard_parameter": torch.Tensor([5.0]),
            "variance_parameter": torch.Tensor([5.0]),
        }
        result_params = self.eigenvalue_generator.inverse(
            eigens, initial_params
        )
        self.assertTrue(
            torch.allclose(
                result_params["ard_parameter"],
                torch.Tensor([[self.ard_parameter]]),
                rtol=1e-3,
            )
        )


class TestMultivariateSmoothExponentialFasshauerEigenvalueGenerator(
    unittest.TestCase
):
    def setUp(self):
        self.order = 10
        self.dimension = 2
        self.precision_parameter = torch.Tensor(
            [[1.0]]
        )  # torch.eye(self.dimension)
        self.ard_parameter = torch.Tensor([[1.0]])  # torch.eye(self.dimension)

        self.eigenvalue_generator = SmoothExponentialFasshauer(
            self.order, self.dimension
        )
        self.params = 2 * (
            {
                "precision_parameter": self.precision_parameter,
                "ard_parameter": self.ard_parameter,
                "variance_parameter": torch.Tensor([1.0]),
            },
        )
        pass

    def test_shape(self):
        eigens = self.eigenvalue_generator(self.params)
        self.assertEqual(
            eigens.shape, torch.Size([self.order**self.dimension])
        )

    # @unittest.skip("Not Implemented for multi dimensional")
    def test_values(self):
        # get the eigens
        eigens = self.eigenvalue_generator(self.params)

        # get the apparent true values
        left_term = math.sqrt(2 / (2 + math.sqrt(3)))
        right_term = 1 / (2 + math.sqrt(3))
        base_eigens = left_term * (
            right_term ** torch.linspace(0, self.order - 1, self.order)
        )

        # now construct tensor product
        true_eigens = torch.einsum(
            "i,j->ij", base_eigens, base_eigens
        ).flatten()

        # check if they're the same
        self.assertTrue(torch.allclose(eigens, true_eigens))

    def test_differential(self):
        # set up the sets of parameters
        altered_params_1 = {
            "precision_parameter": torch.Tensor(
                [self.precision_parameter + 0.5]
            ),
            "ard_parameter": torch.Tensor([self.ard_parameter + 0.5]),
            "variance_parameter": torch.Tensor([1.0]),
        }
        altered_params_2 = {
            "precision_parameter": torch.Tensor(
                [self.precision_parameter + 0.5]
            ),
            "ard_parameter": torch.Tensor([self.ard_parameter + 0.5]),
            "variance_parameter": torch.Tensor([1.0]),
        }
        altered_params = [altered_params_1, altered_params_2]

        # alterations
        left_term = math.sqrt(3 / (3 + 1.5 * math.sqrt(3)))
        right_term = 1.5 / (3 + 1.5 * math.sqrt(3))
        base_eigens = left_term * (
            right_term ** torch.linspace(0, self.order - 1, self.order)
        )
        true_eigens = torch.einsum(
            "i,j->ij", base_eigens, base_eigens
        ).flatten()

        eigens_2 = self.eigenvalue_generator(altered_params)
        self.assertTrue(torch.allclose(eigens_2, true_eigens))

    def test_inverse(self):
        params = 2 * (
            {
                "precision_parameter": torch.Tensor([0.1]),
                "ard_parameter": torch.Tensor([1.0]),
                "variance_parameter": torch.Tensor([1.0]),
            },
        )
        eigens = self.eigenvalue_generator(params)
        initial_params = [
            {
                "precision_parameter": torch.Tensor([0.1]),
                "ard_parameter": torch.Tensor([5.0]),
                "variance_parameter": torch.Tensor([5.0]),
            }
        ] * 2
        result_params = self.eigenvalue_generator.inverse(
            eigens, initial_params
        )

        # breakpoint()
        self.assertTrue(
            torch.allclose(
                result_params[0]["variance_parameter"],
                params[0]["variance_parameter"],
                rtol=1e-1,
            )
        )
        self.assertTrue(
            torch.allclose(
                result_params[0]["ard_parameter"],
                params[0]["ard_parameter"],
                rtol=1e-1,
            )
        )
        self.assertTrue(
            torch.allclose(
                result_params[1]["variance_parameter"],
                params[1]["variance_parameter"],
                rtol=1e-1,
            )
        )
        self.assertTrue(
            torch.allclose(
                result_params[1]["ard_parameter"],
                params[1]["ard_parameter"],
                rtol=1e-1,
            )
        )


class TestFavardEigenvalueGenerator(unittest.TestCase):
    def setUp(self):
        self.order = 10
        self.precision_parameter = 1.0
        self.ard_parameter = 1.0
        betas = torch.Tensor([1.0] * 2 * self.order)
        gammas = torch.Tensor([1.0] * 2 * self.order)
        params = {
            "precision_parameter": torch.Tensor([self.precision_parameter]),
            "ard_parameter": torch.Tensor([self.ard_parameter]),
            "degree": 6,
        }
        self.basis = Basis(
            OrthogonalBasisFunction(self.order, betas, gammas),
            1,
            self.order,
            params,
        )

        self.eigenvalue_generator = FavardEigenvalues(self.order, self.basis)

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
