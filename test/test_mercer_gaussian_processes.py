# test_mercer_gaussian_process.py
import unittest
import torch
from basis_functions import (
    smooth_exponential_basis,
    smooth_exponential_eigenvalues,
    Basis,
)

from kernel import MercerKernel
from MGP import MercerGP


class TestMercerGP(unittest.TestCase):
    def setUp(self):
        # basis parameters
        self.basis_function = smooth_exponential_basis
        self.dimension = 1
        self.order = 10

        # parameters
        l_se = torch.Tensor([[5]])
        sigma_se = torch.Tensor([1])
        prec = torch.Tensor([1])
        sigma_e = torch.Tensor([0.01])
        se_kernel_args = {
            "ard_parameter": l_se,
            "variance_parameter": sigma_se,
            "noise_parameter": sigma_e,
            "precision_parameter": prec,
        }

        self.params = se_kernel_args

        # basis function
        self.basis = Basis(
            self.basis_function, self.dimension, self.order, self.params
        )

        self.eigenvalues = smooth_exponential_eigenvalues(
            self.order, self.params
        )

        # gp perameters
        self.kernel = MercerKernel(
            self.order, self.basis, self.eigenvalues, self.params
        )

        self.mercer_gp = MercerGP(
            self.basis, self.eigenvalues, self.order, self.kernel
        )
        return

    def test_adding_data(self):
        x = torch.Tensor([0.1, 4, 5.24, 7])
        x2 = torch.Tensor([2, 3, 4, 5])
        y = torch.Tensor([3, 4, 5, 6])
        y2 = torch.Tensor([9, 2, 7, 8])

        self.mercer_gp.add_data(x, y)
        self.mercer_gp.add_data(x2, y2)
        xs, ys = self.mercer_gp.get_inputs(), self.mercer_gp.get_outputs()

        self.assertTrue(
            (xs == torch.Tensor([0.1, 4, 5.24, 7, 2, 3, 4, 5])).all()
        )
        self.assertTrue((ys == torch.Tensor([3, 4, 5, 6, 9, 2, 7, 8])).all())

    def test_sampling(self):

        return

    def test_coefficients_shape_flat(self):
        mercer_gp = MercerGP(
            self.basis, self.eigenvalues, self.order, self.kernel
        )
        test_inputs = torch.linspace(0, 1, 20)
        test_outputs = torch.linspace(0, 1, 20)

        mercer_gp.add_data(test_inputs, test_outputs)

        coefficients = mercer_gp._calculate_posterior_coefficients()
        self.assertEqual(coefficients.shape, torch.Size([self.order]))
        return

    def test_coefficients_shape_1d(self):
        mercer_gp = MercerGP(
            self.basis, self.eigenvalues, self.order, self.kernel
        )
        test_inputs = torch.linspace(0, 1, 20).unsqueeze(1)
        test_outputs = torch.linspace(0, 1, 20).unsqueeze(1)

        mercer_gp.add_data(test_inputs, test_outputs)

        coefficients = mercer_gp._calculate_posterior_coefficients()
        self.assertEqual(coefficients.shape, torch.Size([self.order]))
        return


if __name__ == "__main__":
    unittest.main()
