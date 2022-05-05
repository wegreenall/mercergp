import torch
import torch.distributions as D
from ortho.basis_functions import Basis
from ortho.orthopoly import OrthogonalBasisFunction
from mercergp.likelihood import MercerLikelihood

import unittest


class TestLikelihoodMethods(unittest.TestCase):
    def setUp(self):
        # print("likelihood.py")
        torch.manual_seed(1)
        self.order = 8
        self.sample_size = 1000
        input_sample = D.Normal(0.0, 1.0).sample([self.sample_size])
        output_sample = torch.exp(input_sample)

        betas = torch.ones(2 * self.order)
        gammas = torch.ones(2 * self.order)
        gammas.requires_grad = False

        optimiser = torch.optim.SGD([gammas], lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, threshold=1e-8, factor=0.95
        )
        basis_function = OrthogonalBasisFunction(self.order, betas, gammas)
        basis = Basis(basis_function, 1, self.order)

        # fit the likelihood
        self.parameters = {
            "noise_parameter": torch.Tensor([0.1]),
            "eigenvalue_smoothness_parameter": torch.Tensor([1.0]),
            "eigenvalue_scale_parameter": torch.Tensor([1.0]),
            "shape_parameter": torch.Tensor([1.0]),
        }
        self.likelihood = MercerLikelihood(
            self.order,
            optimiser,
            scheduler,
            basis,
            input_sample,
            output_sample,
        )
        # self.likelihood.fit(self.parameters)
        pass

    def test_log_determinant(self):
        ksi = self.likelihood._ksi(self.parameters)
        ksiksi = self.likelihood._ksiksi(self.parameters, ksi)
        determinant = self.likelihood._log_determinant(self.parameters, ksiksi)
        self.assertEqual(determinant.shape, torch.Size([]))
        pass

    def test_ksiksi(self):
        ksi = self.likelihood._ksi(self.parameters)
        ksiksi = self.likelihood._ksiksi(self.parameters, ksi)
        self.assertEqual(ksiksi.shape, torch.Size([self.order, self.order]))
        pass

    def test_eigenvalues(self):
        # Get the eigenvalues and check if the are the right shape
        eigenvalues = self.likelihood._eigenvalues(self.parameters)
        # shape should be:  m x 1
        self.assertEqual(eigenvalues.shape, torch.Size([self.order]))
        pass

    def test_ksi(self):
        # Get the basis function and check if it is the right shape
        ksi = self.likelihood._ksi(self.parameters)
        self.assertEqual(ksi.shape, torch.Size([self.sample_size, self.order]))
        pass

    def test_exp_term(self):
        ksi = self.likelihood._ksi(self.parameters)
        ksiksi = self.likelihood._ksiksi(self.parameters, ksi)
        ksiksi_inverse = self.likelihood._ksiksi_inverse(
            self.parameters, ksi, ksiksi
        )
        exp_term = self.likelihood._exp_term(self.parameters, ksiksi_inverse)
        self.assertEqual(exp_term.shape, torch.Size([]))  # 1
        pass
