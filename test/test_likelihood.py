import torch
import torch.distributions as D
from ortho.basis_functions import (
    Basis,
    OrthonormalBasis,
    smooth_exponential_eigenvalues_fasshauer,
    smooth_exponential_basis_fasshauer,
)
from ortho.orthopoly import (
    OrthogonalBasisFunction,
    SymmetricOrthonormalPolynomial,
)
from ortho.measure import MaximalEntropyDensity
from mercergp.likelihood import MercerLikelihood, FavardLikelihood

# from mercergp.likelihood_refit import TermGenerator

import unittest


# class TestTermGenerator(unittest.TestCase):
# def setUp(self):
# sample_size = 1000
# input_sample = D.Normal(0.0, 1.0).sample([sample_size])
# output_sample = torch.exp(input_sample)
# dim = 1
# order = 10
# parameters = {
# "ard_parameter": torch.Tensor([1.0]),
# "precision_parameter": torch.Tensor([1.0]),
# "noise_parameter": torch.Tensor([0.1]),
# }
# basis = Basis(
# smooth_exponential_basis_fasshauer, dim, order, parameters
# )
# self.term_generator = TermGenerator(basis, input_sample, output_sample)

# def test_inv_param_grad(self):
# inv_param_grad = self.term_generator.inv_param_grad()
# self.assertEqual(inv_param_grad.shape, (self.term_generator.dim,))
# pass

# def test_inv_sigma_grad(self):
# pass

# def test_trace_sigma_term(self):
# pass

# def test_trace_param_term(self):
# pass

# def test_inv_y(self):
# pass

# def test_sigma_grad(self):
# pass


@unittest.skip("Not implemented")
class TestFavardLikelihoodMethods(unittest.TestCase):
    def setUp(self):
        # print("likelihood.py")
        torch.manual_seed(1)
        self.order = 8
        self.sample_size = 1000
        input_sample = D.Normal(0.0, 1.0).sample([self.sample_size])
        output_sample = torch.exp(input_sample)

        betas = torch.zeros(2 * self.order)
        gammas = torch.ones(2 * self.order)
        gammas.requires_grad = False

        optimiser = torch.optim.SGD([gammas], lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, threshold=1e-8, factor=0.95
        )
        basis_function = OrthogonalBasisFunction(self.order, betas, gammas)
        basis = Basis(basis_function, 1, self.order, None)

        orthopoly = SymmetricOrthonormalPolynomial(self.order, gammas)
        weight_function = MaximalEntropyDensity(
            self.order, torch.zeros(2 * self.order), gammas
        )
        basis = OrthonormalBasis(
            orthopoly, weight_function, 1, self.order, None
        )
        # fit the likelihood
        self.parameters = {
            "gammas": gammas,
            "noise_parameter": torch.Tensor([0.1]),
            "eigenvalue_smoothness_parameter": torch.Tensor([1.0]),
            "eigenvalue_scale_parameter": torch.Tensor([1.0]),
            "shape_parameter": torch.Tensor([1.0]),
        }
        self.likelihood = FavardLikelihood(
            self.order,
            optimiser,
            # scheduler,
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
        self.assertEqual(eigenvalues.shape, torch.Size([self.order]))
        pass

    def test_eigenvalues_order(self):
        eigenvalues = self.likelihood._eigenvalues(self.parameters)
        for i in range(1, self.order):
            # print("eigenvalues:", eigenvalues[i])
            self.assertTrue(eigenvalues[i] <= eigenvalues[i - 1])

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

    def test_lambdainv(self):
        lambdaterm = self.likelihood._lambdainv(self.parameters)
        self.assertEqual(
            lambdaterm.shape, torch.Size([self.order, self.order])
        )  # 1


class TestMercerLikelihoodMethods(unittest.TestCase):
    def setUp(self):
        # print("likelihood.py")
        torch.manual_seed(1)
        self.order = 8
        self.sample_size = 1000
        input_sample = D.Normal(0.0, 1.0).sample([self.sample_size])
        output_sample = torch.exp(input_sample)

        betas = torch.zeros(2 * self.order)
        gammas = torch.ones(2 * self.order)
        gammas.requires_grad = False

        optimiser = torch.optim.SGD([gammas], lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, threshold=1e-8, factor=0.95
        )
        basis_function = OrthogonalBasisFunction(self.order, betas, gammas)
        basis = Basis(basis_function, 1, self.order, None)

        orthopoly = SymmetricOrthonormalPolynomial(self.order, gammas)
        weight_function = MaximalEntropyDensity(
            self.order, torch.zeros(2 * self.order), gammas
        )
        basis = OrthonormalBasis(
            orthopoly, weight_function, 1, self.order, None
        )
        # fit the likelihood
        self.parameters = {
            "noise_parameter": torch.Tensor([0.1]),
            "ard_parameter": torch.Tensor([1.0]),
            "precision_parameter": torch.Tensor([1.0]),
        }
        self.likelihood = MercerLikelihood(
            self.order,
            optimiser,
            # scheduler,
            basis,
            input_sample,
            output_sample,
            eigenvalue_generator=lambda params: smooth_exponential_eigenvalues_fasshauer(
                self.order, params
            ),
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
        self.assertEqual(eigenvalues.shape, torch.Size([self.order]))
        pass

    def test_eigenvalues_order(self):
        eigenvalues = self.likelihood._eigenvalues(self.parameters)
        for i in range(1, self.order):
            # print("eigenvalues:", eigenvalues[i])
            self.assertTrue(eigenvalues[i] <= eigenvalues[i - 1])

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

    def test_lambdainv(self):
        lambdaterm = self.likelihood._lambdainv(self.parameters)
        self.assertEqual(
            lambdaterm.shape, torch.Size([self.order, self.order])
        )  # 1
