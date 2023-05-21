import torch
import torch.distributions as D
from ortho.basis_functions import (
    Basis,
    OrthonormalBasis,
    smooth_exponential_eigenvalues_fasshauer,
    smooth_exponential_basis_fasshauer,
)
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from ortho.orthopoly import (
    OrthogonalBasisFunction,
    SymmetricOrthonormalPolynomial,
)
from ortho.measure import MaximalEntropyDensity
from mercergp.likelihood_refit import Likelihood, TermGenerator
from mercergp.kernels import MercerKernel

# from mercergp.likelihood_refit import TermGenerator

import unittest


class TestLikelihoodRefit(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        # first, get the data
        sample_size = 1000
        self.input_sample = D.Normal(0, 4).sample((sample_size,))
        true_noise_parameter = torch.Tensor([0.3])

        # generate the ground truth for the function
        def test_function(x: torch.Tensor) -> torch.Tensor:
            """
            The test function used in an iteration of Daskalakis, Dellaportas and
            Panos.
            """
            return (
                1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8
            ).squeeze()

        self.output_sample = (
            test_function(self.input_sample)
            + D.Normal(0, true_noise_parameter)
            .sample((sample_size,))
            .squeeze()
        )

        order = 10
        eigenvalues = torch.ones(order, 1)
        self.noise = torch.Tensor([1.0])
        self.parameters = {
            "ard_parameter": torch.Tensor([1.0]),
            "precision_parameter": torch.Tensor([1.0]),
            "noise_parameter": self.noise,
            "variance_parameter": torch.Tensor([1.0]),
        }
        basis_function = (
            smooth_exponential_basis_fasshauer  # the basis function
        )
        self.basis = Basis(basis_function, 1, order, self.parameters)
        self.kernel = MercerKernel(
            order, self.basis, eigenvalues, self.parameters
        )

        # eigenvalues and eigenvalue derivatives
        self.eigenvalue_generator = SmoothExponentialFasshauer(order)
        self.eigenvalues = self.eigenvalue_generator(self.parameters)
        self.eigenvalue_derivatives = (
            self.eigenvalue_generator._ard_parameter_derivative(
                self.parameters
            )
        )
        self.kernel.set_eigenvalues(self.eigenvalues)

        self.likelihood = Likelihood(
            order,
            self.kernel,
            self.input_sample,
            self.output_sample,
            self.eigenvalue_generator,
        )
        self.phi_matrix = self.basis(self.input_sample)

        # derivatives kernel
        self.derivative_kernel = MercerKernel(
            order, self.basis, eigenvalues, self.parameters
        )
        self.derivative_kernel.set_eigenvalues(self.eigenvalue_derivatives)
        self.parameter_gradient_term = self.derivative_kernel(
            self.input_sample, self.input_sample
        )

        # kernel inverse
        self.kernel_inverse = self.kernel.kernel_inverse(self.input_sample)

        self.term_generator = TermGenerator(
            self.input_sample, self.output_sample, self.kernel
        )

    def test_parameter_gradient(self):
        """
        To test, get the likelihood to generate the gradient.
        Then do it by hand in the "fast" way that I found.

        If they are the same, replace the likelihood method.
        """
        true_vector_term = self.term_generator.get_vector_term(
            self.eigenvalues, self.noise
        )
        true_term = self.likelihood.parameters_gradient(
            true_vector_term, self.parameters
        )["ard_parameter"]
        _, true_term = self.likelihood.get_gradients(
            self.parameters, self.noise
        )
        true_term = true_term["ard_parameter"]

        # get the derivative kernel matrix term: δK/δθ
        # true terms
        data_term = 0.5 * torch.einsum(
            "i, ij..., jk..., kl..., l  ->",
            self.output_sample,  # i
            self.kernel_inverse,  # ij
            self.parameter_gradient_term,  # jk
            self.kernel_inverse,  # kl
            self.output_sample,  # l
        )

        # trace term
        yy = torch.einsum("i,j->ij", self.output_sample, self.output_sample)
        kernel_term = (  # checking if this is the same as the gradient term...
            self.phi_matrix @ torch.diag(self.eigenvalue_derivatives)
        )
        test_data_term = torch.trace(
            0.5
            * torch.einsum(
                "hi, ij, jk, kl, lm -> hm",
                self.phi_matrix.T,  # hi
                self.kernel_inverse,  # ij
                yy,  # jk
                self.kernel_inverse,  # kl
                kernel_term,  # kl
            )
        )
        # second trace term approach
        term_1 = torch.einsum(
            "hi, ij, jk, kl, lm -> hm",
            self.phi_matrix.T,  # hi
            self.kernel_inverse,  # ij
            yy,  # jk
            self.kernel_inverse,  # kl
            self.phi_matrix,  # hm
        )

        test_data_term_2 = (
            0.5 * torch.diag(term_1) @ self.eigenvalue_derivatives
        )
        self.assertTrue(torch.allclose(data_term, test_data_term_2))

        # breakpoint()
        # # now do it by hand
        phikyvector = self.PhiKyVector()
        phikyvectorsquared = self.PhiKyVector() ** 2
        term_3 = torch.einsum("i,j->ij", phikyvector, phikyvector)
        term_4 = torch.diag(term_3)
        self.assertTrue(term_4.shape == (torch.diag(term_1).shape))
        self.assertTrue(torch.allclose(torch.diag(term_1), term_4))
        test_data_term_3 = 0.5 * term_4 @ self.eigenvalue_derivatives
        self.assertTrue(data_term, test_data_term_3)

        test_data_term_4 = (
            0.5 * phikyvectorsquared @ self.eigenvalue_derivatives
        )
        self.assertTrue(data_term, test_data_term_4)

        # trace_term by hand
        trace_term = (
            0.5
            * torch.diag(
                self.phi_matrix.T @ self.kernel_inverse @ self.phi_matrix
            )
            @ self.eigenvalue_derivatives
        )

        # breakpoint()
        # vector term from term generator
        vector_term = self.term_generator.get_vector_term(
            self.eigenvalues, self.noise
        )
        test_data_term_final = vector_term @ self.eigenvalue_derivatives

        # true term is the full thing with trace term subtracted etc.
        self.assertTrue(
            torch.allclose(
                (data_term - trace_term),
                test_data_term_final,
            )
        )

        # compare the results from the true likelihood, with the results
        # from the term generator
        self.assertTrue(torch.allclose(true_term, test_data_term_final))

    def PhiKyVector(self):
        return self.phi_matrix.T @ self.kernel_inverse @ self.output_sample

    def PhiKPhiVector(self):
        return torch.diag(
            self.phi_matrix.T @ self.kernel_inverse @ self.phi_matrix
        )

    def test_noise_gradient(self):
        # get the noise gradient from the likelihood
        # compare it to that build from the term_vector stuff
        true_term = self.likelihood.noise_gradient(
            self.kernel_inverse, self.parameters, self.noise
        )
        term_generator_version, _ = self.likelihood.get_gradients(
            self.parameters, self.noise
        )
        self.assertTrue(torch.allclose(term_generator_version, true_term))

    def test_noise_gradient_shape(self):
        true_term = self.likelihood.noise_gradient(
            self.kernel_inverse, self.parameters, self.noise
        )
        term_generator_version, _ = self.likelihood.get_gradients(
            self.parameters, self.noise
        )
        # self.assertTrue(torch.allclose(term_generator_version, true_term))
        breakpoint()
        self.assertTrue(true_term.shape == term_generator_version.shape)


class TestTermGenerator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

        # first, get the data
        sample_size = 1000
        self.input_sample = D.Normal(0, 4).sample((sample_size,))
        true_noise_parameter = torch.Tensor([0.3])

        # generate the ground truth for the function
        def test_function(x: torch.Tensor) -> torch.Tensor:
            """
            The test function used in an iteration of Daskalakis, Dellaportas and
            Panos.
            """
            return (
                1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8
            ).squeeze()

        self.output_sample = (
            test_function(self.input_sample)
            + D.Normal(0, true_noise_parameter)
            .sample((sample_size,))
            .squeeze()
        )

        self.order = 10
        self.noise = torch.Tensor([2.0])
        self.parameters = {
            "ard_parameter": torch.Tensor([1.0]),
            "precision_parameter": torch.Tensor([1.0]),
            "noise_parameter": self.noise,
            "variance_parameter": torch.Tensor([1.0]),
        }
        self.eigenvalue_generator = SmoothExponentialFasshauer(self.order)
        self.eigenvalues = self.eigenvalue_generator(self.parameters)
        basis_function = (
            smooth_exponential_basis_fasshauer  # the basis function
        )
        self.basis = Basis(basis_function, 1, self.order, self.parameters)
        self.kernel = MercerKernel(
            self.order, self.basis, self.eigenvalues, self.parameters
        )

        self.term_generator = TermGenerator(
            self.input_sample, self.output_sample, self.kernel
        )
        self.phi_matrix = self.basis(self.input_sample)

    def test_phi_y_shape(self):
        """
        Tests that the shape for the phi_y term property is correct.
        """
        phi_y = self.term_generator.phi_y
        self.assertTrue(phi_y.shape == torch.Size([self.order]))

    def test_phi_phi_shape(self):
        """
        Tests that the shape for the phi_phi term property is correct.
        """
        phi_phi = self.term_generator.phi_phi
        self.assertTrue(phi_phi.shape == torch.Size([self.order, self.order]))

    def test_get_vector_term_shape(self):
        """
        Tests that the shape for the vector term property is correct.
        """
        vector_term = self.term_generator.get_vector_term(
            self.eigenvalues, self.noise
        )
        self.assertTrue(vector_term.shape == torch.Size([self.order]))

    def test_get_vector_term_vals(self):
        """
        Tests that the values for the vector term property are correct.
        """
        # the true term ( z = {(Φ'K^-1y)^2}_i - (Φ'K^-1Φ)_ii )
        vector_term = self.term_generator.get_vector_term(
            self.eigenvalues, self.noise
        )
        # {(Φ'K^-1y)^2}_i
        handmade_vector_term_1 = (
            self.phi_matrix.T
            @ self.kernel.kernel_inverse(self.input_sample)
            @ self.output_sample
        ) ** 2

        # {(Φ'K^-1y)^2}_i
        handmade_vector_term_2 = torch.diag(
            self.phi_matrix.T
            @ self.kernel.kernel_inverse(self.input_sample)
            @ self.phi_matrix
        )
        self.assertTrue(
            torch.allclose(
                vector_term,
                0.5 * (handmade_vector_term_1 - handmade_vector_term_2),
            )
        )

    def test_get_noise_term_shape(self):
        """
        Tests that the shape for the noise term property is correct.
        """
        noise_term = self.term_generator.get_noise_term(
            self.eigenvalues, self.noise
        )
        self.assertTrue(noise_term.shape == torch.Size([1]))


if __name__ == "__main__":
    unittest.main()
