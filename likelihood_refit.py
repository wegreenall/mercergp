# likelihood_refit.py
import torch
import math
import torch.distributions as D
import math
from mercergp.eigenvalue_gen import EigenvalueGenerator
from mercergp.kernels import MercerKernel
from ortho.basis_functions import Basis, OrthonormalBasis
from ortho.orthopoly import OrthogonalBasisFunction, OrthogonalPolynomial
import matplotlib.pyplot as plt
from typing import Tuple, Callable

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(linewidth=300)
from termcolor import colored
from framework.utils import print_dict
import termplot as tplot

from typing import List


class TermGenerator:
    def __init__(
        self,
        input_sample: torch.Tensor,
        eigenvalue_generator: EigenvalueGenerator,
        kernel: MercerKernel,
    ):
        """
        This class contains methods for calculating the various terms in the
        likelihood. This will allow me to separate the terms, their memoisation,
        etc. from the process of actually fitting the likelihood.
        From Refactoring: This is like having done "Inline Class".

        Since the term generator is only useful to the likelihood, I
        will have it instantiated during Likelihood initialisation.


        The gradient is calculated as:
            d ln p/dσ^2 = 1/2 y' K^{-1} [δK/δθ] K^{-1} y - 1/2 tr(K^{-1} δK/δσ^2)

        These can be acquired by the following
            - inner_param_term -> K^{-1} [δK/δθ] K^{-1}
            - inner_sigma_term -> K^{-1} [δK/δσ^2] K^{-1}
            - trace_sigma_term -> tr(K^{-1} [δK/δσ^2])
            - trace_param_term -> tr(K^{-1} [ δK/δθ ])

        """
        self.input_sample = input_sample
        self.eigenvalue_generator = eigenvalue_generator
        self.kernel = kernel

    def inner_param_term(self, kernel_inverse: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """
        Returns the term K^{-1} [δK/δθ] K^{-1}

        Because this is constructing the grad for the parameter tensor,
        it uses the shape of the parameter vector to get the right shape for the 
        extension to the kernel inverse.

        input shapes:
            kernel_inverse: [n x n]
            parameters: [b x 1]

        output shape:
            [n x n x b]
        """
        kernel_inverse_reshaped = kernel_inverse.unsqueeze(2).repeat(1, 1, parameters.shape[0])
        param_grad = self.param_grad(parameters)
        return torch.einsum("ij...,jk...,kl... -> il...",
                            kernel_inverse_reshaped,
                            param_grad,
                            kernel_inverse_reshaped)

    def inner_sigma_term(self, kernel_inverse: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """
        Returns the term K^{-1} δK/δσ^2 K^{-1}

        input shapes:
            kernel_inverse: [n x n]
            parameters: [b x 1]

        output shape:
            [n x n]
        """
        sigma_grad = self.sigma_grad(parameters)
        return torch.einsum("ij,jk,kl -> il", kernel_inverse, sigma_grad, kernel_inverse)

    def trace_param_term(self, kernel_inverse: torch.Tensor, ) -> torch.Tensor:
        """
        Returns the term tr(K^{-1} [δK/δθ])
        """
        kernel_inverse_reshaped = kernel_inverse.unsqueeze(2).repeat(1, 1, parameters.shape[0])
        param_grad = self.param_grad(parameters)
        return torch.einsum("iib, b -> b", kernel_inverse_reshaped, param_grad)


    def trace_sigma_term(self, kernel_inverse: torch.Tensor,  parameters: torch.Tensor) -> torch.Tensor:
        """
        Returns the term tr(K^{-1} δK/δσ^2)

        input shapes:
            kernel_inverse: [n x n]
            parameters: [b x 1]

        output shape:
            [n x n]
        """
        sigma_grad = self.sigma_grad(parameters)
        return torch.trace(kernel_inverse @ sigma_grad)


    def param_grad(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Returns the matrix of \frac{δK}{δθ}
        """
        return

    def sigma_grad(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Returns the matrix of \frac{δK}{δσ^2}
        """
        return


class Likelihood:
    def __init__(
        self,
        order: int,
        basis: Basis,
        input_sample: torch.Tensor,
        output_sample: torch.Tensor,
        eigenvalue_generator: EigenvalueGenerator,
        param_learning_rate: float = 0.0001,
        sigma_learning_rate: float = 0.0001,
        memoise=True,
        optimisation_threshold=0.0001,
    ):
        """
        Initialises the Likelihood class.

        To use this, construct an instance of a torch.optim.Optimizer;
        register the parameters that are to be optimised, and pass it when
        instantiating this class.

        Parameters:
            order: The bandwidth of the kernel/no. of basis functions.
            basis: a Basis object that allows for construction of the various
                   matrices.
            input_sample: The sample of data X.
            output_sample: The (output) sample of data Y.
            mc_sample_size=10000:
        """
        # hyperparameters
        self.order = order
        self.basis = basis
        self.input_sample = input_sample
        self.output_sample = output_sample
        self.eigenvalue_generator = eigenvalue_generator
        self.memoise = memoise

        # learning rates
        self.param_learning_rate = param_learning_rate
        self.sigma_learning_rate = sigma_learning_rate

        # convergence criterion
        self.epsilon = optimisation_threshold

        # term generator
        self.term_generator = TermGenerator(
            basis, input_sample, output_sample, eigenvalue_generator
        )

    def fit(
        self,
        initial_noise: torch.Tensor,
        parameters: torch.Tensor,
        iter_count=10000,
    ) -> List[torch.Tensor]:
        """
        Returns a dictionary containing the trained parameters.

        The noise parameter, as trained is equal to "σ^2" in the standard
        formulation of the noise variance for the Gaussian process.

        We do this because the parameter is never evaluated as σ.
        """
        converged = False
        noise_parameter = initial_noise.clone().detach().requires_grad_(True)
        while not converged:
            # Get the gradients
            noise_gradient, parameters_gradient = self.get_gradients(
                parameters
            )

            # update the parameters
            noise_parameter.data -= self.sigma_learning_rate * noise_gradient

            # it may be better as a tensor of parameter values...
            parameters.data -= self.param_learning_rate * parameters_gradient

            # check the criterion
            converged = (noise_gradient < self.epsilon) and (
                parameters_gradient < self.epsilon
            ).all()

        new_parameters: dict = dict()
        return new_parameters

    def get_gradients(self) -> [torch.Tensor, torch.Tensor]:
        """
        Returns the gradient of the log-likelihood w.r.t the noise parameter.

        Code in here will calculate the appropriate terms for the gradients
        by calling the appropriate methods in the TermGenerator class.

        Returns a tensor scalar containing the gradient information for
        the noise parameter.

        The key difference between this and the param_gradient function
        is that in there the corresponding einsum must take into account the
        extended shape of the parameters Tensor.

        output shapes:
            sigma_grad: [1]
            params_grad: [b x 1]
        """
        kernel_inv = self.kernel.kernel_inverse(self.input_points)
        data_sigma_term = 

        # get the data term
        data_term_parameters = torch.einsum(
            "i, ij, jk, kl, l",
            self.output_sample,
            kernel_inv,
            ,
            kernel_inv,
            self.output_sample,
        )
        return sigma_grad, params_grad

    def noise_gradient(self, parameters: torch.Tensor):
        """
        Returns the gradient of the log-likelihood w.r.t the noise parameter.

        Code in here will calculate the appropriate terms for the gradients
        by calling the appropriate methods in the TermGenerator class.

        Returns a tensor scalar containing the gradient information for
        the noise parameter.

        The key difference between this and the param_gradient function
        is that in there the corresponding einsum must take into account the
        extended shape of the parameters Tensor.

        Because the kernel inverse is common to all terms, we precompute this 
        and pass it as an argument, for efficiency.
        """
        kernel_inverse = self.kernel.kernel_inverse(self.input_sample)

        # get the terms
        inner_sigma_term = self.term_generator.inner_sigma_term(kernel_inverse, parameters)
        trace_sigma_term = self.term_generator.trace_sigma_term(
            parameters
        )  # tr(K^-1 * δK/δσ^2)

        # calculate the gradient
        data_term = 0.5 * torch.einsum(
            "i, ij, j ->",
            self.output_sample,
            inner_sigma_term,
           self.output_sample,
        )
        trace_term = -0.5 * trace_sigma_term
        return data_term + trace_term

    def parameters_gradient(self, parameters: torch.Tensor) -> torch.Tensor:
        """
        Returns the gradient of the negative log likelihood w.r.t the
        parameters.

        Code in here will calculate the appropriate terms for the gradients
        by calling the appropriate methods in the TermGenerator class.

        Returns a tensor containing the gradient information for each of the
        values in the parameter tensor.

        Return shape: len(parameters)
        """
        # get the K inverse term
        kernel_inverse = self.kernel.kernel_inverse()

        # get the terms
        inner_param_term = self.term_generator.inner_param_term(kernel_inverse, parameters)
        trace_param_term = self.term_generator.trace_param_term(parameters)

        # calculate the gradient
        data_term = 0.5 * torch.einsum(
            "i ij..., j -> ...",
            self.output_sample,
            inner_param_term,
            self.output_sample,
        )
        trace_term = -0.5 * trace_param_term
        assert data_term.shape == trace_term.shape
        assert data_term.shape == parameters.shape
        return data_term + trace_term


def optimise_explicit_gradients(
    y: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    sigma: torch.Tensor,
    objective: Callable,
    epsilon: float,
    sample_size: int,
    param_learning_rate: torch.Tensor = 0.0001,
    sigma_learning_rate: torch.Tensor = 0.0001,
):
    """
    Optimises the likelihood w.r.t sigma, b using explicit gradients.

    It does this by waiting for a criterion value to
    be less than epsilon. The gradients are calculated explicitly.
    The gradients are handled in functions:
        - determinant_gradients
        - inverse_kernel_gradients

    See their signatures and bodies for more information.
    """
    pass


if __name__ == "__main__":
    # the program begins here

    # functions that currently exist:
    # [F] optimise_explicit_gradients
    # [F] determinant_gradients ->
    # gets gradients for sigma and b from the det term
    # [F] inverse_kernel_gradients ->
    # gets gradients for sigma and b from the kernel inverse term
    # [F] kernel ->  returns the Gram matrix of the kernel at x,  x'
    # [F] evaluate_negative_log_likelihood -> evaluates the Gaussian log
    # likelihood.
    # [F] build_ground_truth -> (input_sample, noise_distribution, true_function,
    #                            sample_size)
    # [F] run_experiment -> returns
    pass
