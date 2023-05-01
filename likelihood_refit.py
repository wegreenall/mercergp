# likelihood_refit.py
import torch

# import math
import torch.distributions as D

# from termcolor import colored
# from framework.utils import print_dict
# import termplot as tplot

# from typing import List
from mercergp.eigenvalue_gen import (
    EigenvalueGenerator,
    SmoothExponentialFasshauer,
)
from mercergp.kernels import MercerKernel
from ortho.basis_functions import (
    Basis,
    # OrthonormalBasis,
    smooth_exponential_basis_fasshauer,
)

# from ortho.orthopoly import OrthogonalBasisFunction, OrthogonalPolynomial
# import matplotlib.pyplot as plt
from typing import Tuple, Callable

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(linewidth=300)


class Likelihood:
    def __init__(
        self,
        order: int,
        kernel: MercerKernel,
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
        self.kernel = kernel
        self.kernel_derivative = kernel  # will use the same basis
        self.input_sample = input_sample
        self.output_sample = output_sample
        self.eigenvalue_generator = eigenvalue_generator
        self.memoise = memoise

        # learning rates
        self.param_learning_rate = param_learning_rate
        self.sigma_learning_rate = sigma_learning_rate

        # convergence criterion
        self.epsilon = optimisation_threshold

    def fit(
        self,
        initial_noise: torch.Tensor,
        parameters: torch.Tensor,
        iter_count=10000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a dictionary containing the trained parameters.

        The noise parameter, as trained is equal to "σ^2" in the standard
        formulation of the noise variance for the Gaussian process.

        We do this because the parameter is never evaluated as σ.
        """
        converged = False
        trained_noise = initial_noise.clone().detach()
        trained_parameters = parameters.clone().detach()
        while not converged:
            # Get the gradients
            noise_gradient, parameters_gradient = self.get_gradients(
                trained_parameters
            )

            # update the parameters
            trained_noise.data -= self.sigma_learning_rate * noise_gradient

            # it may be better as a tensor of parameter values...
            trained_parameters.data -= (
                self.param_learning_rate * parameters_gradient
            )

            # having updated parameters and noise values, change on the kernel
            self.update_kernel_parameters(trained_parameters, trained_noise)

            # check the criterion
            converged = (noise_gradient < self.epsilon) and (
                parameters_gradient < self.epsilon
            ).all()

        return trained_noise, trained_parameters

    def update_kernel_parameters(
        self, parameters: torch.Tensor, noise: torch.Tensor
    ) -> None:
        """
        Updates the kernel to reflect the new values of the
        parameters and noise.
        """
        self.kernel.set_eigenvalues(self.eigenvalue_generator(parameters))
        self.kernel.set_noise(noise)
        self.kernel_derivative.set_eigenvalues(
            self.eigenvalue_generator.derivatives(parameters)
        )
        self.kernel_derivative.set_noise(noise)

    def get_gradients(
        self, parameters: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the gradient of the log-likelihood w.r.t the noise parameter
        and the parameters tensor.

        Because the calculation of the kernel inverse is ostensibly expensive,
        the kernel inverse is calculated at the top of the "computational graph"
        and passed in to the functions that will then call the TermGenerator
        to construct respective gradient terms.

        output shapes:
            sigma_grad: [1]
            params_grad: [b x 1]
        """
        # precalculates the kernel inverse for speed
        kernel_inverse = self.kernel.kernel_inverse(self.input_sample)

        # get the terms
        noise_gradient = self.noise_gradient(kernel_inverse, parameters)
        parameters_gradient = self.parameters_gradient(
            kernel_inverse, parameters
        )
        return noise_gradient, parameters_gradient

    def noise_gradient(
        self, kernel_inverse: torch.Tensor, parameters: torch.Tensor
    ):
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
        # get the terms
        sigma_gradient_term = torch.eye(self.input_sample.shape[0]).unsqueeze(
            1
        )

        # calculate the two terms comprising the gradient
        breakpoint()
        data_term = 0.5 * torch.einsum(
            "i, ij..., jk..., kl..., l  ->",
            self.output_sample,  # i
            kernel_inverse,  # ij
            sigma_gradient_term,  # jk
            kernel_inverse,  # kl
            self.output_sample,  # l
        )
        trace_term = -0.5 * torch.trace(kernel_inverse @ sigma_gradient_term)

        print("About to leave noise_gradient!")
        return data_term + trace_term

    def parameters_gradient(
        self, kernel_inverse: torch.Tensor, parameters: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the gradient of the negative log likelihood w.r.t the
        parameters.

        Code in here will calculate the appropriate terms for the gradients
        by calling the appropriate methods in the TermGenerator class.

        Returns a tensor containing the gradient information for each of the
        values in the parameter tensor.


        input_shape:
            kernel_inverse: [n x n]
            parameters: [b x 1]
        output shape:
            [b x 1]
        """
        # get the terms
        breakpoint()
        param_gradient_term = self.kernel_derivative(
            self.input_sample, self.input_sample
        )

        kernel_inverse_reshaped = kernel_inverse.unsqueeze(2).repeat(
            1, 1, parameters.shape[0]
        )

        # calculate the gradient
        breakpoint()
        data_term = 0.5 * torch.einsum(
            "i, ijb,jkb, klb, l -> b",
            self.output_sample,  # i
            kernel_inverse_reshaped,  # ijb
            param_gradient_term,  # jkb
            kernel_inverse_reshaped,  # klb
            self.output_sample,  # l
        )
        # the following should break because of incorrect shapes
        trace_term = -0.5 * torch.trace(kernel_inverse @ param_gradient_term)

        return data_term + trace_term


def optimise_explicit_gradients(
    y: torch.Tensor,
    x: torch.Tensor,
    b: torch.Tensor,
    sigma: torch.Tensor,
    objective: Callable,
    epsilon: float,
    sample_size: int,
    param_learning_rate: float = 0.0001,
    sigma_learning_rate: float = 0.0001,
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


if __name__ == "__main__":
    # data setup
    sample_size = 1000
    input_sample = D.Normal(0, 1).sample((sample_size,))
    true_noise_parameter = torch.Tensor([0.5])
    print("check input_sample")

    # generate the ground truth for the function
    def test_function(x: torch.Tensor) -> torch.Tensor:
        """
        The test function used in an iteration of Daskalakis, Dellaportas and Panos.
        """
        return (1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8).squeeze()

    output_sample = (
        test_function(input_sample)
        + D.Normal(0, true_noise_parameter).sample((sample_size,)).squeeze()
    )

    print("check output_sample")

    # kernel setup
    order = 12
    eigenvalues = torch.ones(order, 1)
    parameters = {
        "ard_parameter": torch.Tensor([1.0]),
        "precision_parameter": torch.Tensor([1.0]),
        "noise_parameter": torch.Tensor([1.0]),
    }
    basis_function = smooth_exponential_basis_fasshauer  # the basis function
    basis = Basis(basis_function, 1, order, parameters)
    kernel = MercerKernel(order, basis, eigenvalues, parameters)

    eigenvalue_generator = SmoothExponentialFasshauer(parameters)

    likelihood = Likelihood(
        order,
        kernel,
        input_sample,
        output_sample,
        eigenvalue_generator,
    )

    # initial_values for parameters:
    initial_noise = torch.Tensor([1.0])
    initial_parameters = torch.Tensor([1.0])

    # now fit the parameters
    likelihood.fit(initial_noise, initial_parameters)
