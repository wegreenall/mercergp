# likelihood_refit.py
# breakpoint()
import torch
import torch.distributions as D
from tabulate import tabulate


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
from termcolor import colored

torch.autograd.set_detect_anomaly(True)
torch.set_printoptions(linewidth=300)
import matplotlib.pyplot as plt


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
        optimisation_threshold=0.000001,
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
        parameters: dict,
        max_iterations=10000,
        verbose=True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns a dictionary containing the trained parameters.

        The noise parameter, as trained is equal to "σ^2" in the standard
        formulation of the noise variance for the Gaussian process.

        We do this because the parameter is never evaluated as σ.
        """
        converged = False
        trained_noise = initial_noise.clone().detach()
        trained_parameters = parameters.copy()
        iterations = 0
        while not converged and iterations < max_iterations:
            # Get the gradients
            noise_gradient, parameters_gradients = self.get_gradients(
                trained_parameters, trained_noise
            )

            # update the parameters
            trained_noise.data += self.sigma_learning_rate * noise_gradient
            if verbose:
                if iterations % 1 == 0:
                    print("Iteration: {}".format(iterations))
                    print("Order:", self.order)
                    print(
                        "Noise gradient:",
                        colored(noise_gradient, "green"),
                        end="",
                    )
                    print(
                        "Noise value",
                        colored(trained_noise.data**2, "magenta"),
                    )

            # it may be better as a tensor of parameter values...
            for param in trained_parameters:
                if verbose:
                    if iterations % 100 == 0:
                        print(
                            "param gradient for: {}".format(param),
                            colored(parameters_gradients[param], "blue"),
                            end="",
                        )
                        print(
                            "param value for: {}".format(param),
                            colored(parameters[param], "red"),
                        )
                trained_parameters[param].data += (
                    self.param_learning_rate * parameters_gradients[param]
                )

            # print("\n")

            if verbose:
                pass
                # tabulated_data = get_tabulated_data(
                # trained_noise,
                # trained_parameters,
                # noise_gradient,
                # parameters_gradients,
                # )
                # print(tabulated_data)
            # having updated parameters and noise values, change on the kernel
            self.update_kernel_parameters(trained_parameters, trained_noise)

            # check the criterion
            """
            TEMPORARY SUBSTITUTION: WE WILL JUST USE NOISE AS THE CRITERION
            """
            converged = (
                torch.abs(noise_gradient) * self.sigma_learning_rate * 100
                < self.epsilon
            )
            # converged = (torch.abs(noise_gradient) < self.epsilon) and (
            # torch.Tensor(
            # [
            # torch.abs(gradient) < self.epsilon
            # for gradient in parameters_gradients.values()
            # ]
            # )
            # ).all()
            iterations += 1
            if iterations % 100 == 0:
                print("Iteration: {}".format(iterations))

        if converged:
            print("Converged!")
            self.converged = True
        else:
            self.converged = False
            print("Not converged: {} iterations completed.".format(iterations))
        final_eigenvalues = self.eigenvalue_generator(trained_parameters)
        print("final eigenvalues:", final_eigenvalues)
        # breakpoint()
        experiment_order = sum(
            torch.where(
                final_eigenvalues
                > (trained_noise / self.input_sample.shape[0]),
                torch.ones(final_eigenvalues.shape),
                torch.zeros(final_eigenvalues.shape),
            )
        )
        print("estimated optimal order:", experiment_order)
        return trained_noise, trained_parameters

    def update_kernel_parameters(
        self, parameters: dict, noise: torch.Tensor
    ) -> None:
        """
        Updates the kernel to reflect the new values of the
        parameters and noise.
        """
        self.kernel.set_eigenvalues(self.eigenvalue_generator(parameters))
        self.kernel.set_noise(noise)
        # self.kernel_derivative.set_eigenvalues(
        # self.eigenvalue_generator.derivatives(parameters)
        # )
        # self.kernel_derivative.set_noise(noise)

    def get_gradients(
        self, parameters: dict, noise: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
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

        kernel_inverse: torch.Tensor = self.kernel.kernel_inverse(
            self.input_sample
        )

        # get the terms
        noise_gradient: torch.Tensor = self.noise_gradient(
            kernel_inverse, parameters, noise
        )
        parameters_gradients: dict = self.parameters_gradient(
            kernel_inverse, parameters
        )
        return noise_gradient, parameters_gradients

    def noise_gradient(
        self,
        kernel_inverse: torch.Tensor,
        parameters: dict,
        noise: torch.Tensor,
    ) -> torch.Tensor:
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
        sigma_gradient_term = 2 * noise * torch.eye(self.input_sample.shape[0])

        # print(
        # "kernel inverse, is it chacning with noise changes?",
        # self.kernel.kernel_inverse(self.input_sample),
        # )
        # calculate the two terms comprising the gradient
        data_term = 0.5 * torch.einsum(
            "i, ij..., jk..., kl..., l  ->",
            self.output_sample,  # i
            kernel_inverse,  # ij
            sigma_gradient_term,  # jk
            kernel_inverse,  # kl
            self.output_sample,  # l
        )
        trace_term = 0.5 * torch.trace(kernel_inverse @ sigma_gradient_term)

        return data_term - trace_term

    def parameters_gradient(
        self, kernel_inverse: torch.Tensor, parameters: dict
    ) -> dict:
        """
        Returns the gradient of the negative log likelihood w.r.t the
        parameters.

        Code in here will calculate the appropriate terms for the gradients
        by calling the appropriate methods in the TermGenerator class.

        Returns a tensor containing the gradient information for each of the
        values in the parameter tensor.

        The gradient of the likelihood with respect to the parameters θ is:
              dL/dθ = 1/2 y' K^-1 dK/dθ K^-1 y - 1/2 Tr(K^-1 dK/dθ)

        where dK/dθ is the matrix of derivatives of the kernel with respect to
        the given parameter. The Mercer form of the kernel means that
        this is essentially the same as the kernel, with eigenvalues set as
        the derivative of the eigenvalues:
                             dK/dθ = Φ \hat{Λ}' Φ'

        where \hat{Λ} = diag(dλ_1/dΘ, ..., dλ_n/dθ) and Φ is the matrix of
        eigenfunction evaluations.

        That is, generating the matrix derivative term dK/dθ is equivalent to
        evaluating the kernel with eigenvalue vectors represented by the
        derivatives of the eigenvalues with respect to the parameter.

        input_shape:
            kernel_inverse: [n x n]
            parameters: [b x 1]
        output shape:
            [b x 1]
        """
        # parameter_gradients is a dictionary of the same keys as parameters
        parameter_gradients = parameters.copy()

        eigenvalue_derivatives = self.eigenvalue_generator.derivatives(
            parameters
        )

        # get the terms
        for param in parameters:
            self.kernel.set_eigenvalues(eigenvalue_derivatives[param])
            kernel_gradient_term: dict = self.kernel(
                self.input_sample, self.input_sample
            )

            # calculate the gradient
            data_term = 0.5 * torch.einsum(
                "i, ij, jk, kl, l -> ",
                self.output_sample,  # i
                kernel_inverse,  # ijb
                kernel_gradient_term,  # jkb
                kernel_inverse,  # klb
                self.output_sample,  # l
            )

            trace_term = 0.5 * torch.trace(
                kernel_inverse @ kernel_gradient_term
            )
            parameter_gradients[param] = data_term - trace_term

        return parameter_gradients


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


def get_tabulated_data(
    trained_noise: torch.Tensor,
    trained_parameters: dict,
    noise_gradient: torch.Tensor,
    parameters_gradients: dict,
):
    """
    Given the parameters and their gradients (as well as corresponding values
    for the noise parameter), tabulates the information in a presentable way
    for the repetition in the likelihood fitting.
    """
    data = [
        "_",
        *trained_parameters.values(),
        "_",
        *parameters_gradients.values(),
    ]
    headers_list = [
        "Values:",
        *trained_parameters.keys(),
        "Gradients:",
        *parameters_gradients.keys(),
    ]
    data = tabulate(data, headers=headers_list, tablefmt="grid")

    # data = parameters_gradients.copy()
    # headers = [
    # "Values:",
    # *trained_parameters.keys(),
    # ]
    # data2 =
    return data


if __name__ == "__main__":
    plot = True
    # data setup
    sample_size = 1000
    input_sample = D.Normal(0, 4).sample((sample_size,))
    true_noise_parameter = torch.Tensor([0.3])
    print("check input_sample")

    # generate the ground truth for the function
    def test_function(x: torch.Tensor) -> torch.Tensor:
        """
        The test function used in an iteration of Daskalakis, Dellaportas and
        Panos.
        """
        return (1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8).squeeze()

    output_sample = (
        test_function(input_sample)
        + D.Normal(0, true_noise_parameter).sample((sample_size,)).squeeze()
    )

    print("check output_sample")

    # kernel setup
    order = 7
    eigenvalues = torch.ones(order, 1)
    parameters = {
        "ard_parameter": torch.Tensor([1.0]),
        "precision_parameter": torch.Tensor([1.0]),
        "noise_parameter": torch.Tensor([0.5]),
        "variance_parameter": torch.Tensor([1.0]),
    }
    basis_function = smooth_exponential_basis_fasshauer  # the basis function
    basis = Basis(basis_function, 1, order, parameters)
    kernel = MercerKernel(order, basis, eigenvalues, parameters)

    eigenvalue_generator = SmoothExponentialFasshauer(order)

    likelihood = Likelihood(
        order,
        kernel,
        input_sample,
        output_sample,
        eigenvalue_generator,
    )

    if plot:
        x_axis = torch.linspace(-10, 10, 1000)
        # plot the function
        plt.plot(x_axis, test_function(x_axis), label="true function")
        plt.scatter(
            input_sample.numpy(),
            output_sample.numpy(),
            label="sampled data",
            color="black",
        )
        plt.legend()
        plt.show()
    # initial_values for parameters:
    initial_noise = torch.Tensor([0.5])

    # now fit the parameters
    likelihood.fit(initial_noise, parameters)
