# likelihood_refit.py
import torch
import torch.distributions as D

# from tabulate import tabulate


# from framework.utils import print_dict
# import termplot as tplot

# from typing import List
from mercergp.eigenvalue_gen import (
    EigenvalueGenerator,
    SmoothExponentialFasshauer,
    eigenvalue_reshape,
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
from typing import Tuple, List
from math import prod


class TermGenerator:
    def __init__(
        self,
        input_sample: torch.Tensor,
        output_sample: torch.Tensor,
        kernel: MercerKernel,
    ):
        """
        Stores terms used in the calculation of gradients for the likelihood.

        The gradient terms for the likelihood have the following structure:
            dL/dθ = 0.5 * (y'K^{-1} dK/dθ K^{-1}y - tr(K^{-1}dK/dθ))

        Note however that since the kernel is written ΦΛΦ', and the first term
        is 1x1, we can calculate the first term as:
                    0.5 * z' δΛ/δθ

        where z = {(Φ'K^-1y)^2}_i, by a trace trick.
        Opening the WSM formula for the kernel inverse, we get:
            ΦK^-1y = Φ'y - Φ'Φ(Φ'Φ + Λ^-1)^-1Φ'y
                   = (I - Φ'Φ(Φ'Φ + Λ^-1)^-1)Φ'y
        The terms:
            - Φ'y
            - Φ'Φ
        can be pre-computed, so that the final iteration requires
        only :
            - calculation of (Φ'Φ + Λ^-1)^{-1}
            - multiplication of Φ'Φ with that matrix
            - subtraction from the identity
            - inner product of the result with Φ'y

        This class offers getter methods for precalculated terms  in order to
        speed up the calculation, and is constructed once on initialisation of
        the likelihood.
        """
        self.input_sample = input_sample
        self.output_sample = output_sample
        self.kernel = kernel
        self.order = self.kernel.order
        self.dimension = self.kernel.basis.dimension

        # initialise
        self.phi_data = None
        self.phi_y_data = None
        self.phi_phi_data = None

    @property
    def phi_y(self) -> torch.Tensor:
        """
        Returns the matrix Φ'y.
        """
        if self.phi_y_data is None:
            self.phi_y_data = (
                self.kernel.basis(self.input_sample).T @ self.output_sample
            )
        return self.phi_y_data

    @property
    def phi_phi(self) -> torch.Tensor:
        """
        Returns the matrix Φ'Φ.
        """
        if self.phi_phi_data is None:
            self.phi_phi_data = self.phi.T @ self.phi
        return self.phi_phi_data

    @property
    def phi(self) -> torch.Tensor:
        if self.phi_data is None:
            self.phi_data = self.kernel.basis(self.input_sample)

        return self.phi_data

    def get_vector_term(
        self, eigenvalues: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Returns the term:
             z = {(Φ'K^-1y)^2}_i - (Φ'K^-1Φ)_ii
        for the inverse. Then, z' δΛ/δθ is the gradient for the parameter.

        This is because tr(AD) = diag(A)'diag(D), where D is diagonal,
        as an inner product between the vectors that comprise the diagonals
        of the matrix, when D is diagonal.
        """
        # calculate the inverse
        inverse = torch.inverse(
            self.phi_phi + noise**2 * torch.diag(1 / eigenvalues)
        )
        intermediate_term = (1 / noise**2) * (
            torch.eye(self.order**self.dimension) - self.phi_phi @ inverse
        )

        data_term = ((intermediate_term @ self.phi_y) ** 2).squeeze()

        trace_term = torch.diag(intermediate_term @ self.phi_phi).squeeze()
        return 0.5 * (data_term - trace_term)

    def get_noise_term(self, eigenvalues: torch.Tensor, noise: torch.Tensor):
        """
        To get the noise term gradient, we use the fact that the
        """
        inverse = torch.inverse(
            self.phi_phi + noise**2 * torch.diag(1 / eigenvalues)
        )
        intermediate_term = (1 / noise**2) * (
            torch.eye(self.input_sample.shape[0])
            - self.phi @ inverse @ self.phi.T
        )
        data_term = (
            noise
            * torch.sum(
                (intermediate_term @ self.output_sample) ** 2
            ).squeeze()
        )
        trace_term = noise * torch.trace(intermediate_term).squeeze()
        return (data_term - trace_term).squeeze()


class Likelihood:
    def __init__(
        self,
        order: int,
        kernel: MercerKernel,
        input_sample: torch.Tensor,
        output_sample: torch.Tensor,
        eigenvalue_generator: EigenvalueGenerator,
        param_learning_rate: float = 0.00001,
        sigma_learning_rate: float = 0.00001,
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
        self.input_sample = input_sample
        self.output_sample = output_sample
        self.eigenvalue_generator = eigenvalue_generator
        self.memoise = memoise

        # learning rates
        self.param_learning_rate = param_learning_rate
        self.sigma_learning_rate = sigma_learning_rate

        # convergence criterion
        self.epsilon = optimisation_threshold
        self.term_generator = TermGenerator(
            input_sample, output_sample, kernel
        )

    def fit(
        self,
        initial_noise: torch.Tensor,
        parameters: dict,
        max_iterations=30000,
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
        if isinstance(parameters, dict):
            parameters = (parameters,)
        trained_parameters = [params.copy() for params in parameters]
        iterations = 0
        while not converged and iterations < max_iterations:
            # Get the gradients
            eigenvalues = self.eigenvalue_generator(trained_parameters)
            if (eigenvalues != eigenvalues).any():
                print("NaN detected in eigenvalues")
                breakpoint()
            trained_parameters_gradients = self.get_parameter_gradients(
                trained_parameters, eigenvalues, trained_noise
            )
            # trained_parameters_gradients.append(parameters_gradients)
            noise_gradient = self.get_noise_gradient(
                trained_noise, eigenvalues
            )
            if noise_gradient != noise_gradient:
                print("NaN detected in noise_gradient")
                breakpoint()

            # update the noise
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

            # update the the other parameters
            for params_dict, params_dict_gradients in zip(  # per dimension
                trained_parameters, trained_parameters_gradients
            ):
                for param, gradient in zip(
                    params_dict, params_dict_gradients
                ):  # per parameter
                    if (
                        params_dict_gradients[param]
                        != params_dict_gradients[param]
                    ):
                        print("NaN detected for: {}".format(param))
                        breakpoint()
                        break
                    if verbose:
                        if iterations % 1 == 0:
                            print(
                                "param gradient for: {}".format(param),
                                colored(params_dict_gradients[param], "blue"),
                                end="",
                            )
                            print(
                                "param value for: {}".format(param),
                                colored(params_dict[param], "red"),
                            )
                    params_dict[param].data += (
                        self.param_learning_rate * params_dict_gradients[param]
                    )

            # having updated parameters and noise values, change on the kernel
            # self.update_kernel_parameters(trained_parameters, trained_noise)

            # check the criterion
            """
            TEMPORARY SUBSTITUTION: WE WILL JUST USE NOISE AS THE CRITERION
            """
            convergence_criterion = torch.abs(noise_gradient)
            if convergence_criterion != convergence_criterion:
                print("NaN detected!")
                breakpoint()
            converged = convergence_criterion < self.epsilon
            print(
                "convergence criterion:",
                convergence_criterion,
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
            if iterations % 500 == 0:
                print("Iteration: {}".format(iterations))

        if converged:
            print("Converged!")
            self.converged = True
        else:
            self.converged = False
            print("Not converged: {} iterations completed.".format(iterations))
        final_eigenvalues = self.eigenvalue_generator(trained_parameters)
        print("final eigenvalues:", final_eigenvalues)
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
        eigenvalues = self.eigenvalue_generator(parameters)

        # get the terms
        noise_gradient = self.term_generator.get_noise_term(eigenvalues, noise)
        vector_term = self.term_generator.get_vector_term(eigenvalues, noise)
        parameters_gradients: dict = self.parameters_gradient(
            vector_term, parameters
        )
        return noise_gradient, parameters_gradients

    def get_noise_gradient(self, noise, eigenvalues):
        noise_gradient = self.term_generator.get_noise_term(eigenvalues, noise)
        return noise_gradient

    def get_parameter_gradients(
        self,
        parameters: dict,
        # trained_noise: torch.Tensor,
        eigenvalues: torch.Tensor,
        noise: torch.Tensor,
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
        output:
            a dictionary, with each key in "parameters"
            having a corresponding value of shape [b x 1]
            which is the gradient of the eigenvalues with
            respect to that parameter.
        """
        vector_term = self.term_generator.get_vector_term(eigenvalues, noise)

        # parameter_gradients is a dictionary of the same keys as parameters
        parameter_gradients = [
            params_dict.copy() for params_dict in parameters
        ]

        eigenvalue_derivatives = self.eigenvalue_generator.derivatives(
            parameters
        )  # the dictionary
        # For each of the parameters, take the gradient given by the eigenvalue
        # generator and get the vector of values from the term_generator.
        # Then, for each of the parameters, inner-product the term generator
        # vector

        """
        In multiple dimensions, we have to construct the vector of gradients
        by taking z @ dλ/dθ for each of the parameters. 
        """

        # d λ / d θ is d(λ_1, ..., λ_n) / dθ = (dλ_1/dθ, ..., dλ_n/dθ)
        # however, λ_i is the product of (λ_i1, ..., λ_id)
        # so dλ_i/dθ = dλ_i1/dθ λ_i2 ... λ_id
        # if the parameter is to be found in the first dimension
        eigenvalue_components = [
            self.eigenvalue_generator(params) for params in parameters
        ]
        eigenvalue_components_derivatives = [
            self.eigenvalue_generator.derivatives(params)[0]
            for params in parameters
        ]
        for d in range(self.kernel.basis.dimension):
            params_dict = parameters[d]
            # eigenvalue_derivs_dict = eigenvalue_derivatives[d]
            parameter_gradients_dict = parameter_gradients[d]

            # per dimension
            for param in params_dict:
                # get the corresponding eigenvalue stuff out
                # eigenvalue_derivative_vector = eigenvalue_derivs_dict[param]
                eigenvalue_derivative_vector = (
                    self._get_eigenvalue_derivative_vector(
                        eigenvalue_components,
                        eigenvalue_components_derivatives,
                        param,
                        d,
                    )
                )

                parameter_gradients_dict[param] = (
                    vector_term @ eigenvalue_derivative_vector
                )

        return parameter_gradients

    def _get_eigenvalue_derivative_vector(
        self,
        eigenvalue_components,
        eigenvalue_components_derivatives,
        param: str,
        dimension: int,
    ) -> torch.Tensor:
        """
        Returns the vector of eigenvalue derivatives for the dth dimension.
        """
        derivs_set = [eigenvalue_components_derivatives[dimension][param]]
        eigens_beginning = eigenvalue_components[:dimension]
        eigens_end = eigenvalue_components[dimension + 1 :]
        eigens_for_reshape = torch.vstack(
            eigens_beginning + derivs_set + eigens_end
        ).t()
        product_eigens = eigenvalue_reshape(eigens_for_reshape)
        eigenvalue_derivative_vector = torch.reshape(
            product_eigens,
            (self.order**self.kernel.basis.dimension,),
        )
        return eigenvalue_derivative_vector

    # def noise_gradient(
    # self,
    # kernel_inverse: torch.Tensor,
    # parameters: dict,
    # noise: torch.Tensor,
    # ) -> torch.Tensor:
    # """
    # Returns the gradient of the log-likelihood w.r.t the noise parameter.

    # Code in here will calculate the appropriate terms for the gradients
    # by calling the appropriate methods in the TermGenerator class.

    # Returns a tensor scalar containing the gradient information for
    # the noise parameter.

    # The key difference between this and the param_gradient function
    # is that in there the corresponding einsum must take into account the
    # extended shape of the parameters Tensor.

    # Because the kernel inverse is common to all terms, we precompute this
    # and pass it as an argument, for efficiency.
    # """
    # # get the terms
    # sigma_gradient_term = 2 * noise * torch.eye(self.input_sample.shape[0])

    # data_term = 0.5 * torch.einsum(
    # "i, ij..., jk..., kl..., l  ->",
    # self.output_sample,  # i
    # kernel_inverse,  # ij
    # sigma_gradient_term,
    # kernel_inverse,  # kl
    # self.output_sample,  # l
    # )
    # trace_term = 0.5 * torch.trace(kernel_inverse @ sigma_gradient_term)

    # return (data_term - trace_term).squeeze()  # the whole noise gradient


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
    # data = tabulate(data, headers=headers_list, tablefmt="grid")

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
