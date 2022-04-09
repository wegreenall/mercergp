import torch
import math

from ortho.basis_functions import (
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
    Basis,
)
from mercergp.kernels import MercerKernel
import matplotlib.pyplot as plt


class HilbertSpaceElement:
    """
    A class representing an element of  Hilbert space.
    That is, for a given basis and set of coefficients, instances of this class
    represent functions that belong to the corresponding Hilbert space.

    This is useful when producing Mercer Gaussian Processes.
    """

    def __init__(self, basis, coefficients: torch.Tensor):
        self.basis = basis
        self.coefficients = coefficients
        self.order = len(coefficients)
        return

    def __call__(self, x):
        """
        Evaluates the Hilbert Space element at the given inputs x.
        """
        return torch.inner(self.coefficients, self.basis(x)).squeeze()

    def get_order(self) -> int:
        """
        Getter method for recalling the order of the model; i.e. the bandwidth
        of the kernel whose reproducing space this is an element of.
        """
        return self.order

    def get_coefficients(self):
        """
        Getter method for recalling the coefficients that this Hilbert Space
        element this is comprised of.
        """
        return self.coefficients


class MercerGP:
    """
    A class representing a Mercer Gaussian Process.
    """

    def __init__(
        self,
        basis: Basis,
        order: int,
        dim: int,
        kernel: MercerKernel,
        mean_function=lambda x: torch.zeros(x.shape),
    ):
        """
        : param basis: a Basis class instance
        : param order:  integer describing the maximum order of the kernel
                        truncation
        :param dim: an integer describing the dimension of the model
        : param kernel: a MercerKernel instance
        : param mean_function: a callable representing the
                            prior mean function for the GP.

        Note on the mean_function callable:
        Because it is feasible that the mean function might not be expressed
        as an element of the Hilbert space, we treat it as a direct callable
        function rather than a set of coefficients for the functions in the
        same space.
        """
        self.basis = basis
        self.order = order
        self.kernel = kernel
        self.dim = dim

        # data placeholders
        self.x = torch.Tensor([])
        self.y = torch.Tensor([])

        # stored as a closure - see dosctring
        self.mean_function = mean_function
        self.posterior_coefficients = torch.zeros([self.order])
        return

    def add_data(self, x, y):
        """
        Adds observation data for the given MercerGP.

        :param x: the inputs
        :param y: the outputs
        """
        # add the inputs and alter the coefficients
        self.x = torch.cat([self.x, x])
        self.y = torch.cat([self.y, y])

        self.posterior_coefficients = self._calculate_posterior_coefficients()
        return

    def set_data(self, x, y):
        """
        Replaces observation data for the given MercerGP.

        :param x: the inputs
        :param y: the outputs
        """
        self.x = x
        self.y = y
        self.posterior_coefficients = self._calculate_posterior_coefficients()

    def get_inputs(self):
        """
        Getter method for recalling the inputs that have been passed to the
        MercerGP.
        """
        return self.x

    def get_targets(self):
        """
        Getter method for recalling the outputs that have been passed to the
        MercerGP.

        Note that this returns the raw targets, rather than processed outputs
        including the mean function m(x).
        For that, MercerGP.get_posterior_mean(x) may be more relevant.
        """
        return self.y

    def get_outputs(self):
        """
        outputs the data added to the MercerGP minus the mean function
        at the inputs, for correct construction of the coefficients.
        """
        return self.y - self.mean_function(self.get_inputs())

    def get_posterior_mean(self) -> HilbertSpaceElement:
        """
        Returns the posterior mean function.
        """
        return MercerGPSample(
            self.basis, self.posterior_coefficients, self.mean_function
        )

    def get_order(self) -> int:
        return self.order

    def gen_gp(self) -> HilbertSpaceElement:
        """
        Returns a MercerGPSample object representing the sampled Gaussian
        process. It does this by having on it the basis functions and the set
        coefficients.
        """
        # return a MercerGPSample
        return MercerGPSample(
            self.basis,
            self._get_sample_coefficients() + self.posterior_coefficients,
            self.mean_function,
        )

    def _calculate_posterior_coefficients(self) -> torch.Tensor:
        """
        Returns the non-random coefficients for the posterior mean according
        to the kernel as added to the Gaussian process, and under the data
        passed to the Mercer Gaussian process. This will be zeroes for no
        targets

        That is, these are the poeterior mean coefficients related to
        (y-m)'(K(x, x) + σ^2I)^{-1}
        """
        interim_matrix = self.kernel.get_interim_matrix_inverse(self.x)
        ksi = self.kernel.get_ksi(self.x)

        posterior_coefficients = torch.einsum(
            "jm, mn -> jn", interim_matrix, ksi.t()
        )
        these_outputs = self.get_outputs()
        result = torch.einsum(
            "i..., ji -> j", these_outputs, posterior_coefficients
        )
        return result

    def _get_sample_coefficients(self) -> torch.Tensor:
        """
        Returns random coefficients for a sample according to the kernel.

        Combined with posterior coefficients, these are used to produce a GP
        sample.
        """
        mean = torch.zeros([self.order])
        variance = self.kernel.get_interim_matrix_inverse(self.x)
        # variance = torch.diag(self.kernel.get_eigenvalues())

        if (variance != torch.abs(variance)).all():
            breakpoint()
        # variance += 0.001 * torch.eye(variance.shape[0])
        normal_rv = (
            torch.distributions.MultivariateNormal(
                loc=mean, covariance_matrix=variance
            )
            .sample([1])
            .squeeze()
        )
        return normal_rv

    def set_posterior_coefficients(self, coefficients):
        self.posterior_coefficients = coefficients


class MercerGPSample(HilbertSpaceElement):
    """
    Subclassing the HilbertSpaceElement,
    this adds a passed mean function so as to represent a GP sample function.
    """

    def __init__(self, basis, coefficients, mean_function):
        """
        Modifies the super init to store the mean function callable
        """
        super().__init__(basis, coefficients)
        self.mean_function = mean_function
        return

    def __call__(self, x):
        """
        Adds the mean function evaluation to the MercerGPSample
        """
        return super().__call__(x) + self.mean_function(x)


class HermiteMercerGPSample(MercerGPSample):
    """
    Subclassing the MercerGPSample,
    this represents specifically a sample from a MercerGP with a truncated
    smooth exponential kernel via the use of the Mercer kernel representation.
    """

    def __init__(
        self,
        coefficients,
        dim,
        params,
        mean_function,
        mean_function_derivative,
    ):
        se_basis = Basis(
            smooth_exponential_basis_fasshauer, dim, len(coefficients), params
        )

        derivative_se_basis = Basis(
            smooth_exponential_basis_fasshauer,
            dim,
            len(coefficients) + 1,
            params,
        )

        super().__init__(se_basis, coefficients, mean_function)

        # prepare the derivative function for when necessary
        # dfc_2 "grabs" the basis functions starting with order 1 rather than
        # order 0
        # breakpoint()
        dfc = (
            torch.cat((torch.Tensor([0]), self.coefficients))
            * torch.sqrt((torch.linspace(0, self.order, self.order + 1)))
            * math.sqrt(2)
        )

        self.df_second_gp = HilbertSpaceElement(derivative_se_basis, dfc)

        self.mean_function_derivative = mean_function_derivative
        return

    def derivative(self, x):
        """
        Returns the derivative of this sample, evaluated at x.

        The MercerGPSample, if using the smooth exponential basis,
        is able to produce its own derivative. This may also be true for
        the Chebyshev ones as well.

        The sample derivative is written:

            x f(x) - ∑_i (β_i + λ_i) φ_{i+1}(x) √(2i + 2)

        where f(x) is the same GP. This second term is a GP with m+1
        basis functions, and a 0 coefficient in the beginning

        Variable names here follow the notation in Daskalakis, Dellaportas
        and Panos (2020)
        Here, a, b and e correspond as follows:
            a: the precision parameter for the measure
            b: the beta parameter: = (1 + (2 \frac{e}{a})^2)^0.25
            e: the length-scale parameter
        """

        # get constant coefficients α, β
        a = self.basis.get_params()["precision_parameter"]
        b = self.basis.get_params()["ard_parameter"]
        c = torch.sqrt(a ** 2 + 2 * a * b)
        first_term_coefficient = c + a
        first_term = (
            2 * x * first_term_coefficient * (self(x) - self.mean_function(x))
        )
        second_term_coefficient = torch.sqrt(2 * c)
        second_term = second_term_coefficient * self.df_second_gp(x)
        third_term = self.mean_function_derivative(x)
        return first_term - second_term + third_term


class HermiteMercerGP(MercerGP):
    """
    A Mercer Gaussian process using a truncated smooth exponential kernel
    according to the Mercer kernel formulation.
    """

    def __init__(
        self,
        order: int,
        dim: int,
        kernel: MercerKernel,
        mean_function=lambda x: torch.zeros(x.shape),
        mean_function_derivative=lambda x: torch.zeros(x.shape),
    ):
        """
        Initialises the Hermite mercer GP by constructing the basis functions
        and the derivative of the mean function.

        :param order: the number of functions to use in the kernel; i.e.,
                      the bandwidth
        :param dim: the dimension of the model. Only really feasible with
                    relatively low numbers due to the exponential behaviour of
                    the tensor product.
        """

        se_basis = Basis(
            smooth_exponential_basis_fasshauer, dim, order, kernel.get_params()
        )

        super().__init__(se_basis, order, dim, kernel, mean_function)

        self.mean_function_derivative = mean_function_derivative
        return

    def gen_gp(self):
        """
        Returns a HermiteMercerGPSample, which is a function with random
        coefficients on the basis functions.
        """
        sample_coefficients = self._get_sample_coefficients()
        # sample_coefficients = torch.zeros(sample_coefficients.shape)
        return HermiteMercerGPSample(
            sample_coefficients + self._calculate_posterior_coefficients(),
            self.dim,
            self.kernel.get_params(),
            self.mean_function,
            self.mean_function_derivative,
        )


if __name__ == "__main__":
    """ """
    # parameters for the test
    sample_size = 300

    # build a mercer kernel
    order = 10  # degree of approximation
    dim = 1

    # set up the arguments
    l_se = torch.Tensor([[2]])
    sigma_se = torch.Tensor([3])
    sigma_e = torch.Tensor([1])
    epsilon = torch.Tensor([1])
    mercer_args = {
        "ard_parameter": l_se,
        "variance_parameter": sigma_se,
        "noise_parameter": sigma_e,
        "precision_parameter": epsilon,
    }

    eigenvalues = smooth_exponential_eigenvalues_fasshauer(order, mercer_args)
    basis = Basis(smooth_exponential_basis_fasshauer, 1, order, mercer_args)
    test_kernel = MercerKernel(order, basis, eigenvalues, mercer_args)

    # build the Mercer kernel examples
    dist = torch.distributions.Normal(loc=0, scale=epsilon)
    inputs = dist.sample([sample_size])
    basis(inputs)
    gram = test_kernel(inputs, inputs) + epsilon * torch.eye(sample_size)

    a = 1
    b = 2
    c = 3

    def data_func(x):
        return a * x ** 2 + b * x + x

    data_points = data_func(inputs) + torch.distributions.Normal(0, 1).sample(
        inputs.shape
    )

    # build a standard kernel for comparison
    # show the two kernels for comparison. They're close!
    # create pseudodata for training purposes
    mercer_gp = MercerGP(basis, order, dim, test_kernel)
    mercer_gp.add_data(inputs, data_points)

    # breakpoint()

    # test the inverse
    # inv_1 = test_kernel.kernel_inverse(inputs)
    # inv_3 = torch.inverse(test_kernel(inputs, inputs))
    test_points = torch.linspace(-2, 2, 100)  # .unsqueeze(1)
    test_sample = mercer_gp.gen_gp()  # the problem!
    test_mean = mercer_gp.get_posterior_mean()

    # GP sample
    plt.plot(
        test_points.flatten().numpy(),
        test_sample(test_points).flatten().numpy(),
    )
    # GP mean
    plt.plot(
        test_points.flatten().numpy(),
        test_mean(test_points).flatten().numpy(),
    )
    # true function
    plt.plot(
        test_points.flatten().numpy(), data_func(test_points).flatten().numpy()
    )
    # input/output points
    plt.scatter(inputs, data_points, marker="+")
    plt.show()
