import torch
import torch.distributions as D
import math
from typing import Callable

from ortho.basis_functions import (
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
    Basis,
    CompositeBasis,
    RandomFourierFeatureBasis,
)
from mercergp.posterior_sampling import MercerSpectralDistribution
from mercergp.kernels import MercerKernel, RandomFourierFeaturesKernel
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
import matplotlib.pyplot as plt
from termcolor import colored


class HilbertSpaceElement:
    """
    A class representing an element of a Hilbert space.
    That is, for a given basis and set of coefficients, instances of this class
    represent functions that belong to the corresponding Hilbert space.

    This is useful when producing Mercer Gaussian Processes.
    """

    def __init__(self, basis: Callable, coefficients: torch.Tensor):
        self.basis = basis
        self.coefficients = coefficients
        self.order = len(coefficients)
        return

    def __call__(self, x):
        """
        Evaluates the Hilbert Space element at the given inputs x.
        """
        if self.coefficients.is_complex():
            return torch.inner(
                self.coefficients,
                torch.complex(self.basis(x), torch.zeros(self.basis(x).shape)),
                # self.basis(x),
            ).squeeze()
        else:
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

    def __rmul__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return HilbertSpaceElement(self.basis, self.coefficients * other)
        else:
            raise NotImplementedError(
                "Multiplication only available for int or float"
            )


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
        mean_function=lambda x: torch.zeros(x.shape[0]),
    ):
        """
        : param basis: a Basis class instance
        : param order:  integer describing the maximum order of the kernel
                        truncation
        :param dim: an integer describing the dimension of the model
        : param kernel: a MercerKernel instance
        : param mean_function: a callable representing the
                            prior mean function for the GP.
        : param raw_posterior_sample: a boolean.
            If false,
            posterior samples suffer variance decay but are "pure" in the
            sense that they strictly follow the Karhunen-Loeve representation
            of a Gaussian process.
            If true,
            posterior samples are built via the combination random features
            and posterior component method hinted at in Wilson 2020.


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
        self.posterior_coefficients = torch.zeros(self.order)
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
        self.posterior_coefficients = self._calculate_posterior_coefficients(
            self.x, self.y
        )
        return

    def set_data(self, x, y):
        """
        Replaces observation data for the given MercerGP.

        :param x: the inputs
        :param y: the outputs
        """
        self.x = x
        self.y = y
        self.posterior_coefficients = self._calculate_posterior_coefficients(
            self.x, self.y
        )

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
            self.basis,
            self._calculate_posterior_coefficients(self.x, self.y),
            self.mean_function,
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

    def get_predictive_density(
        self, test_points: torch.Tensor
    ) -> D.Distribution:
        """
        Returns the predictive density of the Mercer Gaussian process
        evaluated at the inputs "input".

        The predictive density of a Gaussian process is given by:

        THIS USES THE FULL MATRIX OF THE MULTIVARIATE NORMAL
        THAT IS IMPLIED BY K(X*, X*) - K(X*, X)(K+σ^2I)^(-1)K(X, X*)

        NEED TO DOUBLE CHECK THAT THIS IS CORRECT
        """
        use_full_predictive = True
        use_noise = True

        # calculate the mean
        posterior_predictive_mean = self.get_posterior_mean()
        posterior_predictive_mean_evaluation = posterior_predictive_mean(
            test_points
        )
        # breakpoint()
        inputs = self.get_inputs()
        test_matrix = self.kernel(test_points, inputs)
        # kernel_inverse = torch.inverse(
        # self.kernel(inputs, inputs)
        # + (self.kernel.kernel_args["noise_parameter"] ** 2)
        # * torch.eye(len(inputs))
        # )
        kernel_inverse = self.kernel.kernel_inverse(inputs)
        # kernel_inverse = self.kernel.kernel_inverse(inputs)
        if use_full_predictive:
            # now calculate the variance
            posterior_predictive_variance = self.kernel(
                test_points, test_points
            ) - (test_matrix @ kernel_inverse @ test_matrix.t())
        else:
            posterior_predictive_variance = self.kernel(
                test_points, test_points
            )

        if use_noise:
            posterior_predictive_variance += (
                self.kernel.kernel_args["noise_parameter"] ** 2
            ) * torch.eye(len(test_points))

        # add jitter for positive definiteness
        # posterior_predictive_variance_jittered = (
        # posterior_predictive_variance + 0.01 * torch.eye(len(test_points))
        # )
        # print(torch.det(posterior_predictive_variance))
        # assert posterior_predictive_variance
        # breakpoint()

        try:
            distribution = D.MultivariateNormal(
                posterior_predictive_mean_evaluation,
                posterior_predictive_variance,
            )
        except ValueError:
            print(colored("FOUND NEGATIVE VARIANCE!", "red"))
            distribution = D.MultivariateNormal(
                posterior_predictive_mean_evaluation,
                torch.abs(posterior_predictive_variance),
            )
        return distribution

    def get_marginal_predictive_density(
        self, test_points: torch.Tensor
    ) -> D.Distribution:
        """
        Returns the predictive density of the Mercer Gaussian process
        evaluated at the inputs "input".

        The posterior predictive density of a Gaussian process is given by:

        THIS USES THE DIAGONAL OF THE MULTIVARIATE NORMAL
        THAT IS IMPLIED BY K(X*, X*) - K(X*, X)(K+σ^2I)^(-1)K(X, X*)

        NEED TO DOUBLE CHECK THAT THIS IS CORRECT
        """
        # calculate the mean
        posterior_predictive_mean = self.get_posterior_mean()
        posterior_predictive_mean_evaluation = posterior_predictive_mean(
            test_points
        )
        inputs = self.get_inputs()

        # now calculate the variance
        posterior_predictive_variance = (
            self.kernel(test_points, test_points)
            - self.kernel(test_points, inputs)
            @ self.kernel.kernel_inverse(inputs)
            @ self.kernel(inputs, test_points)
            # + self.kernel.kernel_args["noise_parameter"] ** 2
        )

        # add jitter for positive definiteness
        posterior_predictive_variance += 0.00001 * torch.eye(len(test_points))
        try:
            distribution = D.Normal(
                posterior_predictive_mean_evaluation,
                torch.diag(posterior_predictive_variance),
            )
        except ValueError:
            distribution = D.Normal(
                posterior_predictive_mean_evaluation,
                torch.abs(torch.diag(posterior_predictive_variance)),
            )
            print("FOUND NEGATIVE VARIANCE!")
        return distribution

    def _calculate_posterior_coefficients(self, x, y) -> torch.Tensor:
        """
        Returns the non-random coefficients for the posterior mean according
        to the kernel as added to the Gaussian process, and under the data
        passed to the Mercer Gaussian process. This will be zeroes for no
        targets

        That is, these are the posterior mean coefficients related to
        (y-m)'(K(x, x) + σ^2I)^{-1}
        """

        interim_matrix = self.kernel.get_interim_matrix_inverse(x)
        ksi = self.kernel.get_ksi(x)
        posterior_coefficients = torch.einsum(
            "jm, mn -> jn", interim_matrix, ksi.t()
        )
        # inner product the matrix parts with the outputs/residual parts
        result = torch.einsum("i..., ji -> j", y, posterior_coefficients)
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
            raise ValueError(
                "Error: some elements of the covariance matrix are negative."
            )

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


class MercerGPFourierPosterior(MercerGP):
    """
    Following Wilson et al (2020), this allows us to construct the posterior
    with a prior component using a Cosine basis and a posterior component
    using a Mercer (orthogonal Favard feature) basis. This combines the two
    to get a better representation of the prior and avoiding the variance decay
    problem.
    """

    def __init__(
        self,
        basis: Basis,
        rff_basis: RandomFourierFeatureBasis,
        order: int,
        rff_order: int,
        dim: int,
        kernel: MercerKernel,
        # spectral_distribution: D.Distribution,
        mean_function=lambda x: torch.zeros(x.shape),
        prior_variance=1.0,
    ):
        """
        Initialises the MercerGPFourierPosterior class.
        Key to this is the initialisation under the superclass to
        produce a standard Mercer posterior component,
        and then a RFF GP for modelling the prior component.
        """
        self.rffgp = NonStationaryRFFGP(
            rff_basis,
            rff_order,
            dim,
            # spectral_distribution,
            # marginal_distribution_1,  # first spectral frequency marginal
            # marginal_distribution_2,  # second spectral frequency marginal
            mean_function=lambda x: torch.zeros(x.shape),
        )
        self.prior_variance = prior_variance
        super().__init__(basis, order, dim, kernel, mean_function)

    def gen_gp(self):
        # i.e., we are generating a Wilson(2020) style prior- and posterior
        # decomposition
        prior_component = (
            self.prior_variance * self.rffgp.gen_gp()
        )  # this should be f + ig

        # evaluate the prior component at the inputs to get the posterior
        # component residuals
        if len(self.x) > 0 and len(self.y) > 0:
            residuals = self.y - prior_component(self.x)
            residual_posterior_coefficients = (
                self._calculate_posterior_coefficients(self.x, residuals.real)
            )
            posterior_component = MercerGPSample(
                self.basis, residual_posterior_coefficients, self.mean_function
            )
        else:
            posterior_component = MercerGPSample(
                self.basis,
                self._calculate_posterior_coefficients(self.x, self.y),
                self.mean_function,
            )

        gp_sample = PosteriorGPSample(prior_component, posterior_component)
        return gp_sample


class RFFGP(MercerGP):
    """
    A class representing a Gaussian process using Random Fourier Features.
    """

    def __init__(
        self,
        order: int,
        dim: int,
        spectral_distribution: torch.distributions.Distribution,
        mean_function=lambda x: torch.zeros(x.shape),
    ):
        kernel_args = {
            "noise_parameter": torch.Tensor([0.0]),
            "variance_parameter": torch.Tensor([1.0]),
        }
        basis = RandomFourierFeatureBasis(dim, order, spectral_distribution)
        kernel = RandomFourierFeaturesKernel(
            order, spectral_distribution, dim, kernel_args
        )
        super().__init__(basis, order, dim, kernel, mean_function)
        return

    def _get_sample_coefficients(self):
        """
        Generates the sample coefficients for a given sample gp."""
        variance = torch.eye(self.order)
        normal_rv = (
            torch.distributions.Normal(
                loc=torch.zeros(self.order), scale=torch.ones(self.order)
            )
            .sample()
            .squeeze()
        )
        return normal_rv

    # def add_data(self, x, y):
    # """
    # Adds observation data for the given MercerGP.

    # :param x: the inputs
    # :param y: the outputs
    # """
    # # add the inputs and alter the coefficients
    # self.x = torch.cat([self.x, x])
    # self.y = torch.cat([self.y, y])
    # return


class NonStationaryRFFGP(RFFGP):
    """
    A Gaussian process based on the Random Fourier Features approach
    but with a non-stationary kernel.
    In order to build Fourier features for a non-stationary kernel,
    it is necessary to utilise Yaglom's theorem in tandem with a
    complex Gaussian process to get the right covariance.
    """

    def __init__(
        self,
        basis: RandomFourierFeatureBasis,
        order: int,
        dim: int,
        # spectral_distribution: torch.distributions.Distribution,
        # marginal_distribution_1: D.Distribution,  # first spectral frequency marginal distribution
        # marginal_distribution_2: D.Distribution,  # second spectral frequency marginal distribution
        mean_function=lambda x: torch.zeros(x.shape),
    ):
        # standard GP model parameters
        if not order % 2 == 0:
            raise ValueError(
                "Order of Nonstationary RFF basis should be an even number"
            )
        if not isinstance(basis.w_dist, MercerSpectralDistribution):
            raise TypeError(
                "self.basis.w_dist should be a MercerSpectralDistribution"
            )
        self.order = order
        self.dim = dim
        self.mean_function = mean_function
        self.basis = basis

        # the 2-d spectral distribution from Yaglom's theorem in tandem with
        # the marginals are necessary to produce the right covariance.

    def gen_gp(self):
        coefficients = self._get_sample_coefficients()
        return HilbertSpaceElement(
            self.basis,
            coefficients,
        )

    def _get_sample_coefficients(self):
        """
        Generates the sample coefficients for a given sample gp.
        The first coefficients are repeated for the resulting
        Non-stationary GP.

        The second coefficients are not repeated.
        In the following, we denote the imaginary unit by j.

        This is because the features are written as:
            φ_i = cos(xω_1 + b) + cos(xω_2 + b)

                     /  cos(xω_1 + b), i odd
            φ'_i =  {
                     \  cos(xω_2 + b), i even
        and we sum to give:
            h(x) = f(x) + ig(x)
                 = Σ_^{R}i θ_i φ_i(x) + j Σ^{2R}_i θ'_i φ'_i(x)

        This can be rearranged as:

                  /  cos(xω_1 + β), 1 < i < R
            φ_i= {
                  \  cos(xω_2 + β), R < i < 2R

            and summing with coefficients:
            h(x) = f(x) + ig(x)
                 = Σ^{2R}_i θ_i φ_i(x) + j Σ^{2R}_i θ'_i φ_i(x)
                 = Σ^{2R}_i (θ_i + jθ'_i) φ_i(x)

            where θ_R+i = θ_i, so that the sum captures each time
            the appropriate basis function.

        This function returns the random sample coefficients according to this
        structure.
        """
        normal_rv = (
            torch.distributions.Normal(
                loc=torch.zeros(math.floor(self.order / 2)),
                scale=torch.ones(math.floor(self.order / 2)),
            )
            .sample()
            .squeeze()
        )
        normal_rv_2 = (
            torch.distributions.Normal(
                loc=torch.zeros(self.order),
                scale=torch.ones(self.order),
            )
            .sample()
            .squeeze()
        )
        # repeat the real part's coefficients
        first_coeffics = torch.cat((normal_rv, normal_rv))

        # don't repeat the imaginary part's coefficients
        second_coeffics = normal_rv_2

        # combine to complex coefficients
        # coeffics = torch.view_as_complex(
        # torch.vstack((first_coeffics, second_coeffics)).t()
        # )
        coeffics = torch.complex(first_coeffics, second_coeffics)
        return coeffics


class SmoothExponentialRFFGP(RFFGP):
    """
    A class representing a Gaussian process using Random Fourier Features
    for the smooth exponential kernel.
    """

    def __init__(
        self,
        order: int,
        dim: int,
        mean_function=lambda x: torch.zeros(x.shape),
    ):
        spectral_distribution = D.Normal(0.0, 1.0)
        super().__init__(order, dim, spectral_distribution, mean_function)
        return


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
        gp_component = super().__call__(x)
        mean_component = self.mean_function(x)
        return gp_component + mean_component


class PosteriorGPSample(HilbertSpaceElement):
    def __init__(
        self,
        prior_component: HilbertSpaceElement,
        posterior_component: MercerGPSample,
    ):
        self.prior_component = prior_component
        self.posterior_component = posterior_component

    def get_rff_order(self):
        return self.prior_component.get_order()

    def get_order(self):
        return self.posterior_component.get_order()

    def __call__(self, x):
        return self.prior_component(x) + self.posterior_component(x)
        # return self.posterior_component(x)


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
        c = torch.sqrt(a**2 + 2 * a * b)
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

    def get_posterior_mean(self) -> HilbertSpaceElement:
        """
        Returns the posterior mean function.
        """
        return HermiteMercerGPSample(
            self._calculate_posterior_coefficients(),
            self.dim,
            self.kernel.get_params(),
            self.mean_function,
            self.mean_function_derivative,
        )


if __name__ == "__main__":
    plot_example = False
    test_predictive_densities = False
    test_single_predictive_density = False
    test_multidim = True

    # parameters for the test
    sample_size = 300

    # build a mercer kernel
    order = 15  # degree of approximation
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

    a = 1
    b = 2
    c = 3

    def data_func(x):
        return a * x**2 + b * x + x

    data_points = data_func(inputs) + torch.distributions.Normal(0, 1).sample(
        inputs.shape
    )

    # build a standard kernel for comparison
    # show the two kernels for comparison. They're close!
    # create pseudodata for training purposes
    mercer_gp = MercerGP(basis, order, dim, test_kernel)
    mercer_gp.add_data(inputs, data_points)

    # test the inverse
    # inv_1 = test_kernel.kernel_inverse(inputs)
    # inv_3 = torch.inverse(test_kernel(inputs, inputs))
    if plot_example:
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
            test_points.flatten().numpy(),
            data_func(test_points).flatten().numpy(),
        )
        # input/output points
        plt.scatter(inputs, data_points, marker="+")
        # plt.show()

        # if test_predictive_densities:
        predictive_density = mercer_gp.get_marginal_predictive_density(
            test_points
        )
        plt.plot(
            test_points,
            predictive_density.loc.numpy().flatten(),
            color="purple",
        )
        plt.plot(
            test_points,
            predictive_density.loc.numpy().flatten()
            + 2 * predictive_density.scale.numpy().flatten(),
            color="red",
        )
        plt.plot(
            test_points,
            predictive_density.loc.numpy().flatten()
            - 2 * predictive_density.scale.numpy().flatten(),
            color="red",
        )
        plt.show()
        # plt.plot(
        # inputs, predictive_density.loc[0].numpy().flatten(), color="purple"
        # )
        print("Predictive density:", predictive_density)

    if test_single_predictive_density:
        test_point = torch.tensor(0.0)
        predictive_density_single = mercer_gp.get_marginal_predictive_density(
            test_point
        )
        print("Predictive Density Single:", predictive_density_single)

    if test_multidim:
        noise_dist = D.Normal(0.0, 0.1)

        def test_function_2d(x: torch.Tensor):
            return torch.sin(x[:, 0]) + torch.cos(x[:, 1])

        eigenvalue_gen = SmoothExponentialFasshauer(order, 2)
        eigenvalues = eigenvalue_gen((mercer_args, mercer_args))
        # basis_1 = Basis(
        # smooth_exponential_basis_fasshauer, 2, order, mercer_args
        # )
        basis = Basis(
            (
                smooth_exponential_basis_fasshauer,
                smooth_exponential_basis_fasshauer,
            ),
            2,
            order,
            mercer_args,
        )
        fineness = 500
        x = torch.linspace(-5, 5, fineness)
        y = torch.linspace(-5, 5, fineness)
        inputs = torch.vstack((x, y)).t()
        # plt.plot(basis(inputs))
        # plt.show()
        test_kernel_2d = MercerKernel(order, basis, eigenvalues, mercer_args)
        results = test_kernel_2d(torch.zeros(1, 2), inputs)

        plt.plot(results[0])
        plt.show()

        multidim_mercer_gp = MercerGP(basis, order, 2, test_kernel_2d)
        multidim_mercer_gp.add_data(
            inputs, test_function_2d(inputs) + noise_dist.sample((fineness,))
        )
        print(colored("ABOUT TO DO IT!", "red"))
        gp_sample = multidim_mercer_gp.gen_gp()
        # results = test_kernel_2d(torch.zeros(1, 2), inputs)

        # plt.plot(results[0])
        # plt.show()
