# builders.py
import torch
import torch.distributions as D
import matplotlib.pyplot as plt

# from mercergp import MGP
from mercergp.eigenvalue_gen import EigenvalueGenerator
from ortho import basis_functions as bf

from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from mercergp.likelihood import MercerLikelihood
from mercergp.kernels import MercerKernel
from mercergp.MGP import MercerGP, MercerGPFourierPosterior
from mercergp.posterior_sampling import (
    histogram_spectral_distribution,
    integer_spectral_distribution,
    gaussian_spectral_distribution,
)

from enum import Enum


class GPBuilderState(Enum):
    """
    Enum for the state of the GPBuilder.
    """

    NOT_READY = 0
    READY = 1


class GPType(Enum):
    """
    Enum for the type of GP.
    """

    MERCER = 0
    FOURIER = 1


class MercerGPBuilder:
    def __init__(self, order: int, dim: int = 1):
        """
        Builder class for generating Mercer Gaussian Process instances.
        Parameters:
            - order: the order of the GP
            - dim: the dimension of the GP; currently implemented only for 1d
            - gp_type: the type of GP to build (Standard Mercer or
                                                FourierPosterior)

        The GPBuilder acts as a state machine that carries information about
        whether it is ready to build a GP.

        A GP requires:
            - a Basis
            - an order
            - a kernel
        Optionally:
            - a mean function

        If building a MercerFourierPosterior GP,

        To get a kernel, we can either generate one outside the builder,
        or pass in a basis and eigenvalue generator, and have the kernel
        be built internally.

        Once the GPBuilder has a basis and a kernel, it enters the READY state
        and can produce a GP.

        After all updates, the "_check_ready" method is called to
        advance the state of the builder.

        Paths through the machine:
            1) generatng a kernel internally:
                (in any order)
                - set a basis
                - set an eigenvalue generator
                - set kernel parameters
                On completion of these, a kernel will be set automatically
                we then have kernel and basis, so the GPBuilder is READY
            2) generating a kernel externally
               - set a kernel
               On setting a kernel, since a kernel requires a basis anyway,
               we pull it from the kernel and use it as the basis.
               the GPBuilder is then set to READY
        """
        self.state = GPBuilderState.NOT_READY
        self.order = order
        if dim != 1:
            raise NotImplementedError(
                "Multidimensional version of GPBuilder not implemented."
            )
        self.dim = dim
        self.has_basis = False
        self.has_eigenvalue_generator = False
        self.has_kernel = False
        self.has_parameters = False

    def _check_ready(self):
        """
        Checks the status of the MercerGP builder.

        If we have a basis, an eigenvalue_generator and kernel parameters,
        we can build a kernel and use that as the kernel.

        If we have a kernel, we can build a GP.
        """
        if self.has_basis and not self.has_kernel and not self.has_parameters:
            self.set_parameters(self.basis.parameters)
        if (
            self.has_basis
            and self.has_eigenvalue_generator
            and self.has_parameters
        ) and not (self.has_kernel):
            self.set_kernel(
                MercerKernel(
                    self.order,
                    self.basis,
                    self.eigenvalue_generator(self.parameters),
                    self.parameters,
                )
            )

        elif self.has_kernel and not self.has_basis:
            self.set_basis(self.kernel.basis)

        if self.has_kernel and self.has_basis:
            self.state = GPBuilderState.READY

    def set_basis(self, basis: bf.Basis):
        self.basis = basis
        self.has_basis = True
        self._check_ready()
        return self

    def set_eigenvalue_generator(
        self, eigenvalue_generator: EigenvalueGenerator
    ):
        self.eigenvalue_generator = eigenvalue_generator
        self.has_eigenvalue_generator = True
        self._check_ready()
        return self

    def set_parameters(self, parameters: dict):
        self.parameters = parameters
        self.has_parameters = True
        self._check_ready()
        return self

    def set_kernel(self, kernel: MercerKernel):
        self.kernel = kernel
        self.has_kernel = True
        self._check_ready()
        return self

    def build(self) -> MercerGP:
        if self.state == GPBuilderState.READY:
            return MercerGP(self.basis, self.order, self.dim, self.kernel)
        else:
            raise RuntimeError(
                "GPBuilder not ready to build GP. Current flags: \
                             \n - has_basis: {} \n - has_eigenvalue_generator:\
                             {} \n - has_kernel {}\n \
                             Please ensure that the \
                             builder has a basis and eigenvalues, or a kernel.".format(
                    self.has_basis,
                    self.has_eigenvalue_generator,
                    self.has_kernel,
                )
            )


class FourierPosteriorMercerGPBuilder(MercerGPBuilder):
    def __init__(self, order: int, rff_order: int, dim: int = 1):
        """
        Since MercerGPFourierPosterior is  subclass
        of MercerGP,
        The requirements for the former are a superset of
        those for the latter. As a result, we can subclass
        MercerGPBuilder to make construction of a builder for
        MercerGPFourierPosterior simpler.

        Specifically, in order for this builder to be READY,
        it must have all the prerequisites for a MercerGP, AND
        have an rff_order and a random fourier feature basis.
        """
        # super(self, MercerGPBuilder).__init__(order, dim)
        super().__init__(order, dim)
        self.state = GPBuilderState.NOT_READY
        self.fourier_state = GPBuilderState.NOT_READY
        self.rff_order = rff_order
        self.has_fourier_basis = False
        self.fourier_basis = None

    def set_fourier_basis(self, fourier_basis: bf.RandomFourierFeatureBasis):
        self.fourier_basis = fourier_basis
        self.has_fourier_basis = True
        self._check_ready()
        return self

    def _check_ready(self):
        super()._check_ready()
        if self.state == GPBuilderState.READY and self.has_fourier_basis:
            self.fourier_state = GPBuilderState.READY

    def build(self):
        if self.fourier_state == GPBuilderState.READY:
            return MercerGPFourierPosterior(
                self.basis,
                self.fourier_basis,
                self.order,
                self.rff_order,
                self.dim,
                self.kernel,
            )
        else:
            raise RuntimeError(
                "FourierPosteriorMercerGPBuilder not ready to build GP. Current flags: \
                             \n - has_basis: {} \n - has_eigenvalue_generator:\
                             {} \n - has_kernel {}\n - has_fourier_basis: {} \n \
                             Please ensure that the \
                             builder has a basis and eigenvalues, or a kernel.".format(
                    self.has_basis,
                    self.has_eigenvalue_generator,
                    self.has_kernel,
                    self.has_fourier_basis,
                )
            )


def build_mercer_gp(
    parameters: dict,
    order: int,
    basis: bf.Basis,
    # input_sample: torch.Tensor,
    eigenvalue_generator: EigenvalueGenerator,
    dim=1,
):
    """
    parameters requires in params:
        - ard_parameter,
        - precision parameter,
        - noise_parameter
    """
    eigenvalues = eigenvalue_generator(parameters)

    # build the kernel
    """
    When in multiple dimensions, the resulting parameters are a tuple or
    list containing dicts, one for each dimension (in the current setup).
    However the kernel needs to be given a dict of parameter to get
    e.g. the noise parameter. As a result, we just pass it the first one
    under the assumption that the noise parameter will be given as the same in each
    dimension (since it's noise on the output rather than the input).
    """
    if dim != 1:
        kernel = MercerKernel(order, basis, eigenvalues, parameters[0])
    else:
        kernel = MercerKernel(order, basis, eigenvalues, parameters)

    # build the gp
    # builder = (
    # GPBuilder(order, dim)
    # .set_basis(basis)
    # .set_eigenvalue_generator(eigenvalue_generator)
    # .set_parameters(parameters)
    # .build()
    # )
    mgp = MercerGP(basis, order, dim, kernel)
    return mgp


def train_mercer_params(
    parameters: dict,
    input_sample: torch.Tensor,
    output_sample: torch.Tensor,
    basis: bf.Basis,
    optimiser: torch.optim.Optimizer,
    eigenvalue_gen: EigenvalueGenerator,
    dim=1,
    memoise=True,
    iter_count=60000,
) -> dict:
    """
    Given:
        - a dictionary of parameters (parameters);
        - the order of the basis (order);
        - an input sample (input_sample);
        - an output sample (output_sample);
        - a basis (basis);
        - a torch.optim.Optimizer marked with the parameters (optimiser);
        - an eigenvalue generating class (eigenvalue_gen);

    this trains the parameters of a Mercer Gaussian process.
    """
    order = basis.get_order()
    mgp_likelihood = MercerLikelihood(
        order,
        optimiser,
        basis,
        input_sample,
        output_sample,
        eigenvalue_gen,
        memoise,
    )
    new_parameters = parameters.copy()
    mgp_likelihood.fit(new_parameters, iter_count=iter_count)
    for param in filter(
        lambda param: isinstance(new_parameters[param], torch.Tensor),
        new_parameters,
    ):
        print(new_parameters[param])
        new_parameters[param] = new_parameters[param].detach()

    return new_parameters


def train_smooth_exponential_mercer_params(
    parameters: dict,
    order: int,
    input_sample: torch.Tensor,
    output_sample: torch.Tensor,
    optimiser: torch.optim.Optimizer,
    dim=1,
    iter_count=60000,
) -> dict:
    """
    Given:
        - a dictionary of parameters (parameters);
        - the order of the basis (order);
        - an input sample (input_sample);
        - an output sample (output_sample);
        - a torch.optim.Optimizer marked with the parameters (optimiser);

    this trains the parameters of a Mercer Gaussian process with Gaussian
    process inputs.

    Using the standard Fasshauer basis, this function trains the smooth
    exponential kernel based GP model parameters.
    """
    # basis = get_orthonormal_basis_from_sample(
    # input_sample, weight_function, order
    # )
    basis = bf.Basis(
        bf.smooth_exponential_basis_fasshauer, dim, order, parameters
    )
    mgp_likelihood = MercerLikelihood(
        order,
        optimiser,
        basis,
        input_sample,
        output_sample,
        SmoothExponentialFasshauer(order),
    )
    new_parameters = parameters.copy()
    # breakpoint()
    mgp_likelihood.fit(new_parameters, iter_count)
    for param in filter(
        lambda param: isinstance(new_parameters[param], torch.Tensor),
        new_parameters,
    ):
        print(new_parameters[param])
        new_parameters[param] = new_parameters[param].detach()
    return new_parameters


def build_smooth_exponential_mercer_gp(
    parameters: dict,
    order: int,
    # input_sample: torch.Tensor,
    dim=1,
):
    """
    parameters requires in params:
        - ard_parameter,
        - precision parameter,
        - noise_parameter
    """
    basis = bf.Basis(
        bf.smooth_exponential_basis_fasshauer, dim, order, parameters
    )
    eigenvalue_generator = SmoothExponentialFasshauer(order)
    eigenvalues = eigenvalue_generator(parameters)

    # build the kernel
    kernel = MercerKernel(order, basis, eigenvalues, parameters)

    # build the gp
    mgp = MercerGP(basis, order, dim, kernel)
    return mgp


def build_smooth_exponential_mercer_gp_fourier_posterior(
    parameters: dict,
    order: int,
    # input_sample: torch.Tensor,
    dim=1,
    begin=-5,
    end=5,
    frequency=1000,
    spectral_distribution_type="gaussian",
    rff_order: int = 5000,
    prior_variance=1.0,
):
    """
    parameters requires in params:
        - ard_parameter,
        - precision parameter,
        - noise_parameter
    """
    basis = bf.Basis(
        bf.smooth_exponential_basis_fasshauer, dim, order, parameters
    )
    eigenvalue_generator = SmoothExponentialFasshauer(order)
    eigenvalues = eigenvalue_generator(parameters)

    # build the kernel
    kernel = MercerKernel(order, basis, eigenvalues, parameters)

    print("frequency in build_mercer_gp_fourier_posterior", frequency)
    # build the rff_basis
    if spectral_distribution_type == "histogram":
        spectral_distribution = histogram_spectral_distribution(
            kernel, begin, end, frequency
        )
    elif spectral_distribution_type == "integer":
        spectral_distribution = integer_spectral_distribution(
            kernel, begin, end, frequency
        )
    elif spectral_distribution_type == "gaussian":
        spectral_distribution = gaussian_spectral_distribution(
            kernel, begin, end, frequency
        )
    rff_basis = bf.RandomFourierFeatureBasis(
        dim, rff_order, spectral_distribution
    )
    # build the gp
    mgp = MercerGPFourierPosterior(
        basis,
        rff_basis,
        order,
        rff_order,
        dim,
        kernel,
        prior_variance=prior_variance,
    )
    return mgp


def build_mercer_gp_fourier_posterior(
    parameters: dict,
    order: int,
    basis: bf.Basis,
    # input_sample: torch.Tensor,
    eigenvalue_generator: EigenvalueGenerator,
    dim=1,
    begin=-5,
    end=5,
    frequency=1000,
    spectral_distribution_type="gaussian",
    rff_order: int = 5000,
    prior_variance=1.0,
) -> MercerGPFourierPosterior:
    """
    parameters requires in params:
        - ard_parameter,
        - precision parameter,
        - noise_parameter
    """

    # build the kernel
    eigenvalues = eigenvalue_generator(parameters)
    kernel = MercerKernel(order, basis, eigenvalues, parameters)

    print("frequency in build_mercer_gp_fourier_posterior", frequency)
    # build the rff_basis
    if spectral_distribution_type == "histogram":
        spectral_distribution = histogram_spectral_distribution(
            kernel, begin, end, frequency
        )
    elif spectral_distribution_type == "integer":
        spectral_distribution = integer_spectral_distribution(
            kernel, begin, end, frequency
        )
    elif spectral_distribution_type == "gaussian":
        spectral_distribution = gaussian_spectral_distribution(
            kernel, begin, end, frequency
        )
    rff_basis = bf.RandomFourierFeatureBasis(
        dim, rff_order, spectral_distribution
    )
    # build the gp
    mgp = MercerGPFourierPosterior(
        basis,
        rff_basis,
        order,
        rff_order,
        dim,
        kernel,
        prior_variance=prior_variance,
    )
    return mgp


# def build_mercer_gp(
# basis: bf.Basis,
# ard_parameter: torch.Tensor,
# precision_parameter: torch.Tensor,
# noise_parameter: torch.Tensor,
# order: int,
# dim: int,
# ) -> MGP.MercerGP:
# """
# Returns a MercerGP instance, with kernel and basis constructed
# from the Gaussian kernel.
# """
# kernel_params = {
# "ard_parameter": ard_parameter,
# "precision_parameter": precision_parameter,
# "noise_parameter": noise_parameter,
# }
# eigenvalues = bf.smooth_exponential_eigenvalues_fasshauer(
# order, kernel_params
# )
# # breakpoint()
# # basis = bf.Basis(
# # bf.smooth_exponential_basis_fasshauer, dim, order, kernel_params
# # )
# kernel = MGP.MercerKernel(order, basis, eigenvalues, kernel_params)
# mercer_gp = MGP.MercerGP(basis, order, dim, kernel)
# return mercer_gp

if __name__ == "__main__":
    """
    The program begins here
    """

    def test_function(x: torch.Tensor) -> torch.Tensor:
        """
        Test function used in an iteration of Daskalakis, Dellaportas and Panos.
        """
        return (1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8).squeeze()

    build_mercer = False
    build_se_mercer = False
    build_mercer_fourier = True
    test_posterior_sampling_correlation = False

    # plotting
    axis_width = 6
    x_axis = torch.linspace(-axis_width, axis_width, 1000)  # .unsqueeze(1)
    test_sample_size = 500
    test_sample_shape = torch.Size([test_sample_size])

    # hyperparameters
    order = 8
    rff_order = 3700
    rff_frequency = 2000
    dimension = 1
    l_se = torch.Tensor([[0.6]])
    sigma_se = torch.Tensor([1.0])
    prec = torch.Tensor([1.0])
    sigma_e = torch.Tensor([0.2])
    kernel_args = {
        "ard_parameter": l_se,
        "variance_parameter": sigma_se,
        "noise_parameter": sigma_e,
        "precision_parameter": prec,
    }
    # basis = Basis()
    basis = bf.Basis(
        bf.smooth_exponential_basis_fasshauer,
        dimension,
        order,
        kernel_args,
    )
    eigenvalue_generator = SmoothExponentialFasshauer(order)

    if build_mercer:
        mercer_gp = build_mercer_gp(
            kernel_args, order, basis, eigenvalue_generator
        )
        inputs = D.Normal(0.0, 1.0).sample(test_sample_shape)
        outputs = test_function(inputs)
        mercer_gp.add_data(inputs, outputs)
        posterior_sample = mercer_gp.gen_gp()
        sample_data = posterior_sample(x_axis)
        breakpoint()
        plt.plot(x_axis, sample_data.real)
        plt.scatter(inputs, outputs)
        plt.show()

    if build_se_mercer:
        pass

    if build_mercer_fourier:
        mercer_gp_fourier_posterior = build_mercer_gp_fourier_posterior(
            kernel_args,
            order,
            rff_order,
            basis,
            eigenvalue_generator,
            frequency=rff_frequency,
        )
        inputs = D.Normal(0.0, 1.0).sample(test_sample_shape)
        outputs = test_function(inputs) + sigma_e * D.Normal(0.0, 1.0).sample(
            inputs.shape
        )

        # mercer
        # mercer_gp.add_data(inputs, outputs)
        # posterior_sample = mercer_gp_fourier_posterior.gen_gp()
        # sample_data = posterior_sample(x_axis)

        # fourier posterior mercer
        mercer_gp_fourier_posterior.add_data(inputs, outputs)
        posterior_sample = mercer_gp_fourier_posterior.gen_gp()
        sample_data = posterior_sample(x_axis)

        plt.plot(x_axis, sample_data.real)
        if sample_data.is_complex():
            plt.plot(x_axis, sample_data.imag)
        plt.scatter(inputs, outputs)
        plt.show()

    if test_posterior_sampling_correlation:
        raise NotImplementedError
        """
        Currently not implemented.
        """
        axis_width = 6
        x_axis = torch.linspace(-axis_width, axis_width, 1000)  # .unsqueeze(1)
        correlation_testing_sample_size = 100000
        correlation_testing_sample_shape = torch.Size(
            [correlation_testing_sample_size]
        )

        test_sample_size = 20
        test_sample_shape = torch.Size([test_sample_size])
        # rff_order = 1000
        eigenvalue_generator = SmoothExponentialFasshauer(order)
        mercer_gp_fourier_posterior = build_mercer_gp_fourier_posterior(
            kernel_args, order, rff_order, basis, eigenvalue_generator
        )

        inputs = D.Normal(0.0, 1.0).sample(test_sample_shape)
        outputs = test_function(inputs) + sigma_e * D.Normal(0.0, 1.0).sample(
            inputs.shape
        )
        mercer_gp.add_data(inputs, outputs)
        mercer_gp_fourier_posterior.add_data(inputs, outputs)
        fourier_posterior_sample = mercer_gp_fourier_posterior.gen_gp()
        fourier_sample_data = fourier_posterior_sample(x_axis)

        # plt.plot(x_axis, sample_data.real)
        # if sample_data.is_complex():
        # plt.plot(x_axis, sample_data.imag)
        # plt.scatter(inputs, outputs)
        # plt.show()

        unif_dist = D.Uniform(-axis_width, axis_width)
        unif_sample = unif_dist.sample(correlation_testing_sample_size)
        posterior_sample_unifs = posterior_sample(unif_sample)
