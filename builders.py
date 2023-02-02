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
    mgp_likelihood.fit(new_parameters)
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
    mgp_likelihood.fit(new_parameters)
    for param in new_parameters:
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


def build_mercer_gp_fourier_posterior(
    parameters: dict,
    order: int,
    rff_order: int,
    basis: bf.Basis,
    # input_sample: torch.Tensor,
    eigenvalue_generator: EigenvalueGenerator,
    dim=1,
    begin=-5,
    end=5,
    frequency=1000,
    spectral_distribution_type="gaussian",
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
        basis, rff_basis, order, rff_order, dim, kernel
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
