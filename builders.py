# builders.py
import torch
from mercergp import MGP
from mercergp.eigenvalue_gen import PolynomialEigenvalues, EigenvalueGenerator
from ortho import basis_functions as bf

from mercergp.eigenvalue_gen import SmoothExponentialFasshauer
from mercergp.likelihood import MercerLikelihood
from mercergp.kernels import MercerKernel
from mercergp.MGP import MercerGP


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
        bf.smooth_exponential_basis_fasshauer, 1, order, parameters
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
