# builders.py
import torch
from mercergp import MGP
from ortho import basis_functions as bf


def build_mercer_gp(
    basis: bf.Basis,
    ard_parameter: torch.Tensor,
    precision_parameter: torch.Tensor,
    noise_parameter: torch.Tensor,
    order: int,
    dim: int,
) -> MGP.MercerGP:
    """
    Returns a MercerGP instance, with kernel and basis constructed
    from the Gaussian kernel.
    """
    kernel_params = {
        "ard_parameter": ard_parameter,
        "precision_parameter": precision_parameter,
        "noise_parameter": noise_parameter,
    }
    eigenvalues = bf.smooth_exponential_eigenvalues_fasshauer(
        order, kernel_params
    )
    # basis = bf.Basis(
    # bf.smooth_exponential_basis_fasshauer, dim, order, kernel_params
    # )
    kernel = MGP.MercerKernel(order, basis, eigenvalues, kernel_params)
    mercer_gp = MGP.MercerGP(basis, order, dim, kernel)
    return mercer_gp
