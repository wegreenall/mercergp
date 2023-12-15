import torch

from mercergp.MGP import HermiteMercerGP, HermiteMercerGPSample
from mercergp import kernels
from ortho.basis_functions import (
    Basis,
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
)
import matplotlib.pyplot as plt

order = 10
half_window_length = 6
fineness = 1000
x_axis = torch.linspace(-half_window_length, half_window_length, fineness)
params = {
    "ard_parameter": torch.Tensor([10.0]),
    "noise_parameter": torch.Tensor([0.01]),
    "variance_parameter": torch.Tensor([1.0]),
    "precision_parameter": torch.Tensor([1.0]),
}
basis = Basis(smooth_exponential_basis_fasshauer, 1, order, params)
eigenvalues = smooth_exponential_eigenvalues_fasshauer(order, params)
# breakpoint()
kernel = kernels.MercerKernel(
    order,
    basis,
    eigenvalues,
    params,
)
torch.manual_seed(6)
my_mercer = HermiteMercerGP(order, 1, kernel)
sample = my_mercer.gen_gp()
plt.plot(x_axis, sample(x_axis))
plt.plot(x_axis, sample.derivative(x_axis))
plt.plot(
    x_axis[1:],
    torch.diff(sample(x_axis)) / (2 * half_window_length / fineness),
    color="black",
)
plt.show()
