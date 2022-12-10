import torch
from ortho.basis_functions import smooth_exponential_eigenvalues_fasshauer
import matplotlib.pyplot as plt
import math
from termcolor import colored


def harmonic(m, k):
    """
    Returns the generalised harmonic number H(m, k)
    """
    # print("m:", m)
    # print("k:", k)
    terms = [1 / (i ** k) for i in range(1, m)]
    harmonics = sum(terms) if m > 1 else 1
    return harmonics


class EigenvalueGenerator:
    """
    When constructing a Mercer Gaussian process kernel, the form of the kernel
    is:
                Σ λ_i φ_i(x) φ_i(x')

    Given a specific basis, a kernel's behaviour can be regulated via choice of
    the eigenvalues. The functional form of the eigenvalues w.r.t some
    parameters affects the behaviour of the Gaussian process samples, and so
    comprises one part of the prior over Gaussian process functions that are
    being modelled. As a result it is necessary to specify the specific
    functional form of the eigenvalues in order to impose the corresponding
    prior behaviour. Implementations of specific eigenvalue forms
    (expressed as subclassses of this class), represent different prior forms
    for the eigenvalues.
    """

    def __init__(self, order):
        self.order = order

    def __call__(self, parameters: dict) -> torch.Tensor:
        """
        Returns the eigenvalues, up to the order set at initialisation,
        given the dictionary of parameters 'parameters'.
        """
        self.check_params(parameters)
        raise NotImplementedError("Please use a subclass of this class")

    def check_params(self, parameters: dict):
        for key in self.required_params:
            if key not in parameters:
                print(
                    "Required parameters not included; please ensure to include the required parameters:\n"
                )
                print([param for param in self.required_parameters])
                raise ValueError("Missing required parameter!")
        return True


class SmoothExponentialFasshauer(EigenvalueGenerator):
    required_parameters = ["ard_parameter", "precision_parameter"]

    def __call__(self, parameters: dict) -> torch.Tensor:
        return smooth_exponential_eigenvalues_fasshauer(self.order, parameters)


class PolynomialEigenvalues(EigenvalueGenerator):
    required_params = ["scale", "shape", "degree"]
    """
    param scale:
        Scales all eigenvalues - like the kernel 'variance' param
    param shape:
        A vector affecting the shape of the eigenvalue sequence.
        Must be increasing.
    param degree:
        The polynomial degree. Following (Reade, 1992), the 
        smoothness of sample functions (in the limit, so this
        is an approximation) will be described by this degree.
        As parameterised here, a degree of k leads to sample functions
        that would be k times differentiable.

    """

    def __init__(
        self,
        order,
    ):
        self.order = order

    def __call__(self, parameters: dict) -> torch.Tensor:
        """
        Calculates and returns the eigenvalues.

        output shape: [order]
        """
        scale = parameters["scale"]
        shape = parameters["shape"]
        degree = parameters["degree"]
        return (scale * torch.ones(self.order) / ((1 + shape))) ** degree


class FavardEigenvalues(EigenvalueGenerator):
    required_params = [
        "ard_parameter",  # ard parameter
        "degree",  # exponent for the index term
    ]
    """
    Builds a sequence of eigenvalues that reflect the proffered
    parameterisation from the ard parameter view.

    param scale:
        Scales all eigenvalues - like the kernel 'variance' param
    param shape:
        A vector affecting the shape of the eigenvalue sequence.
        Must be increasing.
    param degree:
        The polynomial degree. Following (Reade, 1992), the 
        smoothness of sample functions (in the limit, so this
        is an approximation) will be described by this degree.
        As parameterised here, a degree of k leads to sample functions
        that would be k times differentiable.


    The eigenvalues here have the form:
        λ_i = \frac{b}{H_{m, 2k} [φ_i(0)φ''_i(0) + (φ'_i(0))^2]i^{2k}}
    """

    def __init__(self, order, f0, df0, d2f0):
        self.order = order
        self.f0 = f0
        self.df0 = df0
        self.d2f0 = d2f0

    def __call__(self, parameters: dict) -> torch.Tensor:
        ard_parameter = parameters["ard_parameter"]  # scalar
        degree = parameters["degree"]

        numbers = list(range(1, self.order + 1))
        harmonics = torch.Tensor([harmonic(m, 2 * degree) for m in numbers])
        basis_term = self.f0 * self.d2f0 + self.df0 ** 2
        basis_term = torch.ones_like(self.f0)
        poly_term = torch.pow(
            torch.Tensor([range(1, self.order + 1)]), 2 * degree
        )
        eigenvalues = ard_parameter / (harmonics * basis_term * poly_term)

        return eigenvalues.squeeze()


if __name__ == "__main__":
    order = 10
    eigengen = SmoothExponentialFasshauer(order)
    terms = 80
    numbers = list(range(1, terms))
    harmonics = [harmonic(m, 1) for m in numbers]
    logs = [math.log(m) for m in numbers]
    mascheronis = [harmonic - log for harmonic, log in zip(harmonics, logs)]
    # print(mascheronis)
    true_mascheronis = [0.57721566] * terms
    # plt.plot(harmonics)
    # plt.plot(mascheronis)
    # plt.plot(true_mascheronis)
    # plt.show()
