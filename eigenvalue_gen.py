import torch
from ortho.basis_functions import smooth_exponential_eigenvalues_fasshauer


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


class Polynomial(EigenvalueGenerator):
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
        scale = parameters["scale"]
        shape = parameters["shape"]
        degree = parameters["degree"]
        return (scale * torch.ones(self.order) / ((1 + shape))) ** degree


if __name__ == "__main__":
    order = 10
    eigengen = SmoothExponentialFasshauer(order)
