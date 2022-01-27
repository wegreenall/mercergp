# basis_functions.py
import math

# import matplotlib
# matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import torch

# from framework import special, utils
from polynomials import chebyshev_first, chebyshev_second
from typing import Callable
import special

"""
This contains implementations of basis functions useful for constructing
kernels according to the Mercer representation theorem.

The basis functions are expected to have signature (x, i, params),
where:

x : input value
i : the degree of the given function (in most cases, the degree of the
polynomial that corresponds to that function)

params:  a dictionary of parameters that contains the appropriate titles etc.
for that set of polynomials.

Default versions or generator functions
can be created here to allow for ease of use.
(i.e. available here and not necessary to hand-tune every time)

Return value must be a torch.Tensor of dimension [n] only;
not [n, 1] as is the inputs vector - this reflects explicitly that the output
would be 1-d (2-d inputs would be of shape [n,2]). The result of this is that
"""


class Basis:
    def __init__(
        self,
        basis_function: Callable,
        dimension: int,
        order: int,
        params: dict,
    ):
        self.dimension = dimension
        self.order = order
        self.basis_function = basis_function
        self.params = params
        """
        :param basis function: the basis function to use in this
                               Basis class - i.e., smooth_exponential_basis.

                               Expected to have signature:
                                   basis_function(x: torch.Tensor,
                                                  deg: int,
                                                  params: dict)


        """
        return

    def get_dimension(self):
        """
        Getter method for the dimension of the model.
        """
        return self.dimension

    def __call__(self, x):
        """
        Returns the whole basis evaluated at an input. The multiplier function
        multiplies the basis - this is used for constructing the derivative
        of the Mercer Smooth Exponential GP.  This is almost certainly not the
        best way to do it - as it would be nice to vectorise it. However this
        is fine for now

        The return shape:

        """
        # check input shape
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        if x.shape[1] != self.dimension:
            raise ValueError(
                "The dimension of x should be {dim} for this,\
                              because that is the dim of the\
                              basis.".format(
                    dim=self.dimension
                )
            )

        # can I tensorise the hstack? In basis function?
        result = torch.hstack(
            [
                self.basis_function(x, deg, self.params)
                for deg in range(self.order)
            ]
        )

        return result

    def get_params(self):
        return self.params


def smooth_exponential_basis(x: torch.Tensor, deg: int, params: dict):
    """
    Returns the hermite function evaluated at the input point x.

    The Hermite function here is that constructed with the Physicist's Hermite
    Polynomial as opposed to the Probabilist's.

    : param x: the input points to evaluate the function at. Should be of
               dimensions [n,d]
    : param deg: degree of polynomial used in construction of func
    : param params: dict of parameters. keys set on case-by-case basis
    """
    # breakpoint()
    epsilon = torch.diag(params["ard_parameter"])  # ε  - of dimension d
    alpha = torch.diag(params["precision_parameter"])  # precision parameter
    # of the measure w.r.t the hermite functions are orthogonal

    sigma = torch.sqrt(params["variance_parameter"])

    # = (1 + (2ε/α)^2 ) ^(1/4)
    beta = torch.pow((1 + torch.pow((2 * epsilon / alpha), 2)), 0.25)

    log_gamma = 0.5 * (
        torch.log(beta)
        - (deg) * torch.log(torch.tensor(2, dtype=torch.float))
        - torch.lgamma(torch.tensor(deg + 1, dtype=torch.float))
    )

    delta_2 = (torch.pow(alpha, 2) / 2) * (torch.pow(beta, 2) - 1)

    multiplicative_term = torch.exp(
        log_gamma - (delta_2 * torch.pow(x, torch.tensor(2)))
    )

    # calculate the Hermite polynomial term
    # remember, deg = n-1
    hermite_term = special.hermite_function(
        alpha * beta * x, deg, include_constant=False, include_gaussian=False
    )

    # for numerical reasons, we can save the log of the absolute to get the
    #  right value,
    # then exponentiate and multiply by a mask of negative values to get the
    # right value.

    abs_hermite_term = torch.log(torch.abs(hermite_term))
    # breakpoint()
    mask = torch.where(
        hermite_term < 0,
        -torch.ones(hermite_term.shape),
        torch.ones(hermite_term.shape),
    )
    phi_d = mask * torch.exp(abs_hermite_term) * multiplicative_term
    phi = sigma * phi_d
    # if deg == 56 or deg == 57:
    #     breakpoint()
    return phi


def smooth_exponential_eigenvalues(deg: int, params: dict):
    """
    Returns the vector of eigenvalues, up to length deg, using the parameters
    provided in params.
    :param deg: the degree up to which the eigenvalues should be computed.
    :param params: a dictionary of parameters whose keys included
    """

    eigenvalues = torch.zeros(deg)

    for i in range(1, deg + 1):
        epsilon = torch.diag(params["ard_parameter"])  # ε  - of dimension d
        alpha = torch.diag(params["precision_parameter"])  # precision
        # of the measure w.r.t the hermite functions are orthogonal

        beta = torch.pow((1 + torch.pow((2 * epsilon / alpha), 2)), 0.25)

        delta_2 = 0.5 * torch.pow(alpha, 2) * (torch.pow(beta, 2) - 1)

        denominator = torch.pow(alpha, 2) + delta_2 + torch.pow(epsilon, 2)

        left_term = alpha / torch.sqrt(denominator)
        right_term = torch.pow((torch.pow(epsilon, 2) / denominator), i - 1)

        lamda_d = left_term * right_term
        eigenvalue = torch.prod(lamda_d, dim=1)
        eigenvalues[i - 1] = eigenvalue
    # breakpoint()
    return eigenvalues


def standard_chebyshev_basis(x: torch.Tensor, deg: int, params: dict):
    """
    Returns a standard Chebyshev basis function, using Chebyshev
    polynomials of the first kind, evaluated at x, that does
    not impose the zero-endpoint condition of Babolian

    :  param x:      input tensor to evaluate the function at
    :  param deg:    the degree of the basis function
    :  param params: dictionary of kernel arguments. Should
                     contain keys 'upper_bound' and 'lower_bound':
                     tensors representing the upper and lower bound
                     over which the function is to be a basis respectively.
    """

    # check that x is of the right dimension:
    # assert len(x.shape) == 2,\
    #    "inputs to basis function should have 2 dimensions."

    if "lower_bound" not in params:
        lower_boundary = torch.tensor(-1, dtype=float)
    else:
        lower_boundary = params["lower_bound"]

    if "upper_bound" not in params:
        upper_boundary = torch.tensor(1, dtype=float)
    else:
        upper_boundary = params["upper_bound"]

    if "chebyshev" not in params:
        # default result: chebyshev polynomials/basis of the second kind
        chebyshev = "second"
    else:
        chebyshev = params["chebyshev"]

    assert chebyshev in {
        "first",
        "second",
    }, 'chebyshev should be either "first" or "second"'

    # Transform to [-1,1] for processing
    z = (2 * x - (upper_boundary + lower_boundary)) / (
        upper_boundary - lower_boundary
    )

    if chebyshev == "first":
        chebyshev_term = chebyshev_first(z, deg)
        # exponent of weight function (1-z**2)
        weight_power = torch.tensor(-0.25)

        # define the normalising constant
        if deg == 0:
            normalising_constant = math.sqrt(2 / math.pi)
        else:
            normalising_constant = 2 / math.sqrt(math.pi)

    elif chebyshev == "second":
        chebyshev_term = chebyshev_second(z, deg)

        # exponent of weight function (1-z**2)
        weight_power = torch.tensor(0.25)

        # define the normalising constant
        normalising_constant = 2 / math.sqrt(
            (upper_boundary - lower_boundary) * (math.pi)
        )

    # define weight function
    weight_term = torch.pow(1 - z ** 2, weight_power)
    if (chebyshev_term != chebyshev_term).any():
        raise ValueError(
            "Chebyshev returning NaNs. Ensure \
it is being evaluated within boundaries."
        )

    return weight_term * chebyshev_term * normalising_constant


def standard_haar_basis(x: torch.Tensor, deg: int, params: dict):
    """
    Returns a standard Chebyshev basis, using Chebyshev
    polynomials of the first kind, that does not impose
    the zero-endpoint condition of Babolian

    :  param x:      input tensor to evaluate the function at
    :  param deg:    the degree of the basis function
    :  param params: dictionary of kernel arguments. Should
                     contain keys 'upper_bound' and 'lower_bound':
                     tensors representing the upper and lower bound
                     over which the function is to be a basis respectively.
    """
    pass
