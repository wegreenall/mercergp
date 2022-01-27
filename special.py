import matplotlib
import numpy as np

# import scipy.special as sc

# if __name__ == "__main__":
# matplotlib.use("TkAgg")
import math

import matplotlib.pyplot as plt
import torch

from torch.autograd import Function, gradcheck

# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 08:34:01 2019

@author: William Greenall
"""
"""
This contains special functions implemented in pytorch, with a view to keeping
them differentiable.
"""


def lbeta(argument, axis=0):
    # sum of log gammas (i.e. product of gammas)
    numerator = torch.sum(torch.lgamma(argument), axis=axis)
    # log gamma of sum of arguments (i.e. gamma of the sum
    denominator = torch.lgamma(torch.sum(argument, axis=axis))
    return numerator - denominator


def hermite_coefficients(N, whole_matrix=False):
    """
    Returns the physicist's polynomials coefficient for a given order
    """
    a = torch.zeros((N + 2, N + 2))

    a[0, 0] = 1
    a[1, 1] = 2

    for n in range(N + 1):
        for k in range(N + 1):
            if k == 0:
                a[n + 1, k] = -a[n, k + 1]
            else:
                a[n + 1, k] = 2 * a[n, k - 1] - (k + 1) * a[n, k + 1]

    # if whole_matrix
    if whole_matrix:
        return_val = a[:-1, :-1]
    else:
        return_val = a[N, :-1]
    return return_val


"""
def wg_hermval(x, n, prob=True, log=False):
    # get the coefficients
    coefficients = hermite_coefficients(n)  # a vector of length n, i.e.
                   the order of the polynomial
    # signsx = torch.where(x<0, -1*torch.ones(x.shape), torch.ones(x.shape))
    logx = torch.log(torch.abs(x))
    polynomial_terms = torch.Tensor(list(range(n+1))).unsqueeze(1)
    polynoms = polynomial_terms * logx

    # logarithmify and get signs of coefficients

    coefficsigns = torch.where(coefficients<0,
                               -1*torch.ones(coefficients.shape),
                               torch.ones(coefficients.shape))
    polynoms += torch.log(torch.abs(coefficients).unsqueeze(1))
    exponential_polynoms = torch.exp(polynoms)
    summable_polynoms = coefficsigns * exponential_polynoms.T
    # breakpoint()
    hermvals = torch.sum(summable_polynoms, 1)
    # if  not log:
    return hermvals
    # else:
    #     return hermvals, coefficsigns
    """


def hermval(x, c, prob=True):
    """
    This replicates the numpy hermval function for Hermite polynomials but for
    Pytorch tensors.

    truth be told, this doesn't even use any tensor stuff. nevermind.
    :param x:
    :param c:
    :param prob: if True, returns the probabilist's Hermite polynomial;
                 if False, returns the Physicist's.
    :return:
    """
    x2 = x * 2  # for the physicist's version
    # c0 = c[-1] - c1*(2*(nd-1))
    if (x != x).any():
        raise ValueError("x has NaNs!")
    # else:
    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            if prob:
                c0 = c[-i] - c1 * (nd - 1)
                c1 = tmp + c1 * x
            else:
                c0 = c[-i] - c1 * (2 * (nd - 1))
                c1 = tmp + c1 * x2

            # c1 = torch.where(c1!=c1, torch.zeros(c1.shape), c1)
            if (tmp != tmp).any() or (c0 != c0).any() or (c1 != c1).any():
                raise ValueError("The hermval calculations are seeing NaNs.")

    if prob:
        return_val = c0 + c1 * x
    else:
        return_val = c0 + c1 * x2

    return_val = torch.where(
        return_val != return_val, torch.zeros(return_val.shape), return_val
    )

    if (return_val != return_val).any():
        breakpoint()
    return return_val


def hermite_function(x, n, include_constant=True, include_gaussian=True):
    """
    Returns the value of the Hermitian function
    (i.e. the Gauss-Hermitian function constructed from the Hermitian
    polynomials).

    The Hermite polynomials used here are the physicist's polynomials
    if include_constant = False, and include_gaussian=False,
    then the function returns the hermite polynomial of degree
    n evaluated at x. This is useful for the construction of basis functions.
    :return:
    """

    # might want to do all the following in logs for numerical stability

    n_val = torch.Tensor([n])
    if include_constant:
        # constant_component = 1/(math.sqrt((2**n) *\
        # torch.exp(torch.lgamma(n)) * math.sqrt(math.pi)))

        constant_component_log = (
            -(n_val)
            / 2
            * torch.log(
                torch.tensor(
                    2.0,
                )
            )
            - 0.5 * torch.lgamma(n_val + 1)
            - 0.25
            * torch.log(
                torch.tensor(
                    math.pi,
                )
            )
        )
        if (constant_component_log != constant_component_log).any():
            print("constant_component_log has nans!")
            breakpoint()
        constant_component = torch.exp(constant_component_log)
        if (constant_component != constant_component).any():
            print("constant_component has Nans")
            breakpoint()
    else:
        constant_component = 1

    if include_gaussian:
        # gaussian_component_log = -0.5*np.power(x,2)
        gaussian_component = np.exp(-np.power(x, 2) / 2)
        if (gaussian_component != gaussian_component).any():
            print("gaussian component has nans!")
            breakpoint()
    else:
        gaussian_component = 1

    # build the coefficient vector needed to get the right hermite polynomial
    # value based on the way hermval gets it
    zeros_vector = np.zeros([n])
    coeffic_vector = np.concatenate([zeros_vector, np.array([1])])
    hermite_component = hermval(x, coeffic_vector, prob=False)
    # hermite_component = wg_hermval(x, n, prob=False)
    if (hermite_component != hermite_component).any():
        print("hermite component has nans!")
        breakpoint()

    # hermite_result = torch.exp(constant_component_log
    # + gaussian_component_log) * hermite_component
    hermite_result = (
        constant_component * gaussian_component * hermite_component
    )

    # breakpoint()
    return hermite_result


if __name__ == "__main__":
    a = 50
    x = torch.linspace(0.05, 0.99, 1000)
    x.requires_grad = True
    # gammaincinv = GammaIncInv.apply
    # f = gammaincinv(a, x)
    # plt.plot(x.detach().numpy(), f.detach().numpy())
    # print("Function vals:", f)

    # test this
    # print("ABout to do a gradcheck test...")
    # test = gradcheck(gammaincinv, (a, x), eps=1e-6, atol=1e-4)
    # print("Gradcheck test:", test)

    print("Beginning comparisons/plotting")
    # n = 20
    n = 0
    z = torch.linspace(-5, 5, 1000)
    zeros_vector = np.zeros([n])
    coeffic_vector = np.concatenate([zeros_vector, np.array([1])])
    # y0 = hermite_function(z, 0)
    # y1 = hermite_function(z, 1)
    y2 = hermite_function(z, 2, include_constant=False)

    def y2prime(x):
        return (4 * (x ** 2) - 2) * torch.exp(-0.5 * (x ** 2))

    def y3prime(x):
        return (8 * (x ** 3) - 12 * x) * torch.exp(-0.5 * (x ** 2))

    y3 = hermite_function(z, 3, include_constant=False)
    # y3prime = lambda x:     # y3 = hermite_function(z, 3)
    # y4 = hermite_function(z, 4)
    # y5 = hermite_function(z, 5)

    # plt.plot(z, y0)
    # plt.plot(z, y1)
    plt.plot(z.numpy().flatten(), y2.numpy().flatten(), color="red")
    plt.plot(z.numpy().flatten(), y2prime(z).numpy().flatten(), color="black")
    # plt.plot(z, y0+y1+y2+y3+y4+y5)
    plt.show()

    plt.plot(z.numpy().flatten(), y3.numpy().flatten(), color="red")
    plt.plot(z.numpy().flatten(), y3prime(z).numpy().flatten(), color="black")
    # plt.plot(z, y0+y1+y2+y3+y4+y5)
    plt.show()
    # breakpoint()
    # y2 = wg_hermval(x,n)
