import torch
import torch.autograd as autograd
import math
from typing import Union, List, Tuple
from abc import ABC
from dataclasses import dataclass
from torchmin import minimize, minimize_constr

# FOR TESTING:
from termcolor import colored
from matplotlib import pyplot as plt

from math import prod


def anti_sum(tensor: torch.Tensor, dim: int):
    """
    Sums over all but one of the dimensions, up to a count of 26 dimensions,
    by using the torch einsum tool

    :param tensor: the tensor to sum over
    :param dim: dimension not to sum over
    """
    args = "abcdefghijklmnopqrstuvqxyz"

    start_argstring = ""
    end_argstring = ""
    dimension_count = len(tensor.shape)
    for d in range(dimension_count):
        start_argstring += args[d]
        # i.e. we're at the last element
        # if d != dimension_count-1:
        #    start_argstring += ','

    start_argstring += " -> "
    end_argstring += args[dim]

    argstring = start_argstring + end_argstring
    result = torch.einsum(argstring, tensor)
    return result


def eigenvalue_reshape(eigenvalue_tensors: torch.Tensor):
    """For a d-length list of tensors of eigenvalues, returns the einsum
    resulting from:
        torch.einsum("na, nb, nc, nd -> nabcd", tensor[:,:,0],
                     tensor[:,:,1],
                     tensor[:,:,2],
                     tensor[:,:,3])
    """
    einsum_string = ""
    used_chars = ""
    i = 0
    if len(eigenvalue_tensors.shape) > 1:
        eigenvalue_tensor_count = eigenvalue_tensors.shape[1]
        for i in range(eigenvalue_tensor_count - 1):
            einsum_string += chr(ord("a") + i) + ","
            used_chars += chr(ord("a") + i)
        einsum_string += (
            chr(ord("a") + i + 1) + "-> " + used_chars + chr(ord("a") + i + 1)
        )
        result = torch.einsum(
            einsum_string,
            *[eigenvalue_tensors[:, i] for i in range(eigenvalue_tensor_count)]
        )
    else:
        # if there aren't more than one dimension, this would like to return
        # the "product" across a zero dimension. this is obviously just the
        # identity.
        result = eigenvalue_tensors
    return result


def harmonic(m, k):
    """
    Returns the generalised harmonic number H(m, k)
    """
    # print("m:", m)
    # print("k:", k)
    terms = [1 / (i**k) for i in range(1, m)]
    harmonics = sum(terms) if m > 1 else 1
    return harmonics


@dataclass
class EigenvalueGeneratorParameter:
    """
    A dataclass for the parameters of an eigenvalue generator.
    """

    variance_parameter: torch.Tensor
    ard_parameter: torch.Tensor
    precision_parameter: torch.Tensor
    noise_parameter: torch.Tensor


@dataclass
class SmoothExponentialFasshauerParameters(EigenvalueGeneratorParameter):
    ard_parameter: torch.Tensor
    smoothness_parameter: torch.Tensor


class EigenvalueGenerator(ABC):
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

    required_parameters: List[str] = []

    def __init__(self, order, dimension=1):
        self.order = order
        self.dimension = dimension

    def __call__(
        self, parameters: Union[dict, EigenvalueGeneratorParameter]
    ) -> torch.Tensor:
        """
        Returns the eigenvalues, up to the order set at initialisation,
        given the dictionary of parameters 'parameters'.
        """
        self.check_params(parameters)
        raise NotImplementedError("Please use a subclass of this class")

    def check_params(
        self, parameters: Union[dict, EigenvalueGeneratorParameter]
    ):
        """
        If the parameters are passed a dictionary, this checks whether all the
        correct keys have been passed;
        if passed an EigenvalueGeneratorParameter, this is automatically
        correct/has its own error checking, so it does not need to be checked.
        """
        if isinstance(parameters, dict):
            for key in self.required_params:
                if key not in parameters:
                    print(
                        "Required parameters not included; please ensure to \
                        include the required parameters: \n"
                    )
                    print([param for param in self.required_parameters])
                    raise ValueError("Missing required parameter!")
        return True

    def derivatives(self, parameters: dict) -> dict:
        """
        Returns the derivatives of the eigenvalues w.r.t the parameters.
        """
        raise NotImplementedError("Please use a subclass of this class")

    def inverse(
        self,
        eigenvalues: torch.Tensor,
        initial_params: Union[List[dict], dict],
    ) -> Union[List[dict], dict]:
        """
        Given a set of eigenvalues, returns a dict containing parameters that
        induced them. This should be identifiable since the parameter space is
        smaller than the sequence of eigenvalues, and we assume a common
        functional form.
        """
        raise NotImplementedError("Please use a subclass of this class")


class SmoothExponentialFasshauer(EigenvalueGenerator):
    required_parameters = ["ard_parameter", "precision_parameter"]

    def _check_parameters(
        self,
        parameters: Union[List[dict], dict],
    ) -> Union[Tuple[dict], List[dict]]:
        """
        if we are passed a list, then it will be for multiple dimensions
        for parameter in self.required_parameters:
        """
        if isinstance(parameters, dict):
            parameters_list = [parameters]
        elif isinstance(parameters, list) or isinstance(parameters, tuple):
            parameters_list = parameters
            if len(parameters) != self.dimension:
                raise ValueError(
                    "The number of parameter dicts passed must match the \
                     dimension parameter"
                )
            for params in parameters_list:
                for param in self.required_parameters:
                    if param not in params:
                        raise ValueError(
                            "The parameter "
                            + param
                            + " is required but not found"
                        )
        return parameters_list

    def __call__(
        self,
        parameters: Union[
            list, dict, Tuple, SmoothExponentialFasshauerParameters
        ],
    ) -> torch.Tensor:
        parameters = self._check_parameters(parameters)

        eigens_tensor = torch.zeros(self.order, len(parameters))
        for d, param_set in enumerate(parameters):
            # for d in range(self.dimension):
            eigens = self._smooth_exponential_eigenvalues_fasshauer(
                param_set
            )  # m x dim
            try:
                eigens_tensor[:, d] = eigens
            except RuntimeError as e:
                print(e)
                breakpoint()
        product_eigens = eigenvalue_reshape(eigens_tensor)
        result = torch.reshape(
            product_eigens, (self.order ** len(parameters),)
        )
        return result

    def derivatives(
        self, parameters: Union[List[dict], dict]
    ) -> Union[List[dict], dict]:
        """
        Returns the derivatives of the eigenvalues w.r.t the parameters.

        The parameters are passed in as a dictionary, and we want to
        return, for each parameter, the derivative of the eigenvalues
        w.r.t that parameter. Each element will be a vector of gradients
        with respect to the given parameter.
        """
        # convert from dict to list so we can iterate over the dimensions, even
        # if there is only one
        parameters = self._check_parameters(parameters)

        vector_gradients_list = []
        # other_eigen_term = prod(self(parameters))
        for params in parameters:
            vector_gradients: dict = {}
            # ard_parameter
            vector_gradients["ard_parameter"] = self._ard_parameter_derivative(
                params
            )

            # precision parameter
            vector_gradients[
                "precision_parameter"
            ] = self._precision_parameter_derivative(params)

            # since we pass in all of the dictionary, we need to also
            # set the "noise_parameter" gradient to zeroes
            vector_gradients["noise_parameter"] = torch.zeros(self.order)

            # variance parameter
            vector_gradients[
                "variance_parameter"
            ] = self._variance_parameter_derivative(params)
            vector_gradients_list.append(vector_gradients)
        return vector_gradients_list

    def inverse(
        self,
        target_eigenvalues: torch.Tensor,
        initial_params: Union[List[dict], dict],
    ) -> dict:
        """
        Given a set of eigenvalues, returns a dict containing parameters that
        induced them. This should be identifiable since the parameter space is
        smaller than the sequence of eigenvalues, and we assume a common
        functional form.

        Furthermore, the precision parameter is not identifiable from the
        ard_parameter, so we leave it fixed (arbitrarily).
        """
        initial_params = self._check_parameters(initial_params)
        guess_params = initial_params.copy()  # defined outside the loop...

        # initialise the loop variables
        suboptimal = True
        counter = 0
        ard_learning_rate = 1
        variance_learning_rate = 1
        while suboptimal:
            # get the derivatives
            derivatives = self.derivatives(
                guess_params
            )  # will be a list of dicts of parameter derivatives
            eigens = self(guess_params)
            stopped_changing_variance = True
            stopped_changing_ard = True

            # for each dimension's set of parameters,
            eigen_prod_term = prod(
                [self(param_set) for param_set in guess_params]
            )

            # print("before loop:", guess_params)

            # torch.reshape(
            # target_eigenvalues, (self.order, self.order)
            # )  # m x m
            for param_set, derivative_set in zip(guess_params, derivatives):
                sum_term = self._get_sum_term(
                    eigens, target_eigenvalues, guess_params, param_set
                )
                # print("sign of sum_term:", torch.sign(sum_term))

                # calculate each updated parameter
                new_ard_parameter = param_set[
                    "ard_parameter"
                ] - ard_learning_rate * (
                    sum_term @ derivative_set["ard_parameter"]
                )

                new_variance_parameter = param_set[
                    "variance_parameter"
                ] - variance_learning_rate * (
                    sum_term @ derivative_set["variance_parameter"]
                )

                # check if they stopped changing
                stopped_changing_ard &= torch.allclose(
                    new_ard_parameter, param_set["ard_parameter"]
                )
                stopped_changing_variance &= torch.allclose(
                    new_variance_parameter,
                    param_set["variance_parameter"],
                )

                if stopped_changing_ard and stopped_changing_variance:
                    print(colored("Parameter values stopped changing", "blue"))

                # update the actual parameters
                param_set["ard_parameter"] = new_ard_parameter
                param_set["variance_parameter"] = new_variance_parameter

            distance = torch.norm(target_eigenvalues - self(guess_params))
            if counter % 200 == 0:
                print("distance:", distance)

            suboptimal = distance > 1e-6 and not (
                stopped_changing_ard and stopped_changing_variance
            )
            counter += 1

        # if the dimension is one, don't return a list,
        # just leave it as is!
        if self.dimension == 1:
            return guess_params[0]
        return guess_params

    def _get_sum_term(
        self,
        eigens: torch.Tensor,
        target_eigens: torch.Tensor,
        parameters: Union[List[dict], dict],
        current_params: dict,
    ):
        """
        Calculates the sum term for the inverse method.

        The norm between the true eigens and the current eigens
        is written:
                Σ_k (λ_k - λ*_k)^2

        Taking the derivative, we get:
                Σ_κ 2(λ_k - λ*_k) * dλ_k/dθ
                     or, in e.g. 2-d
                Σ_i Σ_j 2(λ_iλ_j - λ*_ij) * dλ_i/dθ * λ_j

                            Δλ'S
        where:
            - Δλ = dλ_k/dθ (as a vector of partial derivatives)
            - S = Σ_j 2(λ_i λ_j - λ*_ij) λ_j

        This function returns S.
        """
        eigens_reshaped = torch.reshape(eigens, [self.order] * len(parameters))
        target_eigens_reshaped = torch.reshape(
            target_eigens, [self.order] * len(parameters)
        )

        matrix_eigens = eigens_reshaped - target_eigens_reshaped
        try:
            # print("current params:", current_params)
            # print("param_set", parameters)
            eigen_prod_term = prod(
                [self(param_set) for param_set in parameters]
            ) / self(current_params)

        except RuntimeError as e:
            print(e)
        eigen_vals = [self(param_set) for param_set in parameters]
        current_eigs = self(current_params)
        # breakpoint()
        sum_term = anti_sum((2 * matrix_eigens * eigen_prod_term), 0)
        return sum_term

    def _smooth_exponential_eigenvalues_fasshauer(self, parameters: dict):
        if isinstance(parameters, dict):
            return parameters[
                "variance_parameter"
            ] * self._raw_smooth_exponential_eigenvalues_fasshauer(parameters)
        elif isinstance(parameters, SmoothExponentialFasshauerParameters):
            return (
                parameters.variance_parameter
                * self._raw_smooth_exponential_eigenvalues_fasshauer(
                    parameters
                )
            )
        else:
            raise ValueError(
                "The parameters passed are not of the correct type. Please pass either\
                a dictionary or a SmoothExponentialFasshauerParameters object."
            )

    def _raw_smooth_exponential_eigenvalues_fasshauer(self, parameters: dict):
        """
        If in one dimension, returns the vector of eigenvalues, up to length
        order, using the parameters provided in params. Otherwise, returns a
        matrix of [order, dimension] per-dimension eigenvalue vectors as
        columns. The calling EigenvalueGenerator class is then expected to
        convert these to tensor product length i.e. to become
        [order ** dimension].
        parameters:
            - order: the degree up to which the eigenvalues should be computed.
            - params: a dictionary of parameters whose keys included
        """
        if isinstance(parameters, dict):
            ard_parameter = parameters["ard_parameter"]
            precision_parameter = parameters["precision_parameter"]
        elif isinstance(parameters, SmoothExponentialFasshauerParameters):
            ard_parameter = parameters.ard_parameter
            precision_parameter = parameters.precision_parameter
        b = torch.diag(ard_parameter)  # ε  - of dimension d
        a = torch.diag(precision_parameter)  # precision
        c = torch.sqrt(a**2 + 2 * a * b)
        left_term = torch.sqrt(2 * a / (a + b + c))
        right_term = b / (a + b + c)

        # construct the vector
        exponents = (
            torch.linspace(0, self.order - 1, self.order).repeat(len(b), 1).t()
        )
        eigenvalues = left_term * torch.pow(right_term, exponents)
        return eigenvalues.squeeze()

    def _ard_parameter_derivative(self, parameters: dict):
        """
        Returns a vector of gradients of the eigenvalues w.r.t
        the ard parameter.
        """
        if isinstance(parameters, dict):
            sigma = parameters["variance_parameter"]
            b = torch.diag(parameters["ard_parameter"])  # ε  - of dimension d
            a = torch.diag(parameters["precision_parameter"])  # precision
        elif isinstance(parameters, SmoothExponentialFasshauerParameters):
            sigma = parameters.variance_parameter
            b = torch.diag(parameters.ard_parameter)
            a = torch.diag(parameters.precision_parameter)
        c = a**2 + 2 * a * b
        # eigenvalue_derivatives = torch.zeros(self.order)

        index_vector = torch.linspace(0, self.order - 1, self.order)

        # construct the derivatives of the eigenvalues w.r.t the ard
        # parameter

        denominator_derivative = 1 + 0.5 * torch.pow(
            a**2 + 2 * a * b, -0.5
        ) * (2 * a)

        L = torch.sqrt(2 * a / (a + b + c))
        # dL/da
        dL = (
            (0.5 * torch.pow((2 * a / (a + b + c)), -0.5))
            * (-2 * a * denominator_derivative)
            / (a + b + c) ** 2
        )
        # R
        R = (b / (a + b + c)) ** index_vector

        # dR
        dR = (
            index_vector
            * (torch.pow((b / (a + b + c)), (index_vector - 1)))
            * ((a + b + c) - b * denominator_derivative)
            / (a + b + c) ** 2
        )

        eigenvalue_derivatives = sigma * (dL * R + dR * L)
        return eigenvalue_derivatives.squeeze()

    def _variance_parameter_derivative(self, parameters: dict):
        return self._raw_smooth_exponential_eigenvalues_fasshauer(parameters)

    def _precision_parameter_derivative(self, parameters: dict):
        """
        Returns a vector of gradients of the eigenvalues w.r.t
        the precision parameter.
        """
        if isinstance(parameters, dict):
            b = torch.diag(parameters["ard_parameter"])
            a = torch.diag(parameters["precision_parameter"])
        elif isinstance(parameters, SmoothExponentialFasshauerParameters):
            b = torch.diag(parameters.ard_parameter)
            a = torch.diag(parameters.precision_parameter)

        b = torch.diag(parameters["ard_parameter"])  # ε  - of dimension d
        a = torch.diag(parameters["precision_parameter"])  # precision
        c = torch.sqrt(a**2 + 2 * a * b)
        eigenvalue_derivatives = torch.zeros(self.order)

        # check whether it works like this or not (although i can't see why!)
        # index_vector = torch.linspace(0, self.order - 1, self.order)
        index_vector_alternative = torch.linspace(1, self.order, self.order)
        index_vector = index_vector_alternative

        c_derivative = 0.5 * (1 / c) * (2 * a + 2 * b)
        denominator_derivative = 1 + c_derivative

        # L
        L = torch.sqrt(2 * a / (a + b + c))

        # R
        R = (b / (a + b + c)) ** index_vector

        # dL/da
        dL = (
            (0.5 * 1 / L)
            * (2 * (a + b + c) - 2 * a * denominator_derivative)
            / (a + b + c) ** 2
        )

        # dR/da
        dR = (
            index_vector
            * (torch.pow((b / (a + b + c)), (index_vector - 1)))
            * (-b * denominator_derivative)
            / (a + b + c) ** 2
        )

        eigenvalue_derivatives = dL * R + dR * L
        # assert (term_1 == dL * R).all()
        # assert (term_2 == dR * L).all()
        # print("precision derivs:", colored(eigenvalue_derivatives, "red"))
        # eigenvalue_derivatives = torch.zeros(eigenvalue_derivatives.shape)
        return eigenvalue_derivatives.squeeze()


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
        return (
            scale
            * torch.ones(self.order)
            / ((torch.linspace(0, self.order - 1, self.order) + shape))
        ) ** degree

    def derivatives(self, parameters: dict) -> dict:
        """
        Returns the derivatives of the eigenvalues w.r.t the parameters.
        """
        scale_derivatives = self._scale_derivatives(parameters)
        shape_derivatives = self._shape_derivatives(parameters)
        # degree_derivatives = self._degree_derivatives(parameters)
        parameter_derivatives = parameters.copy()

        for param in parameter_derivatives:
            parameter_derivatives[param] = torch.zeros(self.order)

        parameter_derivatives["scale"] = scale_derivatives
        parameter_derivatives["shape"] = shape_derivatives
        """
        Whilst we can technically get the degree derivative, it is not likely
        that this is a good parameter to try to choose. Essentially, the
        degree parameter is a way of controlling the smoothness of the
        sample functions; and constitutes an object of prior information,
        that cannot readily be selected or inferred from data given that it is
        generally not a well-posed problem.
        """
        # parameter_derivatives["degree"] = degree_derivatives

        return parameter_derivatives

    def _scale_derivatives(self, parameters: dict) -> dict:
        """
        Calculates the derivative of the eigenvalues w.r.t the scale parameter.

        output shape: [order]
        """
        # scale = parameters["scale"]
        shape = parameters["shape"]
        degree = parameters["degree"]
        return (
            torch.ones(self.order)
            / ((torch.linspace(0, self.order - 1, self.order) + shape))
        ) ** degree

    def _shape_derivatives(self, parameters: dict) -> dict:
        """
        Calculates the derivative of the eigenvalues w.r.t the shape parameter.

        output shape: [order]
        """
        scale = parameters["scale"]
        shape = parameters["shape"]
        degree = parameters["degree"]
        return (
            -degree
            * scale
            * torch.ones(self.order)
            / ((torch.linspace(0, self.order - 1, self.order) + shape))
        ) ** (degree + 1)

    def _degree_derivatives(self, parameters: dict) -> torch.Tensor:
        """
        Calculates the derivative of the eigenvalues w.r.t the degree
        parameter.

        output shape: [order]
        """
        shape = parameters["shape"]
        degree = parameters["degree"]
        return (
            torch.log(1 + shape)
            * torch.ones(self.order)
            / ((torch.linspace(0, self.order - 1, self.order) + shape))
        ) ** degree


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

    def __init__(self, order, basis):
        """
        When initialising the Favard eigenvalues, we
        need to calculate the various terms associated with
        the derivative of the basis funciton, etc.

        To calculate them, we accept a Basis type, and evaluate
        it at a tensor of 0 that has its gradient required.
        """
        self.order = order
        f0, df0, d2f0 = self._get_basis_terms(basis)
        self.f0 = f0  # φ_i(0)
        self.df0 = df0  # φ'_i(0)
        self.d2f0 = d2f0  # φ''_i(0)

    def _get_basis_terms(
        self, basis
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given a basis functions, uses autograd to
        calculate its derivatives at 0 in order to calculate the terms
        """
        x = torch.Tensor([0.0])
        x.requires_grad = True
        f0 = basis(x)
        df0 = autograd.functional.jacobian(basis, x).squeeze(2)
        d2f0 = f0.clone()  # autograd.grad(df0, x)[0]
        return f0, df0, d2f0

    def __call__(self, parameters: dict) -> torch.Tensor:
        ard_parameter = parameters["ard_parameter"]  # scalar
        degree = parameters["degree"]

        numbers = list(range(1, self.order + 1))
        harmonics = torch.Tensor([harmonic(m, 2 * degree) for m in numbers])
        basis_term = self.f0 * self.d2f0 + self.df0**2
        basis_term = torch.ones_like(self.f0)
        poly_term = torch.pow(
            torch.Tensor([range(1, self.order + 1)]), 2 * degree
        )
        eigenvalues = ard_parameter / (harmonics * basis_term * poly_term)

        return eigenvalues.squeeze()

    def derivatives(self, parameters: dict) -> dict:
        """
        Returns the derivatives of the eigenvalues vector w.r.t each of the
        relevant parameters.
        """
        raise NotImplementedError(
            "Not yet set up derivatives for potential\
                                  favard eigenvalues"
        )

        ard_parameter = parameters["ard_parameter"]  # scalar
        degree = parameters["degree"]

        numbers = list(range(1, self.order + 1))
        harmonics = torch.Tensor([harmonic(m, 2 * degree) for m in numbers])
        basis_term = self.f0 * self.d2f0 + self.df0**2
        basis_term = torch.ones_like(self.f0)
        poly_term = torch.pow(
            torch.Tensor([range(1, self.order + 1)]), 2 * degree
        )
        eigenvalues = ard_parameter / (harmonics * basis_term * poly_term)
        return eigenvalues.squeeze()


if __name__ == "__main__":
    test_inverse = False
    test_inverse_multivariate = True
    test_harmonics = False
    order = 10
    dimension = 1
    if test_harmonics:
        eigengen = SmoothExponentialFasshauer(order, dimension)
        terms = 80
        numbers = list(range(1, terms))
        harmonics = [harmonic(m, 1) for m in numbers]
        logs = [math.log(m) for m in numbers]
        mascheronis = [
            harmonic - log for harmonic, log in zip(harmonics, logs)
        ]
        # print(mascheronis)
        true_mascheronis = [0.57721566] * terms
    # plt.plot(harmonics)
    # plt.plot(mascheronis)
    # plt.plot(true_mascheronis)
    # plt.show()
    if test_inverse:
        eigenvalue_generator = SmoothExponentialFasshauer(order, dimension)
        initial_params = {
            "ard_parameter": torch.Tensor([1.0]),
            "precision_parameter": torch.Tensor([1.0]),
            "variance_parameter": torch.Tensor([1.0]),
        }
        initial_guess_params = {
            "ard_parameter": torch.Tensor([5.0]),
            "precision_parameter": torch.Tensor([1.0]),
            "variance_parameter": torch.Tensor([5.0]),
        }
        eigens = eigenvalue_generator(initial_params)
        params = eigenvalue_generator.inverse(eigens, initial_guess_params)
        print(
            "ard:",
            params["ard_parameter"],
            initial_params["ard_parameter"],
        )
        print(
            "precision:",
            params["precision_parameter"],
            initial_params["precision_parameter"],
        )
        print(
            "variance:",
            params["variance_parameter"],
            initial_params["variance_parameter"],
        )
        print(
            torch.allclose(
                params["precision_parameter"],
                initial_params["precision_parameter"],
            )
        )
        print(
            torch.allclose(
                params["ard_parameter"],
                initial_params["ard_parameter"],
            )
        )
        print(
            torch.allclose(
                params["variance_parameter"],
                initial_params["variance_parameter"],
            )
        )
    if test_inverse_multivariate:
        dimension = 2
        eigenvalue_generator = SmoothExponentialFasshauer(order, dimension)
        initial_params = [
            {
                "ard_parameter": torch.Tensor([1.0]),
                "precision_parameter": torch.Tensor([0.1]),
                "variance_parameter": torch.Tensor([1.0]),
            }
        ] * 2
        initial_guess_params = [
            {
                "ard_parameter": torch.Tensor([5.0]),
                "precision_parameter": torch.Tensor([0.1]),
                "variance_parameter": torch.Tensor([5.0]),
            }
        ] * 2
        eigens = eigenvalue_generator(initial_params)  # get the eigens...
        print("True eigenvalues:", eigens)
        # plt.plot(eigens)
        # plt.show()
        plt.imshow(eigens.reshape((order, order)))
        plt.show()
        params = eigenvalue_generator.inverse(eigens, initial_guess_params)
        print("First dimension:")
        print(
            "ard:",
            params[0]["ard_parameter"],
            initial_params[0]["ard_parameter"],
        )
        print(
            "precision:",
            params[0]["precision_parameter"],
            initial_params[0]["precision_parameter"],
        )
        print(
            "variance:",
            params[0]["variance_parameter"],
            initial_params[0]["variance_parameter"],
        )
        print("Second dimension:")
        print(
            "ard:",
            params[1]["ard_parameter"],
            initial_params[1]["ard_parameter"],
        )
        print(
            "precision:",
            params[1]["precision_parameter"],
            initial_params[1]["precision_parameter"],
        )
        print(
            "variance:",
            params[1]["variance_parameter"],
            initial_params[1]["variance_parameter"],
        )
        # print(
        # torch.allclose(
        # params[1]["precision_parameter"],
        # initial_params[1]["precision_parameter"],
        # )
        # )
        print(
            torch.allclose(
                params[1]["ard_parameter"],
                initial_params[1]["ard_parameter"],
            )
        )
        print(
            torch.allclose(
                params[1]["variance_parameter"],
                initial_params[1]["variance_parameter"],
            )
        )
