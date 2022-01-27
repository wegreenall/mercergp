import math
import unittest

import torch

from basis_functions import (
    Basis,
    smooth_exponential_basis,
    standard_chebyshev_basis,
)
from special import hermval


class TestBasisClass(unittest.TestCase):
    def setUp(self):
        bases = standard_chebyshev_basis
        self.dimension = 1
        self.max_degree = 10
        params_1 = {
            "upper_bound": torch.tensor(10.0, dtype=float),
            "lower_bound": torch.tensor(0.0, dtype=float),
        }
        self.basis = Basis(bases, self.dimension, self.max_degree, params_1)

    def test_shape_flat(self):
        N = 100
        x = torch.linspace(0.1, 10 - 0.1, N)
        y = self.basis(x)  # test output, should be a 10^2 matrix
        self.assertEqual(y.shape, torch.Size([N, self.max_degree]))
        pass

    def test_shape(self):
        N = 100
        x = torch.linspace(0.1, 10 - 0.1, N).unsqueeze(1)
        y = self.basis(x)  # test output, should be a 10^2 matrix
        self.assertEqual(y.shape, torch.Size([N, self.max_degree]))
        pass


class TestHermitePolynomials(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.linspace(0, 10, 1000)
        self.subtest_count = 5
        self.prob_polynomials = [
            lambda x: 1,
            lambda x: x,
            lambda x: x ** 2 - 1,
            lambda x: x ** 3 - 3 * x,
            lambda x: x ** 4 - 6 * x ** 2 + 3,
            lambda x: x ** 5 - 10 * x ** 3 + 15 * x,
        ]
        self.phys_polynomials = [
            lambda x: 1,
            lambda x: 2 * x,
            lambda x: 4 * x ** 2 - 2,
            lambda x: 8 * x ** 3 - 12 * x,
            lambda x: 16 * x ** 4 - 48 * x ** 2 + 12,
            lambda x: 32 * x ** 5 - 160 * x ** 3 + 120 * x,
        ]

    def test_probability_polynomials(self):
        # compares the correct polynomials with the constructed hermite poly.
        for i in range(1, self.subtest_count):
            with self.subTest(i=i):
                zeros_vector = torch.zeros([i])
                c = torch.cat(
                    [
                        zeros_vector,
                        torch.Tensor(
                            [
                                1,
                            ]
                        ),
                    ]
                )
                values = hermval(self.inputs, c, prob=True)
                test_values = self.prob_polynomials[i](self.inputs)
                self.assertTrue(abs(values[10] - test_values[10]) < 1e-14)
            return

    def test_physicists_polynomials(self):
        # compares the correct polynomials with the constructed hermite poly.
        for i in range(1, self.subtest_count):
            with self.subTest(i=i):
                zeros_vector = torch.zeros([i])
                c = torch.cat(
                    [
                        zeros_vector,
                        torch.Tensor(
                            [
                                1,
                            ]
                        ),
                    ]
                )
                values = hermval(self.inputs, c, prob=False)
                test_values = self.phys_polynomials[i](self.inputs)
                self.assertTrue(abs(values[10] - test_values[10]) < 1e-14)
            return


class TestChebyshevBasis(unittest.TestCase):
    def setUp(self):
        self.lb = 0
        self.ub = 4
        self.kernel_params = {
            "upper_bound": torch.tensor(self.ub, dtype=float),
            "lower_bound": torch.tensor(self.lb, dtype=float),
        }

        self.subtest_count = 5

        self.x = torch.linspace(self.lb, self.ub - 0.01, 1000)
        self.inputs = (2 * self.x - (self.ub + self.lb)) / (self.ub - self.lb)
        # weight_function = (
        # 2
        # / math.sqrt((ub - lb) * (math.pi))
        # * ((1 - (self.input ** 2)) ** (0.25))
        # )

        self.chebyshev_polynomials = [
            lambda x: 1,
            lambda x: 2 * x,
            lambda x: 4 * x ** 2 - 1,
            lambda x: 8 * x ** 3 - 4 * x,
            lambda x: 16 * x ** 4 - 12 * x ** 2 + 1,
            lambda x: 32 * x ** 5 - 32 * x ** 3 + 6 * x,
        ]
        self.epsilon = 0.00000001
        pass

    def weight_function(self, x, lb, ub):
        return 2 / math.sqrt((ub - lb) * (math.pi)) * ((1 - x ** 2) ** 0.25)

    def test_chebyshev_basis_correctness(self):
        # tests whether calculation of the chebyshev bases gets right answer
        for i in range(1, self.subtest_count):
            with self.subTest(i=i):
                values = standard_chebyshev_basis(
                    self.x, i, self.kernel_params
                )
                test_values = self.chebyshev_polynomials[i](
                    self.inputs
                ) * self.weight_function(self.inputs, self.lb, self.ub)
                self.assertTrue(abs(values[10] - test_values[10]) < 1e-10)
            return

    def test_orthonormality(self):
        """
        To check the integral of the function, remember
        that the function always maps its upper and lower bounds into
        [-1, 1] via the transformation
        z = 2x - (b+a)/(b-a)

        i.e. the integral:
        # int_a^b f_i(x)^2 dx = int_a^b phi((2*x - (b+a))/(b-a))dx

        Defining z = (2*x - (b+a))/(b-a) we get:

        # int_{z|x = a}^{z|x=b}  phi(z)dx/dz  dz
        # = ((b-a)/2) *  int_{-1}^1 phi(z) dz
        # = (b-a)/2 * 1 since phi orthonormal by construction.

        The integral of f_i(x)^2
        can be calculate by taking the mean of f_i(x)^2 and multiplying by
        (b-a) since if you divide the integral of this by (b-a)/2 you should
        get 1, the result is that we can compare mean(f(sample))2 with 1
        to see if the integral is correct, REGARDLESS OF THE VALUE OF -1, 1.

        The basis is therefore orthonormal WHEN we also multiply by root 2.
        Hence the normalising term being 2 / sqrt(π) rather than
        sqrt(2/π) which appears to be implied by the standard formula
        """
        lb = 1
        ub = 12
        # x = torch.linspace(lb, ub, 4000)

        kernel_params = {
            "upper_bound": torch.tensor(ub, dtype=float),
            "lower_bound": torch.tensor(lb, dtype=float),
        }
        n = 20

        # unif = torch.distributions.Uniform(lb, ub)
        N = 1000000
        # sample = unif.sample([N])
        sample = torch.quasirandom.SobolEngine(1).draw(N) * (
            ub - 0.000001 - (lb + 0.000001)
        ) + (lb + 0.000001)

        # plt.plot(sample)
        # plt.show()
        # breakpoint()
        func_means = torch.zeros(n)
        for i in range(0, n):
            func_sample = (
                standard_chebyshev_basis(
                    sample,
                    i,
                    kernel_params,
                )
                ** 2
            )
            # duh, this is always the integral:
            func_mean = torch.mean(func_sample) * (ub - lb)
            func_means[i] = func_mean
        # print(torch.mean(func_means))
        # breakpoint()
        self.assertTrue(
            (torch.abs(torch.mean(func_means) - 1) < 0.00001).all()
        )
        # print("integral of square basis function: ", func_mean)


class TestHermiteBasis(unittest.TestCase):
    def setUp(self):
        l_se = torch.Tensor([[1]])  # length scale
        sigma_se = torch.Tensor([2])  # variance
        prec = torch.Tensor([1])
        sigma_e = torch.Tensor([1])  # noise parameter
        se_kernel_args = {
            "ard_parameter": l_se,
            "variance_parameter": sigma_se,
            "noise_parameter": sigma_e,
            "precision_parameter": prec,
        }

        # parameters
        self.inputs = torch.linspace(-10, 10, 1000)
        self.subtest_count = 5
        self.params = se_kernel_args
        pass

    def test_hermite_basis(self):
        for deg in range(1, self.subtest_count):
            with self.subTest(i=deg):
                # get the test values
                test_values = smooth_exponential_basis(
                    self.inputs, deg, self.params
                )

                # compare them to what they should be
                zeros_vector = torch.zeros([deg])
                c = torch.cat(
                    [
                        zeros_vector,
                        torch.Tensor(
                            [
                                1,
                            ]
                        ),
                    ]
                )

                # scalars:
                #   α = 1,
                #   β = (1 + 4(ε/α)**2)**(0.25),
                #   γ = √(β / 2^n n!),
                #   δ = α^2 (β^2 - 1) / 2,
                #   ε = 1,
                #   σ = 1
                alpha = self.params["ard_parameter"]
                beta_part = torch.pow(torch.tensor(5), torch.tensor(0.25))
                delta_part = alpha * (beta_part ** 2 - 1) / 2

                constant_part = torch.sqrt(beta_part) / math.sqrt(
                    (2 ** (deg)) * math.factorial(deg)
                )

                exponential_part = torch.exp(
                    -delta_part * torch.pow(self.inputs, torch.tensor(2))
                )

                hermite_part = hermval(self.inputs * beta_part, c, prob=False)
                values = (
                    torch.sqrt(self.params["variance_parameter"])
                    * constant_part
                    * exponential_part
                    * hermite_part
                )
                self.assertTrue((torch.abs(values - test_values) < 1e-6).all())
        return

        def test_orthonormality(self):
            pass


if __name__ == "__main__":
    unittest.main()
