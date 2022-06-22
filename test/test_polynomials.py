import torch
import unittest
from ortho.polynomials import (
    chebyshev_first,
    chebyshev_second,
    generalised_laguerre,
)


class TestChebyshevPolynomials(unittest.TestCase):
    def setUp(self):
        self.x = torch.linspace(-1 + 0.001, 1 - 0.001, 1000)
        self.chebyshev_polynomials_first = [
            lambda x: torch.ones(x.shape),
            lambda x: x,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x,
        ]

        self.chebyshev_polynomials_second = [
            lambda x: torch.ones(x.shape),
            lambda x: 2 * x,
            lambda x: 4 * x ** 2 - 1,
            lambda x: 8 * x ** 3 - 4 * x,
            lambda x: 16 * x ** 4 - 12 * x ** 2 + 1,
            lambda x: 32 * x ** 5 - 32 * x ** 3 + 6 * x,
        ]

    def test_chebyshev_first(self):
        # tests whether calculation of the chebyshev bases gets right answer
        for i in range(6):
            with self.subTest(i=i):
                values = chebyshev_first(self.x, i)
                test_values = torch.Tensor(
                    self.chebyshev_polynomials_first[i](self.x)
                )
                self.assertTrue(abs(values[10] - test_values[10]) < 1e-10)
            return

    def test_chebyshev_second(self):
        # tests whether calculation of the chebyshev bases gets right answer
        for i in range(6):
            with self.subTest(i=i):
                values = chebyshev_second(self.x, i)
                test_values = torch.Tensor(
                    self.chebyshev_polynomials_second[i](self.x)
                )
                self.assertTrue(abs(values[10] - test_values[10]) < 1e-10)
            return


class TestLaguerrePolynomials(unittest.TestCase):
    def setUp(self):
        # setup in here
        self.params = {"alpha": 0}
        self.x = torch.linspace(0 + 0.001, 5 - 0.001, 1000)

        self.laguerre_polynomials = [
            lambda x: torch.ones(x.shape),  # 1
            lambda x: -x + 1,  # 1 - x
            lambda x: (1 / 2) * (x ** 2 - 4 * x + 2),
            lambda x: (1 / 6) * (-(x ** 3) + 9 * x ** 2 - 18 * x + 6),
            lambda x: (1 / 24)
            * (x ** 4 - 16 * x ** 3 + 72 * x ** 2 - 96 * x + 24),
            lambda x: (1 / 120)
            * (
                -(x ** 5)
                + 25 * x ** 4
                - 200 * x ** 3
                + 600 * x ** 2
                - 600 * x
                + 120
            ),
        ]

    def test_laguerre_polynomials(self):
        for i in range(6):
            with self.subTest(i=i):
                values = generalised_laguerre(self.x, i, self.params)
                test_values = torch.Tensor(
                    self.laguerre_polynomials[i](self.x)
                )
                self.assertTrue(abs(values[10] - test_values[10]) < 1e-10)
            return
        return


if __name__ == "__main__":
    unittest.main()
