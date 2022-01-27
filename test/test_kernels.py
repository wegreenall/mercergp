# test_kernels.py
import unittest
import torch

import kernel as K
from ortho.basis_functions import (
    Basis,
    smooth_exponential_basis,
    smooth_exponential_eigenvalues,
)


class TestStationaryKernels(unittest.TestCase):
    def setUp(self):
        self.kernel_args = {
            "noise_parameter": torch.Tensor([1]),
            "variance_parameter": torch.Tensor([1]),
            "ard_parameter": torch.Tensor([[1]]),
        }

        self.kernel = K.SmoothExponentialKernel(self.kernel_args)

        pass

    def test_flat(self):
        # test how calling the kernel on various shapes works.
        test_inputs = torch.linspace(0, 1, 10)
        result = self.kernel(test_inputs, test_inputs)
        self.assertEqual(result.shape, torch.Size([10, 10]))

    def test_1d(self):
        # test how calling the kernel on various shapes works.
        test_inputs = torch.linspace(0, 1, 10).unsqueeze(1)
        result = self.kernel(test_inputs, test_inputs)
        self.assertEqual(result.shape, torch.Size([10, 10]))

    def test_2d(self):
        test_inputs = torch.Tensor(
            list(zip(torch.linspace(0, 1, 10), torch.linspace(1, 2, 10)))
        )
        # breakpoint()
        result = self.kernel(test_inputs, test_inputs)
        self.assertEqual(result.shape, torch.Size([10, 10]))

    def test_inverse(self):
        test_inputs = torch.linspace(0, 1, 10).unsqueeze(1)
        case_1 = self.kernel(test_inputs, test_inputs)
        case_1 += self.kernel_args["noise_parameter"] ** 2 * torch.eye(
            test_inputs.shape[0]
        )
        test_1 = torch.inverse(case_1)

        test_2 = self.kernel.kernel_inverse(test_inputs)
        self.assertTrue((test_1 == test_2).all())


class TestMercerKernel(unittest.TestCase):
    def setUp(self):
        self.mercer_kernel_args = {
            "noise_parameter": torch.Tensor([1]),
            "precision_parameter": torch.Tensor([1]),
            "ard_parameter": torch.Tensor([[1]]),
            "variance_parameter": torch.Tensor([1]),
        }
        self.order = 10
        self.basis = Basis(
            smooth_exponential_basis, 1, self.order, self.mercer_kernel_args
        )

        self.eigenvalues = smooth_exponential_eigenvalues(
            self.order, self.mercer_kernel_args
        )

        self.kernel = K.MercerKernel(
            self.order, self.basis, self.eigenvalues, self.mercer_kernel_args
        )
        pass

    def test_flat(self):
        # test how calling the kernel on various shapes works.
        test_inputs = torch.linspace(0, 1, 10)
        result = self.kernel(test_inputs, test_inputs)
        self.assertEqual(result.shape, torch.Size([10, 10]))

    def test_1d(self):
        # test how calling the kernel on various shapes works.
        test_inputs = torch.linspace(0, 1, 10).unsqueeze(1)
        result = self.kernel(test_inputs, test_inputs)
        self.assertEqual(result.shape, torch.Size([10, 10]))

    @unittest.skip("Not Implemented for 2d at the moment...")
    def test_2d(self):
        test_inputs = torch.Tensor(
            list(zip(torch.linspace(0, 1, 10), torch.linspace(1, 2, 10)))
        )
        result = self.kernel(test_inputs, test_inputs)
        self.assertEqual(result.shape, torch.Size([10, 10]))

    def test_inverse(self):
        test_inputs = torch.linspace(0, 1, 10).unsqueeze(1)
        case_1 = self.kernel(test_inputs, test_inputs)
        case_1 += self.mercer_kernel_args["noise_parameter"] ** 2 * torch.eye(
            test_inputs.shape[0]
        )
        test_1 = torch.inverse(case_1)

        test_2 = self.kernel.kernel_inverse(test_inputs)
        self.assertTrue((torch.abs(test_1 - test_2) < 1e-6).all())
        return


if __name__ == "__main__":
    unittest.main()
