import unittest
import torch
from mercergp.builders import GPBuilder, GPBuilderState
from mercergp.kernels import MercerKernel
from mercergp.MGP import MercerGP
from ortho.builders import OrthoBuilder
from ortho.basis_functions import Basis, smooth_exponential_basis_fasshauer
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer


class BuilderTest(unittest.TestCase):
    def setUp(self):
        self.order = 10
        self.gp_builder = GPBuilder(self.order)
        self.ortho_builder = OrthoBuilder(self.order)
        self.parameters = {
            "ard_parameter": torch.Tensor([1.0]),
            "precision_parameter": torch.Tensor([1.0]),
            "variance_parameter": torch.Tensor([1.0]),
            "noise_parameter": torch.Tensor([1.0]),
        }
        self.basis = Basis(
            smooth_exponential_basis_fasshauer, 1, self.order, self.parameters
        )
        self.eigenvalue_generator = SmoothExponentialFasshauer(self.order)

    def test_set_kernel(self):
        eigenvalues = self.eigenvalue_generator(self.parameters)
        kernel = MercerKernel(
            self.order, self.basis, eigenvalues, self.parameters
        )
        self.gp_builder.set_kernel(kernel)
        self.assertTrue(self.gp_builder.kernel is not None)
        self.assertTrue(self.gp_builder.basis is not None)
        self.assertTrue(self.gp_builder.state == GPBuilderState.READY)

    def test_set_basis(self):
        self.gp_builder.set_basis(self.basis)
        self.assertTrue(self.gp_builder.basis is not None)
        self.assertTrue(self.gp_builder.state == GPBuilderState.NOT_READY)

    def test_set_parameters(self):
        self.gp_builder.set_parameters(self.parameters)
        self.assertTrue(self.gp_builder.parameters is not None)
        self.assertTrue(self.gp_builder.state == GPBuilderState.NOT_READY)

    def test_set_eigenvalue_generator(self):
        self.gp_builder.set_eigenvalue_generator(self.eigenvalue_generator)
        self.assertTrue(self.gp_builder.eigenvalue_generator is not None)
        self.assertTrue(self.gp_builder.state == GPBuilderState.NOT_READY)

    def test_set_all(self):
        self.gp_builder.set_basis(self.basis)
        self.gp_builder.set_parameters(self.parameters)
        self.gp_builder.set_eigenvalue_generator(self.eigenvalue_generator)
        self.assertTrue(self.gp_builder.basis is not None)
        self.assertTrue(self.gp_builder.parameters is not None)
        self.assertTrue(self.gp_builder.eigenvalue_generator is not None)
        self.assertTrue(self.gp_builder.kernel is not None)
        self.assertTrue(self.gp_builder.state == GPBuilderState.READY)

    def test_build(self):
        self.gp_builder.set_basis(self.basis)
        self.gp_builder.set_parameters(self.parameters)
        self.gp_builder.set_eigenvalue_generator(self.eigenvalue_generator)
        gp = self.gp_builder.build()
        self.assertTrue(gp is not None)
        self.assertTrue(isinstance(gp, MercerGP))

    def test_build_fail(self):
        self.assertRaises(RuntimeError, self.gp_builder.build)

    def test_self_ref_path_1(self):
        gp = (
            self.gp_builder.set_basis(self.basis)
            .set_parameters(self.parameters)
            .set_eigenvalue_generator(self.eigenvalue_generator)
            .build()
        )
        self.assertTrue(gp is not None)
        self.assertTrue(isinstance(gp, MercerGP))

    def test_self_ref_path_2(self):
        eigenvalues = self.eigenvalue_generator(self.parameters)
        kernel = MercerKernel(
            self.order, self.basis, eigenvalues, self.parameters
        )
        gp = self.gp_builder.set_kernel(kernel).build()
        self.assertTrue(gp is not None)
        self.assertTrue(isinstance(gp, MercerGP))
