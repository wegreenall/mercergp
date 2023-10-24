import unittest
import torch
import torch.distributions as D
from mercergp.builders import (
    MercerGPBuilder,
    GPBuilderState,
    FourierPosteriorMercerGPBuilder,
)
from mercergp.kernels import MercerKernel
from mercergp.MGP import MercerGP, MercerGPFourierPosterior
from ortho.builders import OrthoBuilder
from ortho.basis_functions import (
    Basis,
    smooth_exponential_basis_fasshauer,
    RandomFourierFeatureBasis,
)
from mercergp.posterior_sampling import gaussian_spectral_distribution
from mercergp.eigenvalue_gen import SmoothExponentialFasshauer


class MercerBuilderTest(unittest.TestCase):
    def setUp(self):
        self.order = 10
        self.gp_builder = MercerGPBuilder(self.order)
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


class FourierPosteriorGPMercerBuilderTest(unittest.TestCase):
    def setUp(self):
        self.dim = 1
        self.order = 10

        # fourier posterior parameters
        self.rff_order = 2000
        self.frequency = 200

        # Standard parameters
        self.gp_builder = FourierPosteriorMercerGPBuilder(
            self.order, self.rff_order
        )
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
        self.eigenvalues = self.eigenvalue_generator(self.parameters)
        self.kernel = MercerKernel(
            self.order, self.basis, self.eigenvalues, self.parameters
        )
        self.spectral_distribution = gaussian_spectral_distribution(
            self.kernel, -5, 5, self.frequency
        )
        self.rff_basis = RandomFourierFeatureBasis(
            self.dim, self.rff_order, self.spectral_distribution
        )

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
        self.gp_builder.set_fourier_basis(self.rff_basis)
        self.assertTrue(self.gp_builder.basis is not None)
        self.assertTrue(self.gp_builder.parameters is not None)
        self.assertTrue(self.gp_builder.eigenvalue_generator is not None)
        self.assertTrue(self.gp_builder.fourier_basis is not None)
        self.assertTrue(self.gp_builder.kernel is not None)
        self.assertTrue(self.gp_builder.state == GPBuilderState.READY)
        self.assertTrue(self.gp_builder.fourier_state == GPBuilderState.READY)

    def test_build(self):
        self.gp_builder.set_basis(self.basis)
        self.gp_builder.set_parameters(self.parameters)
        self.gp_builder.set_eigenvalue_generator(self.eigenvalue_generator)
        self.gp_builder.set_fourier_basis(self.rff_basis)
        gp = self.gp_builder.build()
        self.assertTrue(gp is not None)
        self.assertTrue(isinstance(gp, MercerGP))
        self.assertTrue(isinstance(gp, MercerGPFourierPosterior))

    def test_build_fail(self):
        self.assertRaises(RuntimeError, self.gp_builder.build)

    def test_self_ref_path_1(self):
        gp = (
            self.gp_builder.set_basis(self.basis)
            .set_parameters(self.parameters)
            .set_eigenvalue_generator(self.eigenvalue_generator)
            .set_fourier_basis(self.rff_basis)
            .build()
        )
        self.assertTrue(gp is not None)
        self.assertTrue(isinstance(gp, MercerGP))
        self.assertTrue(isinstance(gp, MercerGPFourierPosterior))

    def test_self_ref_path_2(self):
        eigenvalues = self.eigenvalue_generator(self.parameters)
        kernel = MercerKernel(
            self.order, self.basis, eigenvalues, self.parameters
        )
        gp = (
            self.gp_builder.set_kernel(kernel)
            .set_fourier_basis(self.rff_basis)
            .build()
        )
        self.assertTrue(gp is not None)
        self.assertTrue(isinstance(gp, MercerGP))
        self.assertTrue(isinstance(gp, MercerGPFourierPosterior))


@unittest.skip("Legacy")
class LegacyBuildFunctions(unittest.TestCase):
    def setUp(self):
        self.order = 10
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

    def test_build_mercer_gp(self):
        pass
