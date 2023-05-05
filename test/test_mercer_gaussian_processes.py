# test_mercer_gaussian_process.py
import unittest
import torch
import torch.distributions as D
from ortho.basis_functions import (
    smooth_exponential_basis_fasshauer,
    smooth_exponential_eigenvalues_fasshauer,
    Basis,
    RandomFourierFeatureBasis,
)

from mercergp.kernels import MercerKernel
from mercergp.MGP import (
    MercerGP,
    RFFGP,
    NonStationaryRFFGP,
    HilbertSpaceElement,
    MercerGPFourierPosterior,
    PosteriorGPSample,
)
from mercergp.posterior_sampling import MercerSpectralDistribution


def output_function(x: torch.Tensor) -> torch.Tensor:
    """
    A test function for generating output values where necessary in the following
    unit tests.
    """
    return (1.5 * torch.sin(x) + 0.5 * torch.cos(4 * x) + x / 8).squeeze()


class TestMercerGP(unittest.TestCase):
    def setUp(self):
        # basis parameters
        self.basis_function = smooth_exponential_basis_fasshauer
        self.dimension = 1
        self.order = 10

        # parameters
        l_se = torch.Tensor([[5]])
        sigma_se = torch.Tensor([1])
        prec = torch.Tensor([1])
        sigma_e = torch.Tensor([0.01])
        se_kernel_args = {
            "ard_parameter": l_se,
            "variance_parameter": sigma_se,
            "noise_parameter": sigma_e,
            "precision_parameter": prec,
        }

        self.params = se_kernel_args

        # basis function
        self.basis = Basis(
            self.basis_function, self.dimension, self.order, self.params
        )

        self.eigenvalues = smooth_exponential_eigenvalues_fasshauer(
            self.order, self.params
        )

        # gp parameters
        self.kernel = MercerKernel(
            self.order, self.basis, self.eigenvalues, self.params
        )

        self.mercer_gp = MercerGP(
            self.basis, self.order, self.dimension, self.kernel
        )

        # test points
        torch.manual_seed(1)
        test_sample_size = 50
        test_sample_shape = torch.Size([test_sample_size])
        self.input_points = D.Normal(0.0, 1.0).sample(test_sample_shape)
        return

    def test_adding_data(self):
        x = torch.Tensor([0.1, 4, 5.24, 7])
        x2 = torch.Tensor([2, 3, 4, 5])
        y = torch.Tensor([3, 4, 5, 6])
        y2 = torch.Tensor([9, 2, 7, 8])

        self.mercer_gp.add_data(x, y)
        self.mercer_gp.add_data(x2, y2)
        xs, ys = self.mercer_gp.get_inputs(), self.mercer_gp.get_outputs()

        self.assertTrue(
            (xs == torch.Tensor([0.1, 4, 5.24, 7, 2, 3, 4, 5])).all()
        )
        self.assertTrue((ys == torch.Tensor([3, 4, 5, 6, 9, 2, 7, 8])).all())

    def test_coefficients_shape_flat(self):
        # mercer_gp = MercerGP(
        # self.basis, self.eigenvalues, self.order, self.kernel
        # )
        test_inputs = torch.linspace(0, 1, 20)
        test_outputs = torch.linspace(0, 1, 20)

        self.mercer_gp.add_data(test_inputs, test_outputs)

        coefficients = self.mercer_gp._calculate_posterior_coefficients(
            test_inputs, test_outputs
        )
        self.assertEqual(coefficients.shape, torch.Size([self.order]))
        return

    def test_coefficients_shape_1d(self):
        test_inputs = torch.linspace(0, 1, 20).unsqueeze(1)
        test_outputs = torch.linspace(0, 1, 20).unsqueeze(1)

        self.mercer_gp.add_data(test_inputs, test_outputs)

        coefficients = self.mercer_gp._calculate_posterior_coefficients(
            test_inputs, test_outputs
        )
        self.assertEqual(coefficients.shape, torch.Size([self.order]))
        return

    def test_marginal_predictive_density(self):
        """
        Tests if the output is of the right shape.
        """
        predictive_density = self.mercer_gp.get_marginal_predictive_density(
            self.input_points
        )
        self.assertTrue(isinstance(predictive_density, D.Normal))

    def test_marginal_predictive_density_size(self):
        """
        Tests if the output is of the right shape.
        """
        predictive_density = self.mercer_gp.get_marginal_predictive_density(
            self.input_points
        )
        test_vector = output_function(self.input_points)
        potential_values = torch.exp(predictive_density.log_prob(test_vector))
        # breakpoint()
        self.assertEqual(potential_values.shape, self.input_points.shape)

    def test_marginal_predictive_density_output_mapping(self):
        """
        Tests if the n-th value of the predictive density log prob is the same
        as the log prob at the given input points elementwise.
        i.e, is evaluating log_prob on a vector of output points, when
        the appropriate input points were passed to generate the predictive
        density, giving the right evaluations?

        Do this by:
            a) generating the predictive density over the vector of input points,
            which is an element-wise Normal distribution over the values
            of the outputs for those inputs. Save this vector of predictive
            densities
            b) generating element-wise Normal distributions over the input points,
            and testing the density at the corresponding output evaluation.
            Each of these densities should be saved into a vector of predictive
            densities.
        If the values in these two vectors coincide, then it is working
        correctly.
        """
        # get the vector of output evaluations
        test_vector = output_function(self.input_points)
        # # standard predictive density over the vector
        predictive_density = self.mercer_gp.get_marginal_predictive_density(
            self.input_points
        )
        potential_values = torch.exp(predictive_density.log_prob(test_vector))

        # calculate them elementwise
        probs = []
        for input_point, test_point in zip(self.input_points, test_vector):
            pred_density = self.mercer_gp.get_marginal_predictive_density(
                input_point.unsqueeze(0)
            )
            probs.append(torch.exp(pred_density.log_prob(test_point)))
        # breakpoint()
        probs_tensor = torch.Tensor(probs)

        self.assertTrue(torch.allclose(probs_tensor, potential_values))


# @unittest.skip("Not correct")
class TestRFFMercerGP(unittest.TestCase):
    def setUp(self):
        # basis parameters
        # self.basis_function = smooth_exponential_basis_fasshauer
        self.dimension = 1
        self.order = 1000

        # parameters
        l_se = torch.Tensor([[5]])
        sigma_se = torch.Tensor([1])
        prec = torch.Tensor([1])
        sigma_e = torch.Tensor([0.01])
        se_kernel_args = {
            "ard_parameter": l_se,
            "variance_parameter": sigma_se,
            "noise_parameter": sigma_e,
            "precision_parameter": prec,
        }

        self.params = se_kernel_args

        self.spectral_distribution = D.Normal(
            torch.Tensor([0.0]), torch.Tensor([1.0])
        )

        # gp parameters
        self.rff_gp = RFFGP(
            self.order,
            self.dimension,
            self.spectral_distribution,
        )
        return

    @unittest.skip("")
    def test_adding_data(self):
        x = torch.Tensor([0.1, 4, 5.24, 7]).t()
        x2 = torch.Tensor([2, 3, 4, 5]).t()
        y = torch.Tensor([3, 4, 5, 6]).t()
        y2 = torch.Tensor([9, 2, 7, 8]).t()

        self.rff_gp.add_data(x, y)
        self.rff_gp.add_data(x2, y2)
        xs, ys = self.rff_gp.get_inputs(), self.rff_gp.get_outputs()

        self.assertTrue(
            (xs == torch.Tensor([0.1, 4, 5.24, 7, 2, 3, 4, 5])).all()
        )
        self.assertTrue((ys == torch.Tensor([3, 4, 5, 6, 9, 2, 7, 8])).all())

    @unittest.skip("")
    def test_coefficients_shape_flat(self):
        test_inputs = torch.linspace(0, 1, 20)
        test_outputs = torch.linspace(0, 1, 20)

        self.rff_gp.add_data(test_inputs, test_outputs)

        coefficients = self.rff_gp._calculate_posterior_coefficients(
            test_inputs, test_outputs
        )
        self.assertEqual(coefficients.shape, torch.Size([self.order]))
        return

    @unittest.skip("")
    def test_coefficients_shape_1d(self):
        test_inputs = torch.linspace(0, 1, 20).unsqueeze(1)
        test_outputs = torch.linspace(0, 1, 20).unsqueeze(1)

        self.rff_gp.add_data(test_inputs, test_outputs)

        coefficients = self.rff_gp._calculate_posterior_coefficients(
            test_inputs, test_outputs
        )
        self.assertEqual(coefficients.shape, torch.Size([self.order]))
        return


class TestMercerFourierGPPosterior(unittest.TestCase):
    def setUp(self):
        # hyperparameters and basis counts
        self.order = 15
        self.rff_order = 5000
        self.frequency = 1000
        self.dimension = 1
        self.kernel_args = {
            "ard_parameter": torch.Tensor([[1.0]]),
            "noise_parameter": torch.Tensor([0.1]),
            "precision_parameter": torch.Tensor([1.0]),
            "variance_parameter": torch.Tensor([1.0]),
        }
        self.eigenvalues = smooth_exponential_eigenvalues_fasshauer(
            self.order, self.kernel_args
        )
        # MercerGPFourierPosterior parameters
        self.spectral_distribution = MercerSpectralDistribution(
            self.frequency, torch.ones(self.rff_order)
        )

        self.basis = Basis(
            smooth_exponential_basis_fasshauer, 1, self.order, self.kernel_args
        )
        self.rff_basis = RandomFourierFeatureBasis(
            self.dimension, self.order, self.spectral_distribution
        )
        self.dimension = 1
        self.kernel = MercerKernel(
            self.order, self.basis, self.eigenvalues, self.kernel_args
        )

        # GP setup
        self.mercer_gp = MercerGPFourierPosterior(
            self.basis,
            self.rff_basis,
            self.order,
            self.rff_order,
            self.dimension,
            self.kernel,
        )
        pass

    def test_gen_gp(self):
        gp = self.mercer_gp.gen_gp()
        self.assertTrue(isinstance(gp, PosteriorGPSample))
        pass


class TestNonStationaryRFFGP(unittest.TestCase):
    def setUp(self):
        self.frequency = 1000
        self.order = 5000
        self.dimension = 1

        self.spectral_distribution = MercerSpectralDistribution(
            self.frequency, torch.ones(self.order)
        )
        basis = RandomFourierFeatureBasis(
            self.dimension, self.order, self.spectral_distribution
        )
        self.nsrffgp = NonStationaryRFFGP(basis, self.order, self.dimension)
        pass

    def test_get_sample_coefficients(self):
        sample_coeffics = self.nsrffgp._get_sample_coefficients()
        self.assertEqual(sample_coeffics.shape, torch.Size([self.order]))

    def test_get_gp_size(self):
        sample_gp = self.nsrffgp.gen_gp()
        self.assertEqual(sample_gp.get_order(), self.order)

    def test_get_gp(self):
        sample_gp = self.nsrffgp.gen_gp()
        self.assertTrue(
            isinstance(sample_gp, HilbertSpaceElement),
            2 * self.order,
        )


if __name__ == "__main__":
    unittest.main()
