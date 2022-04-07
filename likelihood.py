# likelihood.py
import torch
import torch.distributions as D
import math
from ortho.basis_functions import Basis, OrthonormalBasis
from ortho.orthopoly import OrthogonalBasisFunction
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


class MercerLikelihood:
    def __init__(
        self,
        order: int,
        optimiser: torch.optim.Optimizer,
        basis: Basis,
        input_sample: torch.Tensor,
        output_sample: torch.Tensor,
    ):
        """
        Initialises the Likelihood class.

        To use this, construct an instance of a torch.optim.Optimizer;
        register the parameters that are to be optimised, and pass it when
        instantiating this class.

        Parameters:
            input_sample: The sample of data X.
            order:        The bandwidth of the kernel/no. of basis functions.
            # base_mean:    The constant μ; mean of the base distribution
            # base_var:     The constant σ^2; variance of the base distribution
            mean_coeffics:The vector of coefficients for the Karhunen-Loeve
            # alpha:
            # epsilon:
            optimiser:
            mc_sample_size=10000:
        """
        self.order = order
        self.optimiser = optimiser
        self.basis = basis

        self.input_sample = input_sample
        self.sample_size = input_sample.shape[0]  # this should be alright
        self.output_sample = output_sample
        pass

    def fit(self, parameters, iter_count=10000, convergence_eps=1e-9):
        """
        Fits the parameters for the given Mercer Gaussian process regression.
        i.e. runs the iteration that maximises the likelihood.
        """
        convergence_criterion = False
        for i in range(iter_count):
            if i % 1 == 0:
                print("iteration:", i)
                print("Gamma grad:", parameters["gammas"].grad)
                print("Gamma", parameters["gammas"])
            loss_value = self.step_optimisation(parameters)
            convergence_criterion = (
                torch.abs(parameters["gammas"].grad) < convergence_eps
            ).all()
            # print("Convergence criterion:", convergence_criterion)
            # last_loss_value = loss_value
            if convergence_criterion:
                print(parameters["gammas"].grad)
                # print("The system thinks it has converged...")
                break
        return

    def step_optimisation(self, parameters):
        """
        Steps the optimiser, having evaluated the likelihood (i.e. the loss).
        """
        self.optimiser.zero_grad()
        loss = self.evaluate_likelihood(parameters)
        # print("log-likelihood:", -loss)
        loss.backward(retain_graph=False)
        self.optimiser.step()
        return loss

    def evaluate_likelihood(self, parameters) -> torch.Tensor:
        """
        Evaluates the marginal likelihood as constructed from the Mercer
        Guassian process formalism; i.e. using the appropriate Woodbury Sherman
        Morrison formulae, constructs and evaluates the likelihood function at
        the input data to allow for optimisation of kernel parameters appearing
        in the model.
        """
        const_term = (
            self.sample_size
            * 0.5
            * torch.log(
                torch.tensor(
                    2 * math.pi,
                )
            )
        )  # 1

        log_det_term, exp_term = self._det_and_exp(parameters)
        return const_term + log_det_term + exp_term

    """
    The following are a sequence of helper methods to maintain a clean approach
    to the evaluation of the separate "components" of the likelihood function,
    as well as to impose the single responsibility principle.
    """

    def _det_and_exp(self, parameters) -> (torch.Tensor, torch.Tensor):
        """
        Returns the log determinant and exponential term of the parameters.

        Because each of these terms depends on multiple copies of some of
            Ξ, Ξ'Ξ, and (Ξ'Ξ)^{-1},

        we can speed things up by calculating these things only once.
        """
        ksi = self._ksi(parameters)
        ksiksi = self._ksiksi(parameters, ksi)
        ksiksi_inverse = self._ksiksi_inverse(parameters, ksi, ksiksi)

        log_det_term = 0.5 * self._log_determinant(parameters, ksiksi)

        exp_term = self._exp_term(parameters, ksiksi_inverse)
        exp_term = torch.Tensor([0.0])
        return (log_det_term, exp_term)

    def _log_determinant(self, parameters, ksiksi) -> torch.Tensor:
        """
        Calculates the log determinant component of the likelihood.

        The WSM formula gives that the determinant:
                |K + σ^2Ι| ≈ |ΞΛΞ' + σ^2Ι|
        can be calculated as:
             |ΞΛΞ' + σ^2Ι| = σ^{2n}|Λ||Λ^-1 + σ^2Ξ'Ξ|

        And therefore the log can be written:

           ln|ΞΛΞ' + σ^2Ι| = 2n * ln(σ)+|Λ||Λ^-1 + σ^2Ξ'Ξ|
        """
        eigenvalues = self._eigenvalues(parameters)
        # breakpoint()
        term_1 = self.sample_size * torch.log(parameters["noise_parameter"])
        # term_1: det((diag(A)) = prod(aii)
        """
        The determinant of a diagonal matrix is the product of its diagonal
        elements; the log of the determinant of a diagonal matrix is the
        sum of the logs of its diagonal elements.
        """

        term_2 = torch.sum(torch.log(eigenvalues))

        term_3 = torch.log(
            torch.linalg.det(self._lambdainv(parameters) + ksiksi)
        )
        return (term_1 + term_2 + term_3).squeeze()

    def _exp_term(self, parameters, ksiksi_inverse):
        # ksiksi_inverse = self._ksiksi_inverse(parameters)

        return torch.einsum(
            "i,ij,j ->", self.output_sample, ksiksi_inverse, self.output_sample
        )

    def _ksiksi(self, parameters, ksi):
        """
        Returns σ^2 (Ξ'Ξ)
        """
        # ksi = self._ksi(parameters)  # N x m
        # print("ksi is:", ksi, "from: _ksiksi(parameters)")
        ksiksi = torch.einsum("ij, jk -> ik", ksi.t(), ksi)  # m x m

        try:
            returnval = (1 / parameters["noise_parameter"]) * ksiksi
            # print(ksiksi)
        except RuntimeError:
            breakpoint()

        return returnval

    def _ksiksi_inverse(self, parameters, ksi, ksiksi):
        """
        Calculates and returns the inverse kernel matrix in the exponential
        component of the likelihood (the quadratic term in the log-likelihood).

        The WSM formula gives that the inverse:
                (K + σ^2Ι)^(-1) ≈ (ΞΛΞ' + σ^2Ι)^(-1)

        can be calculated as:
             (ΞΛΞ' + σ^2Ι)^{-1} = σ^{-2} I_n - σ^{-2} Ξ(σ^2Λ^-1 + Ξ'Ξ)^{-1}Ξ'

        """
        inverse_sigma = 1 / parameters["noise_parameter"]
        term_1 = inverse_sigma * torch.eye(self.sample_size)

        term_2 = inverse_sigma * torch.linalg.multi_dot(
            (
                ksi,
                torch.inverse(
                    (1 / inverse_sigma) * self._lambdainv(parameters) + ksiksi
                ),
                ksi.t(),
            )
        )
        # breakpoint()
        return term_1 - term_2

    def _ksi(self, parameters):
        """
        Returns Ξ.

        It does this by setting the parameters of the basis
        and then evaluating the basis.
        Return shape: N x m
        """
        self.basis.basis_function.set_gammas(
            torch.cat((torch.Tensor([1.0]), parameters["gammas"][1:]))
        )

        # breakpoint()
        return self.basis(self.input_sample)

    def _lambdainv(self, parameters):
        """
        returns the diagonal matrix λ^-1.

        Return shape m x m
        """
        return torch.diag(1 / self._eigenvalues(parameters))

    def _eigenvalues(self, parameters) -> torch.Tensor:
        """
        Returns the vector of eigenvalues, up to the order of the model.

        The current form of the eigenvalues will be taken to be the following,
        for each n <= m:
                     λ_n = λ / (n + c_n)^p

        where: p denotes the smoothness parameter;
               λ denotes a scale parameter;
               c_n denotes a per-eigenvalue shape parameter.

        Return shape: m
        """
        p: int = parameters["eigenvalue_smoothness_parameter"]
        l: torch.Tensor = parameters["eigenvalue_scale_parameter"]  # shape: 1
        cn: torch.Tensor = parameters["shape_parameter"]  # shape: m
        eigenvalues = (
            l / (torch.linspace(0, self.order - 1, self.order) + cn) ** p
        )
        # print(eigenvalues)
        return eigenvalues


if __name__ == "__main__":

    def test_function(x: torch.Tensor):
        return 2.5 - x ** 2

    # torch.manual_seed(5)
    print("likelihood.py")
    order = 8
    sample_size = 400
    input_sample = D.Normal(0.0, 1.0).sample([sample_size])
    noise_parameter = torch.Tensor([0.2])
    output_sample = test_function(input_sample) + D.Normal(0.0, 0.5).sample(
        input_sample.shape
    )

    plt.scatter(
        input_sample.numpy().flatten(), output_sample.numpy().flatten()
    )
    plt.show()
    betas = 0 * torch.ones(2 * order + 1)
    log_gammas = 0 * torch.ones(2 * order + 1)
    log_gammas[0] = torch.log(torch.Tensor([1.0]))
    log_gammas.requires_grad = True

    basis_function = OrthogonalBasisFunction(
        order, betas, torch.exp(log_gammas)
    )
    # params_from_net = basis_function.med.moment_net.parameters()
    # breakpoint()
    optimiser = torch.optim.SGD(
        [
            log_gammas,
        ],
        lr=0.01,
    )
    basis = OrthonormalBasis(basis_function, 1, order)

    # fit the likelihood
    parameters = {
        "gammas": log_gammas,
        "noise_parameter": noise_parameter,
        "eigenvalue_smoothness_parameter": torch.Tensor([4.0]),
        "eigenvalue_scale_parameter": torch.Tensor([1.0]),
        "shape_parameter": torch.Tensor([1.0]),
    }
    likelihood = MercerLikelihood(
        order, optimiser, basis, input_sample, output_sample
    )

    likelihood.fit(parameters)
    # print("Entering likelihood fit")
    # fineness = math.floor(100)
    # # x = torch.linspace(0.0, 2.18, fineness)
    # x = torch.linspace(0.01, 5, fineness)
    # likelihood_vals = []
    # for i in range(fineness):
    # # log_gammas  =
    # log_gammas[2] = torch.log(x[i])

    # print(x[i])
    # parameters["gammas"] = log_gammas
    # try:
    # likelihood_vals.append(likelihood.evaluate_likelihood(parameters))
    # except RuntimeError:
    # print("Failed to invert the matrix at x:", x[i])
    # plt.plot(x[:i], likelihood_vals)
    # plt.show()
    # break
    # # likelihood_vals

    # plt.plot(x, likelihood_vals)
    # plt.show()