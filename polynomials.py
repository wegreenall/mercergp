import torch


def chebyshev_first(x: torch.Tensor, deg: int) -> torch.Tensor:
    """
    Evaluates the Chebyshev polynomial of the first kind at x, for order i
    """
    return torch.cos(deg * torch.arccos(x))


def chebyshev_second(x: torch.Tensor, deg: int) -> torch.Tensor:
    """
    Evaluates the Chebyshev polynomial of the second kind at x, for order i
    """
    return torch.sin((deg + 1) * torch.arccos(x)) / torch.sin(torch.arccos(x))


def generalised_laguerre(x: torch.Tensor, deg: int, params: dict):
    """
    Implements the Generalized Laguerre polynomials.

    The generalised Laguerre polynomials can be written recursively:
        L_0^α(x) = 1
        L_1^α(x) = 1 + α - x
    and then:
        L_{k+1}^α(x) = ((2k + 1 + α  - x)L_k^α(x) - (k+α)L_{k-1}^α(x)) / k+1
    """
    assert (
        "alpha" in params
    ), "Missing parameter for generalised laguerre polynomial: alpha"
    alpha = params["alpha"]

    if deg == 0:
        return torch.ones(x.shape)
    elif deg == 1:
        return 1 + alpha - x
    else:
        # k = deg - 1
        coeffic_1 = 2 * (deg - 1) + 1 + alpha - x
        coeffic_2 = deg - 1 + alpha
        denom = deg
        return (
            coeffic_1 * generalised_laguerre(x, deg - 1, params)
            - coeffic_2 * generalised_laguerre(x, deg - 2, params)
        ) / denom
