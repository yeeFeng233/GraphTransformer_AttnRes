from torch import Tensor
from torch.nn import Module

__all__ = ["AntiSymmetric", "Symmetric"]


class AntiSymmetric(Module):
    r"""
    Anti-Symmetric Parametrization

    A weight matrix :math:`\mathbf{W}` is parametrized as
    :math:`\mathbf{W} = \mathbf{W} - \mathbf{W}^T`
    """

    def __init__(self):
        super().__init__()

    def forward(self, W: Tensor) -> Tensor:
        return W.triu(diagonal=1) - W.triu(diagonal=1).T

    def right_inverse(self, W: Tensor) -> Tensor:
        return W.triu(diagonal=1)


class Symmetric(Module):
    r"""
    Symmetric Parametrization

    A weight matrix :math:`\mathbf{W}` is parametrized as
    :math:`\mathbf{W} = \mathbf{W} + \mathbf{W}^T`
    """

    def __init__(self):
        super().__init__()

    def forward(self, W: Tensor) -> Tensor:
        return W.triu() + W.triu().T

    def right_inverse(self, W: Tensor) -> Tensor:
        return W.triu()
