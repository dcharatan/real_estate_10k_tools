from math import log

import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def inverse_cdf(
    x: Float[Tensor, " *batch"],
    min_frequency: float,
    max_frequency: float,
) -> Float[Tensor, " *batch"]:
    """This transforms a uniform frequency distribution between 0 and 1 to resemble a
    positional encoding's frequency distribution. See the interactive version:
    https://www.desmos.com/calculator/wids3vx9lj
    """
    a = log(min_frequency)
    b = log(max_frequency)
    return torch.exp(x * (b - a) + a)


class RandomSinusoidEncoding(nn.Module):
    frequencies: Float[Tensor, "frequency dim"]
    phases: Float[Tensor, " frequency"]

    def __init__(
        self,
        dimensionality: int,
        num_frequencies: int,
        largest_period: float,
        num_octaves: int,
    ):
        super().__init__()

        # Pick appropriately distributed random frequencies.
        min_frequency = 2 * torch.pi / largest_period
        max_frequency = min_frequency * 2**num_octaves
        samples = torch.rand(num_frequencies, dtype=torch.float32)
        samples = inverse_cdf(samples, min_frequency, max_frequency)
        sample_dirs = torch.randn((num_frequencies, dimensionality))
        sample_dirs = sample_dirs / sample_dirs.norm(dim=-1, keepdim=True)
        samples = sample_dirs * samples[..., None]
        self.register_buffer("frequencies", samples, persistent=True)

        # Pick uniformly distributed random phases.
        phases = torch.rand(num_frequencies, dtype=torch.float32) * 2 * torch.pi
        self.register_buffer("phases", phases, persistent=True)

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        samples = einsum(samples, self.frequencies, "... d, f d -> ... f")
        return torch.sin(samples + self.phases)

    @property
    def d_out(self) -> int:
        return self.frequencies.shape[0]
