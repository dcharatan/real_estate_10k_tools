import torch
import torch.nn as nn
from einops import einsum
from jaxtyping import Float
from torch import Tensor


class RandomSinusoidEncoding(nn.Module):
    frequencies: Float[Tensor, "frequency dim"]
    phases: Float[Tensor, " frequency"]

    def __init__(
        self,
        dimensionality: int,
        num_frequencies: int,
        largest_period: float,
    ):
        super().__init__()

        # Pick uniformly distributed random frequencies.
        frequencies = torch.rand((num_frequencies, dimensionality), dtype=torch.float32)
        frequencies = 2 * torch.pi * frequencies / largest_period
        self.register_buffer("frequencies", frequencies, persistent=False)

        # Pick uniformly distributed random phases.
        phases = torch.rand(num_frequencies, dtype=torch.float32) * 2 * torch.pi
        self.register_buffer("phases", phases, persistent=False)

    def forward(
        self,
        samples: Float[Tensor, "*batch dim"],
    ) -> Float[Tensor, "*batch embedded_dim"]:
        samples = einsum(samples, self.frequencies, "... d, f d -> ... f")
        return torch.sin(samples + self.phases)

    @property
    def d_out(self) -> int:
        return self.frequencies.shape[0]
