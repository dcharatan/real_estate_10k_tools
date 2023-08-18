from typing import Tuple

import torch
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def intersect_gaussians(
    means: Float[Tensor, "*#batch dim"],
    covariances_inverse: Float[Tensor, "*#batch dim dim"],
    origins: Float[Tensor, "*#batch dim"],
    directions: Float[Tensor, "*#batch dim"],
    eps: float = torch.finfo(torch.float32).eps,
) -> Tuple[
    Float[Tensor, " *batch"],  # ray parameter
    Float[Tensor, "*batch dim"],  # intersection point
]:
    # Compute the ray parameter for the intersection.
    numerator = einsum(
        means - origins,
        covariances_inverse,
        directions,
        "... i, ... i j, ... j -> ...",
    )
    denominator = einsum(
        directions,
        covariances_inverse,
        directions,
        "... i, ... i j, ... j -> ...",
    )
    t = numerator / (denominator + eps)

    return t, origins + t[..., None] * directions


def evaluate_gaussians(
    means: Float[Tensor, "*#batch dim"],
    covariances_inverse: Float[Tensor, "*#batch dim dim"],
    points: Float[Tensor, "*#batch dim"],
) -> Float[Tensor, "*#batch"]:
    deltas = means - points
    distance = einsum(
        deltas,
        covariances_inverse,
        deltas,
        "... i, ... i j, ... j -> ...",
    )
    return torch.exp(-0.5 * distance)
