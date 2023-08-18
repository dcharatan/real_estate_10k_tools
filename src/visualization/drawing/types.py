from typing import Iterable, Union

import torch
from einops import repeat
from jaxtyping import Float, Shaped
from torch import Tensor

Real = Union[float, int]

Vector = Union[
    Real,
    Iterable[Real],
    Shaped[Tensor, "3"],
    Shaped[Tensor, "batch 3"],
]


def sanitize_vector(
    vector: Vector,
    dim: int,
    device: torch.device,
) -> Float[Tensor, "*#batch dim"]:
    vector = torch.tensor(vector, dtype=torch.float32, device=device)
    while vector.ndim < 2:
        vector = vector[None]
    if vector.shape[-1] == 1:
        vector = repeat(vector, "... () -> ... c", c=dim)
    assert vector.shape[-1] == dim
    assert vector.ndim == 2
    return vector


Scalar = Union[
    Real,
    Iterable[Real],
    Shaped[Tensor, ""],
    Shaped[Tensor, " batch"],
]


def sanitize_scalar(scalar: Scalar, device: torch.device) -> Float[Tensor, "*#batch"]:
    scalar = torch.tensor(scalar, dtype=torch.float32, device=device)
    while scalar.ndim < 1:
        scalar = scalar[None]
    assert scalar.ndim == 1
    return scalar


Pair = Union[
    Iterable[Real],
    Shaped[Tensor, "2"],
]


def sanitize_pair(pair: Pair, device: torch.device) -> Float[Tensor, "2"]:
    pair = torch.tensor(pair, dtype=torch.float32, device=device)
    assert pair.shape == (2,)
    return pair
