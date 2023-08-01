from typing import Tuple

import torch
from einops import einsum, rearrange, reduce, repeat
from jaxtyping import Bool, Float
from torch import Tensor


@torch.no_grad()
def draw_markers(
    image: Float[Tensor, "channel height width"],
    marker_xy: Float[Tensor, "batch 2"],
    marker_color: Float[Tensor, "batch 3"],
    inner_radius: Float[Tensor, " batch"] = torch.tensor(8.0, dtype=torch.float32),
    outer_radius: Float[Tensor, " batch"] = torch.tensor(12.0, dtype=torch.float32),
) -> Float[Tensor, "channel height width"]:
    device = image.device
    b, _ = marker_xy.shape
    _, h, w = image.shape

    # Generate a pixel grid.
    x = torch.arange(w, device=device) + 0.5
    y = torch.arange(h, device=device) + 0.5
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)
    xy = rearrange(xy, "h w xy -> h w () xy")

    # Compute distance to the pixel grid to determine which pixels are marked.
    marker_xy = rearrange(marker_xy, "b xy -> () () b xy")
    delta = (xy - marker_xy).norm(dim=-1)
    marked = (delta >= inner_radius) & (delta <= outer_radius)
    marked = repeat(marked, "h w b -> b h w")

    # Select the highest-priority (largest index in array) color for each pixel.
    index = torch.arange(b, device=image.device)
    index = repeat(index, "b -> b h w", h=h, w=w).clone()
    index[~marked] = -1
    index = index.max(dim=0).values.clip(min=0)
    color = rearrange(marker_color[index], "h w c -> c h w")

    # Mark the pixels.
    mask = reduce(marked, "b h w -> h w", "max")
    mask = repeat(mask, "h w -> c h w", c=3)
    marked_image = image.detach().clone()
    marked_image[mask] = color[mask]
    return marked_image


def _generate_line_mask(
    shape: Tuple[int, int],
    start: Float[Tensor, "line 2"],
    end: Float[Tensor, "line 2"],
    width: float,
) -> Bool[Tensor, "height width"]:
    device = start.device

    # Generate a pixel grid.
    h, w = shape
    x = torch.arange(w, device=device) + 0.5
    y = torch.arange(h, device=device) + 0.5
    xy = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)

    # Define a vector between the start and end points.
    delta = end - start
    delta_norm = delta.norm(dim=-1, keepdim=True)
    u_delta = delta / delta_norm

    # Define a vector between each pixel and the start point.
    indicator = xy - start[:, None, None]

    # Determine whether each pixel is inside the line in the parallel direction.
    parallel = einsum(u_delta, indicator, "l xy, l h w xy -> l h w")
    parallel_inside_line = (parallel <= delta_norm[..., None]) & (parallel > 0)

    # Determine whether each pixel is inside the line in the perpendicular direction.
    perpendicular = indicator - parallel[..., None] * u_delta[:, None, None]
    perpendicular_inside_line = perpendicular.norm(dim=-1) < (0.5 * width)

    return (parallel_inside_line & perpendicular_inside_line).any(dim=0)


def draw_lines(
    image: Float[Tensor, "3 height width"],
    start: Float[Tensor, "line 2"],
    end: Float[Tensor, "line 2"],
    color: Float[Tensor, "3"],
    width: float = 4.0,
    supersample: int = 5,
) -> Float[Tensor, "3 height width"]:
    _, h, w = image.shape
    s = supersample
    mask = _generate_line_mask((h * s, w * s), start * s, end * s, width * s)
    mask = reduce(mask.float(), "(h hs) (w ws) -> h w", "mean", hs=s, ws=s)

    # Paint the line on the image.
    return image * (1 - mask[None]) + color[:, None, None] * mask[None]
