import torch
from einops import rearrange, reduce, repeat
from jaxtyping import Float
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
