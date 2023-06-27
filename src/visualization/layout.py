"""This file contains useful layout utilities for images. They are:

- add_border: Add a border to an image.
- cat/hcat/vcat: Join images by arranging them in a line. If the images have different
  sizes, they are aligned as specified (start, end, center). Allows you to specify a gap
  between images.

Images are assumed to be float32 tensors with shape (channel, height, width).
"""

from typing import Any, Generator, Iterable, List, Literal, Union

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
Color = Union[
    int,
    float,
    Iterable[int],
    Iterable[float],
    Float[Tensor, "#channel"],
    Float[Tensor, ""],
]


def _sanitize_color(color: Color) -> Float[Tensor, "#channel"]:
    # Convert tensor to list (or individual item).
    if isinstance(color, torch.Tensor):
        color = color.tolist()

    # Turn iterators and individual items into lists.
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]

    return torch.tensor(color, dtype=torch.float32)


def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _get_main_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 2,
        "vertical": 1,
    }[main_axis]


def _get_cross_dim(main_axis: Axis) -> int:
    return {
        "horizontal": 1,
        "vertical": 2,
    }[main_axis]


def _compute_offset(base: int, overlay: int, alignment: Alignment) -> slice:
    assert base >= overlay
    offset = {
        "start": 0,
        "center": (base - overlay) // 2,
        "end": base - overlay,
    }[alignment]
    return slice(offset, offset + overlay)


def overlay(
    base: Float[Tensor, "channel base_height base_width"],
    overlay: Float[Tensor, "channel overlay_height overlay_width"],
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Float[Tensor, "channel base_height base_width"]:
    # The overlay must be smaller than the base.
    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width

    # Compute spacing on the main dimension.
    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(
        base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment
    )

    # Compute spacing on the cross dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )

    # Combine the slices and paste the overlay onto the base accordingly.
    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[selector] = overlay
    return result


def cat(
    images: List[Float[Tensor, "channel _ _"]],
    main_axis: Axis,
    alignment: Alignment,
    gap: int = 0,
    gap_color: Color = 1,
) -> Float[Tensor, "channel height width"]:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    # Find the maximum image side length in the cross axis dimension.
    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    # Pad the images.
    padded_images = []
    for image in images:
        # Create an empty image with the correct size.
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", alignment))

    # Intersperse separators if necessary.
    if gap > 0:
        # Generate a separator.
        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones((*separator_size, c), dtype=torch.float32, device=device)
        separator = rearrange(separator * gap_color, "h w c -> c h w")

        # Intersperse the separator between the images.
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))


def hcat(
    images: List[Float[Tensor, "channel _ _"]],
    alignment: Literal["start", "center", "end", "top", "bottom"],
    gap: int = 0,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        images,
        "horizontal",
        {
            "start": "start",
            "center": "center",
            "end": "end",
            "top": "start",
            "bottom": "end",
        }[alignment],
        gap=gap,
        gap_color=gap_color,
    )


def vcat(
    images: List[Float[Tensor, "channel _ _"]],
    alignment: Literal["start", "center", "end", "left", "right"],
    gap: int = 0,
    gap_color: Color = 1,
):
    """Shorthand for a horizontal linear concatenation."""
    return cat(
        images,
        "vertical",
        {
            "start": "start",
            "center": "center",
            "end": "end",
            "left": "start",
            "right": "end",
        }[alignment],
        gap=gap,
        gap_color=gap_color,
    )


def add_border(
    image: Float[Tensor, "channel height width"],
    border: int,
    border_color: Color = 1,
) -> Float[Tensor, "channel new_height new_width"]:
    border_color = _sanitize_color(border_color).to(device)
    c, h, w = image.shape
    result = torch.empty(
        (c, h + 2 * border, w + 2 * border), dtype=torch.float32, device=image.device
    )
    result[:] = border_color[:, None, None]
    result[:, border : h + border, border : w + border] = image
    return result
