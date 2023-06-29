from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from .layout import vcat


def crop_whitespace(
    image: Float[Tensor, "3 height width"],
) -> Float[Tensor, "3 cropped_height cropped_width"]:
    content_pixels = (image != 1).all(dim=0)

    content_rows = content_pixels.any(dim=1)
    from_row = torch.where(content_rows)[0].min()
    to_row = torch.where(content_rows)[0].max() + 1

    content_cols = content_pixels.any(dim=0)
    from_col = torch.where(content_cols)[0].min()
    to_col = torch.where(content_cols)[0].max() + 1

    return image[:, from_row:to_row, from_col:to_col]


def draw_label(
    text: str,
    font: Path,
    font_size: int,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    """Draw a black label on a white background with no border."""
    font = ImageFont.truetype(str(font), font_size)
    width, height = font.getsize(text)
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    image = torch.tensor(np.array(image) / 255, dtype=torch.float32, device=device)
    return crop_whitespace(rearrange(image, "h w c -> c h w"))


def add_label(
    image: Float[Tensor, "3 width height"],
    label: str,
    font: Path = Path("assets/Inter-Regular.otf"),
    font_size: int = 24,
) -> Float[Tensor, "3 width_with_label height_with_label"]:
    return vcat([draw_label(label, font, font_size, image.device), image], align="left")
