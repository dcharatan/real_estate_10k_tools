from pathlib import Path
from string import ascii_letters, digits, punctuation

import numpy as np
import torch
from einops import rearrange
from jaxtyping import Float
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor

from .layout import vcat

EXPECTED_CHARACTERS = digits + punctuation + ascii_letters


def draw_label(
    text: str,
    font: Path,
    font_size: int,
    device: torch.device = torch.device("cpu"),
) -> Float[Tensor, "3 height width"]:
    """Draw a black label on a white background with no border."""
    font = ImageFont.truetype(str(font), font_size)
    width, _ = font.getsize(text)
    _, height = font.getsize(EXPECTED_CHARACTERS)
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=font, fill="black")
    image = torch.tensor(np.array(image) / 255, dtype=torch.float32, device=device)
    return rearrange(image, "h w c -> c h w")


def add_label(
    image: Float[Tensor, "3 width height"],
    label: str,
    font: Path = Path("assets/Inter-Regular.otf"),
    font_size: int = 24,
) -> Float[Tensor, "3 width_with_label height_with_label"]:
    return vcat(
        draw_label(label, font, font_size, image.device),
        image,
        align="left",
        gap=4,
    )
