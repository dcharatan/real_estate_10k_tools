from typing import Tuple

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int64
from torch import Tensor


def homogenize_points(
    points: Float[Tensor, "*batch n"],
) -> Float[Tensor, "*batch n+1"]:
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(
    vectors: Float[Tensor, "*batch n"],
) -> Float[Tensor, "*batch n+1"]:
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_xyz: Float[Tensor, "*#batch 4"],
    transformation: Float[Tensor, "*#batch 4 4"],
) -> torch.Tensor:
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(transformation, homogeneous_xyz, "... i j, ... j -> ... i")


def transform_cam2world(
    homogeneous_camera_xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_camera_xyz, cam2world)


def project(
    points: Float[Tensor, "batch point 3"],
    intrinsics: Float[Tensor, "batch 3 3"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> Float[Tensor, "batch point 2"]:
    points = points / (points[..., -1:] + epsilon)
    points = einsum(intrinsics, points, "b i j, b p j -> b p i")
    return points[..., :2]


def unproject(
    coordinates_xy: Float[Tensor, "batch ray 2"],
    z: Float[Tensor, "batch ray"],
    intrinsics: Float[Tensor, "batch 3 3"],
) -> Float[Tensor, "batch ray 3"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates_xy = homogenize_points(coordinates_xy)
    coordinates_xyz = einsum(
        intrinsics.inverse(), coordinates_xy, "b i j, b r j -> b r i"
    )

    # Apply the supplied depth values.
    return coordinates_xyz * z[..., None]


def get_world_rays(
    coordinates_xy: Float[Tensor, "batch ray 2"],
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
) -> Tuple[
    Float[Tensor, "batch ray 3"],  # origins
    Float[Tensor, "batch ray 3"],  # directions
]:
    # Extract ray origins.
    origins = extrinsics[..., :3, 3]

    # Get camera-space ray directions.
    directions = unproject(
        coordinates_xy,
        torch.ones_like(coordinates_xy[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(
        directions,
        rearrange(extrinsics, "b h w -> b () h w"),
    )

    # Tile the ray origins to have the same shape as the ray directions.
    _, num_rays, _ = directions.shape
    origins = repeat(origins, "b xyz -> b r xyz", r=num_rays)

    return origins, directions[..., :3]


def sample_image_grid(
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
) -> Tuple[
    Float[Tensor, "height width 2"],  # (x, y) coordinates
    Int64[Tensor, "height width 2"],  # (row, col) indices
]:
    """Get normalized (range 0 to 1) xy coordinates and row-col indices for an image."""

    # Each entry is a pixel-wise (row, col) coordinate.
    row = torch.arange(height, device=device)
    col = torch.arange(width, device=device)
    selector = torch.stack(torch.meshgrid(row, col, indexing="ij"), dim=-1)

    # Each entry is a spatial (x, y) coordinate in the range (0, 1).
    x = (col + 0.5) / width
    y = (row + 0.5) / height
    coordinates = torch.stack(torch.meshgrid(x, y, indexing="xy"), dim=-1)

    return coordinates, selector
