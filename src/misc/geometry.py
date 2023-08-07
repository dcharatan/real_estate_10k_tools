from typing import Tuple

import torch
from einops import einsum, rearrange, reduce, repeat
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
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_camera_xyz, cam2world)


def transform_world2cam(
    homogeneous_camera_xyz: Float[Tensor, "*#batch 4"],
    cam2world: Float[Tensor, "*#batch 4 4"],
) -> Float[Tensor, "*batch 4"]:
    """Transform points from 3D world coordinates to 3D camera coordinates."""
    return transform_rigid(homogeneous_camera_xyz, cam2world.inverse())


def project(
    points: Float[Tensor, "*#batch 3"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    epsilon: float = torch.finfo(torch.float32).eps,
) -> Float[Tensor, "*batch 2"]:
    points = points / (points[..., -1:] + epsilon)
    points = einsum(intrinsics, points, "... i j, ... j -> ... i")
    return points[..., :2]


def unproject(
    coordinates_xy: Float[Tensor, "*#batch 2"],
    z: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch 3"]:
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates_xy = homogenize_points(coordinates_xy)
    coordinates_xyz = einsum(
        intrinsics.inverse(), coordinates_xy, "... i j, ... j -> ... i"
    )

    # Apply the supplied depth values.
    return coordinates_xyz * z[..., None]


def get_world_rays(
    coordinates_xy: Float[Tensor, "*#batch 2"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> Tuple[
    Float[Tensor, "*batch 3"],  # origins
    Float[Tensor, "*batch 3"],  # directions
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates_xy,
        torch.ones_like(coordinates_xy[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :3]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :3, 3].broadcast_to(directions.shape)

    return origins, directions


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


def sample_training_rays(
    image: Float[Tensor, "batch view channel height width"],
    extrinsics: Float[Tensor, "batch view 4 4"],
    intrinsics: Float[Tensor, "batch view 3 3"],
    num_rays: int,
) -> Tuple[
    Float[Tensor, "batch ray 3"],  # origins
    Float[Tensor, "batch ray 3"],  # directions
    Float[Tensor, "batch ray 3"],  # sampled color
]:
    device = extrinsics.device
    b, v, _, h, w = image.shape

    # Generate all possible target rays.
    xy, _ = sample_image_grid(h, w, device)
    origins, directions = get_world_rays(
        xy,
        rearrange(extrinsics, "b v i j -> b v () () i j"),
        rearrange(intrinsics, "b v i j -> b v () () i j"),
    )
    origins = rearrange(origins, "b v h w xy -> b (v h w) xy", b=b, v=v, h=h, w=w)
    directions = rearrange(directions, "b v h w xy -> b (v h w) xy", b=b, v=v, h=h, w=w)
    pixels = rearrange(image, "b v c h w -> b (v h w) c")

    # Sample random rays.
    num_possible_rays = v * h * w
    ray_indices = torch.randint(num_possible_rays, (b, num_rays), device=device)
    batch_indices = repeat(torch.arange(b, device=device), "b -> b n", n=num_rays)

    return (
        origins[batch_indices, ray_indices],
        directions[batch_indices, ray_indices],
        pixels[batch_indices, ray_indices],
    )


def intersect_rays(
    origins_x: Float[Tensor, "*#batch 3"],
    directions_x: Float[Tensor, "*#batch 3"],
    origins_y: Float[Tensor, "*#batch 3"],
    directions_y: Float[Tensor, "*#batch 3"],
) -> Float[Tensor, "*batch 3"]:
    """Compute the least-squares intersection of rays. Uses the math from here:
    https://math.stackexchange.com/a/1762491/286022
    """

    # Broadcast the rays so their shapes match.
    shape = torch.broadcast_shapes(
        origins_x.shape,
        directions_x.shape,
        origins_y.shape,
        directions_y.shape,
    )
    origins_x = origins_x.broadcast_to(shape)
    directions_x = directions_x.broadcast_to(shape)
    origins_y = origins_y.broadcast_to(shape)
    directions_y = directions_y.broadcast_to(shape)

    # Stack the rays into (2, *shape).
    origins = torch.stack([origins_x, origins_y], dim=0)
    directions = torch.stack([directions_x, directions_y], dim=0)
    dtype = origins.dtype
    device = origins.device

    # Compute n_i * n_i^T - eye(3) from the equation.
    *batch, _ = shape
    n = einsum(directions, directions, "r ... i, r ... j -> r ... i j")
    n = n - torch.eye(3, dtype=dtype, device=device).broadcast_to((2, *batch, 3, 3))

    # Compute the left-hand side of the equation.
    lhs = reduce(n, "r ... i j -> ... i j", "sum")

    # Compute the right-hand side of the equation.
    rhs = einsum(n, origins, "r ... i j, r ... j -> r ... i")
    rhs = reduce(rhs, "r ... i -> ... i", "sum")

    # Left-matrix-multiply both sides by the pseudo-inverse of lhs to find p.
    return torch.linalg.lstsq(lhs, rhs).solution
