import itertools
from typing import Iterable, Literal, Tuple, TypedDict

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.data.dataloader import default_collate

from .geometry import homogenize_points, homogenize_vectors, project


def _is_in_bounds(
    xy: Float[Tensor, "*batch 2"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified XY coordinates are within the normalized image plane,
    which has a range from 0 to 1 in each direction.
    """
    return (xy >= -epsilon).all(dim=-1) & (xy <= 1 + epsilon).all(dim=-1)


def _is_in_front_of_camera(
    xyz: Float[Tensor, "*batch 3"],
    epsilon: float = 1e-6,
) -> Bool[Tensor, " *batch"]:
    """Check whether the specified points in camera space are in front of the camera."""
    return xyz[..., -1] > -epsilon


class PointProjection(TypedDict):
    t: Float[Tensor, " *batch"]  # ray parameter, as in xyz = origin + t * direction
    xy: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # A "valid" projection satisfies two conditions:
    # 1. It is in front of the camera (i.e., its 3D Z coordinate is positive).
    # 2. It is within the image frame (i.e., its 2D coordinates are between 0 and 1).
    valid: Bool[Tensor, " *batch"]


def _intersect_image_coordinate(
    intrinsics: Float[Tensor, "*#batch 3 3"],
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    dimension: Literal["x", "y"],
    coordinate_value: float,
) -> PointProjection:
    """Compute the intersection of the projection of a camera-space ray with a line
    that's parallel to the image frame, either horizontally or vertically.
    """

    # Define shorthands.
    dim = "xy".index(dimension)
    other_dim = 1 - dim
    fs = intrinsics[..., dim, dim]  # focal length, same coordinate
    fo = intrinsics[..., other_dim, other_dim]  # focal length, other coordinate
    cs = intrinsics[..., dim, 2]  # principal point, same coordinate
    co = intrinsics[..., other_dim, 2]  # principal point, other coordinate
    os = origins[..., dim]  # ray origin, same coordinate
    oo = origins[..., other_dim]  # ray origin, other coordinate
    ds = directions[..., dim]  # ray direction, same coordinate
    do = directions[..., other_dim]  # ray direction, other coordinate
    oz = origins[..., 2]  # ray origin, z coordinate
    dz = directions[..., 2]  # ray direction, z coordinate
    c = (coordinate_value - cs) / fs  # coefficient (computed once and factored out)

    # Compute the value of t at the intersection.
    # Note: Infinite values of t are fine. No need to handle division by zero.
    t_numerator = c * oz - os
    t_denominator = ds - c * dz
    t = t_numerator / t_denominator

    # Compute the value of the other coordinate at the intersection.
    # Note: Infinite coordinate values are fine. No need to handle division by zero.
    coordinate_numerator = fo * (oo * (c * dz - ds) + do * (os - c * oz))
    coordinate_denominator = dz * os - ds * oz
    coordinate_other = co + coordinate_numerator / coordinate_denominator
    coordinate_same = torch.ones_like(coordinate_other) * coordinate_value
    xy = [coordinate_same]
    xy.insert(other_dim, coordinate_other)
    xy = torch.stack(xy, dim=-1)
    xyz = origins + t[..., None] * directions

    # These will all have exactly the same batch shape (no broadcasting necessary). In
    # terms of jaxtyping annotations, they all match *batch, not just *#batch.
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz),
    }


def _compare_projections(
    intersections: Iterable[PointProjection],
    reduction: Literal["min", "max"],
) -> PointProjection:
    intersections = {k: v.clone() for k, v in default_collate(intersections).items()}
    t = intersections["t"]
    xy = intersections["xy"]
    valid = intersections["valid"]

    # Make sure out-of-bounds values are not chosen.
    lowest_priority = {
        "min": torch.inf,
        "max": -torch.inf,
    }[reduction]
    t[~valid] = lowest_priority

    # Run the reduction (either t.min() or t.max()).
    reduced, selector = getattr(t, reduction)(dim=0)

    # Index the results.
    return {
        "t": reduced,
        "xy": xy.gather(0, repeat(selector, "... -> () ... xy", xy=2))[0],
        "valid": valid.gather(0, selector[None])[0],
    }


def _compute_point_projection(
    xyz: Float[Tensor, "*#batch 3"],
    t: Float[Tensor, "*#batch"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
) -> PointProjection:
    xy = project(xyz, intrinsics)
    return {
        "t": t,
        "xy": xy,
        "valid": _is_in_bounds(xy) & _is_in_front_of_camera(xyz),
    }


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def project_rays(
    origins: Float[Tensor, "*#batch 3"],
    directions: Float[Tensor, "*#batch 3"],
    extrinsics: Float[Tensor, "*#batch 4 4"],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    epsilon: float = 1e-6,
) -> RaySegmentProjection:
    # Transform the rays into camera space.
    world_to_cam = torch.linalg.inv(extrinsics)
    origins = homogenize_points(origins)
    origins = einsum(world_to_cam, origins, "... i j, ... j -> ... i")
    directions = homogenize_vectors(directions)
    directions = einsum(world_to_cam, directions, "... i j, ... j -> ... i")
    origins = origins[..., :3]
    directions = directions[..., :3]

    # Compute intersections with the image's frame.
    frame_intersections = (
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 0.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "x", 1.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 0.0),
        _intersect_image_coordinate(intrinsics, origins, directions, "y", 1.0),
    )
    frame_intersection_min = _compare_projections(frame_intersections, "min")
    frame_intersection_max = _compare_projections(frame_intersections, "max")

    # Compute the ray's projection at zero depth. If an origin's depth (z value) is
    # within epsilon of zero, this can mean one of two things:
    # 1. The origin is at the camera's position. In this case, use the direction instead
    #    (the ray is probably coming from the camera).
    # 2. The origin isn't at the camera's position, and randomly happens to be on the
    #    plane at zero depth. In this case, its projection is outside the image plane,
    #    and is thus marked as invalid.
    origins_for_projection = origins.clone()
    mask_depth_zero = origins_for_projection[..., -1] < epsilon
    mask_at_camera = origins_for_projection.norm(dim=-1) < epsilon
    origins_for_projection[mask_at_camera] = directions[mask_at_camera]
    projection_at_zero = _compute_point_projection(
        origins_for_projection,
        torch.zeros_like(frame_intersection_min["t"]),
        intrinsics,
    )
    projection_at_zero["valid"][mask_depth_zero & ~mask_at_camera] = False

    # Compute the ray's projection at infinite depth. Using the projection function with
    # directions (vectors) instead of points may seem wonky, but is equivalent to
    # projecting the point at (origins + infinity * directions).
    projection_at_infinity = _compute_point_projection(
        directions,
        torch.ones_like(frame_intersection_min["t"]) * torch.inf,
        intrinsics,
    )

    # Build the result by handling cases for ray intersection.
    result = {
        "t_min": torch.empty_like(projection_at_zero["t"]),
        "t_max": torch.empty_like(projection_at_infinity["t"]),
        "xy_min": torch.empty_like(projection_at_zero["xy"]),
        "xy_max": torch.empty_like(projection_at_infinity["xy"]),
        "overlaps_image": torch.empty_like(projection_at_zero["valid"]),
    }

    for min_valid, max_valid in itertools.product([True, False], [True, False]):
        min_mask = projection_at_zero["valid"] ^ (not min_valid)
        max_mask = projection_at_infinity["valid"] ^ (not max_valid)
        mask = min_mask & max_mask
        min_value = projection_at_zero if min_valid else frame_intersection_min
        max_value = projection_at_infinity if max_valid else frame_intersection_max
        result["t_min"][mask] = min_value["t"][mask]
        result["t_max"][mask] = max_value["t"][mask]
        result["xy_min"][mask] = min_value["xy"][mask]
        result["xy_max"][mask] = max_value["xy"][mask]
        result["overlaps_image"][mask] = (min_value["valid"] & max_value["valid"])[mask]

    return result


class RaySegmentProjection(TypedDict):
    t_min: Float[Tensor, " *batch"]  # ray parameter
    t_max: Float[Tensor, " *batch"]  # ray parameter
    xy_min: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)
    xy_max: Float[Tensor, "*batch 2"]  # image-space xy (normalized to 0 to 1)

    # Whether the segment overlaps the image. If not, the above values are meaningless.
    overlaps_image: Bool[Tensor, " *batch"]


def unnormalize_projection(
    projection: RaySegmentProjection,
    shape: Tuple[int, int],
) -> RaySegmentProjection:
    h, w = shape
    device = projection["xy_min"].device
    multiplier = torch.tensor((w, h), dtype=torch.float32, device=device)
    return {
        "t_min": projection["t_min"].clone(),
        "t_max": projection["t_max"].clone(),
        "xy_min": projection["xy_min"] * multiplier,
        "xy_max": projection["xy_max"] * multiplier,
        "overlaps_image": projection["overlaps_image"].clone(),
    }


def dot(
    x: Float[Tensor, "*#batch dim"],
    y: Float[Tensor, "*#batch dim"],
) -> Float[Tensor, "*batch 1"]:
    return (x * y).sum(dim=-1, keepdim=True)


def get_distance_to_projection(
    projection: RaySegmentProjection,
    samples: Float[Tensor, "batch sample xy"],
) -> Float[Tensor, "batch ray sample"]:
    """Note: To avoid distortion from non-square normalized images, use the
    unnormalize_projection function before using this one.
    """

    xy_min = rearrange(projection["xy_min"], "b r xy -> b r () xy")
    xy_max = rearrange(projection["xy_max"], "b r xy -> b r () xy")
    valid = rearrange(projection["overlaps_image"], "b r -> b r ()")
    samples = rearrange(samples, "b s xy -> b () s xy")

    # Decompose the samples into parallel and perpendicular components.
    ab = xy_max - xy_min
    ac = samples - xy_min
    ab_norm = ab.norm(dim=-1, keepdim=True)
    para_scalar = dot(ab, ac) / (ab_norm + 1e-9)
    para = ab / (ab_norm + 1e-9) * para_scalar
    perp = ac - para

    # Compute endpoint distances.
    near_distance = (para + perp).norm(dim=-1)
    far_distance = (ab - para + perp).norm(dim=-1)
    use_endpoints = ((para_scalar < 0) | (para_scalar > ab_norm))[..., 0]

    # Compute masked distances for the middle (line part) and endpoints.
    middle = (valid & ~use_endpoints) * perp.norm(dim=-1)
    endpoints = (valid & use_endpoints) * torch.minimum(near_distance, far_distance)
    invalid = (~valid * torch.inf).nan_to_num(0)
    return middle + endpoints + invalid
