"""
camera.py

Functions for building cameras.
"""

from jaxtyping import Float, jaxtyped
import numpy as np
import torch
from typeguard import typechecked

from .utils import homogenize_matrix


@jaxtyped(typechecker=typechecked)
def compute_proj_mat(
    aspect_ratio: float,
    fov: float,
    near: float,
    far: float,
    device: torch.device,
) -> Float[torch.Tensor, "4 4"]:
    """
    Builds the OpenGL projection matrix for the current camera.
    """
    aspect_ratio = torch.tensor(aspect_ratio, device=device)
    fov_rad = torch.tensor(fov, device=device) * np.pi / 180.0
    near = torch.tensor(near, device=device)
    far = torch.tensor(far, device=device)

    proj_mat = torch.zeros(4, 4, device=device)
    proj_mat[0, 0] = 1.0 / (aspect_ratio * torch.tan(fov_rad / 2.0))
    proj_mat[1, 1] = 1.0 / torch.tan(fov_rad / 2.0)
    proj_mat[2, 2] = - (near + far) / (far - near)
    proj_mat[2, 3] = - (2.0 * near * far) / (far - near)
    proj_mat[3, 2] = -1.0

    return proj_mat

@jaxtyped(typechecker=typechecked)
def compute_lookat_mat(
    cam_pos: Float[torch.Tensor, "3"],
    origin: Float[torch.Tensor, "3"],
    up: Float[torch.Tensor, "3"] = torch.tensor([0.0, 1.0, 0.0]),
) -> Float[torch.Tensor, "4 4"]:
    """
    Computes a camera pose matrix given the camera position and the origin.

    The coordinate frame is defined in a way that the look-at vector,
    which is the vector from the camera position to the origin, is aligned
    with the negative z-axis of the camera coordinate frame.
    """
    # compute z-axis from the inverted "look-at" vector
    z_axis = cam_pos - origin
    z_axis = z_axis / torch.sqrt(torch.sum(z_axis ** 2))

    # compute x-axis by finding the vector orthogonal to up and z-axis
    up = up.to(cam_pos.device)
    x_axis = torch.cross(up, z_axis)
    x_axis = x_axis / torch.sqrt(torch.sum(x_axis ** 2))

    # compute y-axis by finding the vector orthogonal to z-axis and x-axis
    y_axis = torch.cross(z_axis, x_axis)
    y_axis = y_axis / torch.sqrt(torch.sum(y_axis ** 2))

    # construct the camera pose matrix
    cam2world = torch.stack([x_axis, y_axis, z_axis], dim=1)
    cam2world = torch.cat([cam2world, cam_pos[:, None]], dim=1)

    # homogenize the matrix
    cam2world = homogenize_matrix(cam2world)

    return cam2world

@jaxtyped(typechecker=typechecked)
def convert_spherical_to_cartesian(
    radius: float,
    azimuth: float,
    elevation: float,
) -> Float[torch.Tensor, "3"]:
    """
    Converts a point in spherical coordinates to cartesian coordinates.

    Args:
        radius: The radius of the sphere.
        azimuth: The azimuth angle in radian.
        elevation: The elevation angle in radian.
    
    Returns:
        The Cartesian coordinate (x, y, z) of the point specified by
        the given spherical coordinates.
    """

    # check arguments
    assert radius > 0.0, f"{radius:.3f}"
    assert 0 <= azimuth <= 2 * np.pi, f"{azimuth:.3f}"
    assert -np.pi / 2 <= elevation <= np.pi / 2, f"{elevation:.3f}"

    theta: float = np.pi / 2 - elevation
    phi: float = azimuth

    x = radius * np.sin(theta) * np.sin(phi)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta) * np.cos(phi)
    position = torch.tensor([x, y, z], dtype=torch.float32)

    return position

@jaxtyped(typechecker=typechecked)
def sample_location_on_sphere(
    radius: float
) -> Float[torch.Tensor, "3"]:
    """Randomly samples a point over a sphere"""

    # sample azimuth and elevation angles
    azimuth = float(np.random.uniform(0, 2 * np.pi))
    elevation = float(np.random.uniform(-np.pi / 2, np.pi / 2))

    # compute Cartesian coordinates
    position = convert_spherical_to_cartesian(
        radius, azimuth, elevation
    )

    return position

@jaxtyped(typechecker=typechecked)
def sample_trajectory_along_upper_hemisphere(
    radius: float,
    elevation: float,
    n_step: int,
) -> Float[torch.Tensor, "num_step 3"]:
    """
    Samples camera positions along the upper hemisphere of a sphere.

    Args:
        radius: The radius of the sphere.
        elevation: The elevation angle in radian.
        num_step: The number of azimuth steps to sample.
    """

    # sample azimuth values
    azimuths = torch.linspace(0, 2 * np.pi, n_step + 1)[:-1]

    # compute Cartesian coordinates at each sample point
    positions = torch.zeros((n_step, 3))
    for index, azimuth in enumerate(azimuths):
        position = convert_spherical_to_cartesian(
            radius, azimuth.item(), elevation
        )
        positions[index] = position

    return positions
