"""
utils.py

Utility functions for implementing renderer.
"""

from jaxtyping import Float, Shaped, jaxtyped
import torch
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def homogenize_vector(
    v: Shaped[torch.Tensor, "*B 3"],
    x_w: float = 1.0,
) -> Shaped[torch.Tensor, "*B 4"]:
    """
    Homogenize the given 3D coordinates.
    """
    w = torch.full(
        v.shape[:-1] + (1,),
        x_w,
        dtype=v.dtype,
        device=v.device,
    )
    v = torch.cat([v, w], dim=-1)
    return v

@jaxtyped(typechecker=typechecked)
def homogenize_matrix(
    mats: Shaped[torch.Tensor, "*B 3 4"],
) -> Shaped[torch.Tensor, "*B 4 4"]:
    """
    Homogenize the given 3x4 matrices.
    """
    ones = torch.zeros(
        mats.shape[:-2] + (1, 4),
        dtype=mats.dtype,
        device=mats.device,
    )
    ones[..., 0, 3] = 1.0
    mats = torch.cat([mats, ones], dim=-2)
    return mats

@jaxtyped(typechecker=typechecked)
def compute_inverse_affine(
    mat: Float[torch.Tensor, "4 4"],
) -> Float[torch.Tensor, "4 4"]:
    """
    Computes the inverse of the given Affine transformation matrix.
    """
    rot = mat[:3, :3]
    trans = mat[:3, 3]

    inv_mat = torch.zeros_like(mat)
    inv_mat[3, 3] = 1.0
    inv_mat[:3, :3] = rot.t()
    inv_mat[:3, 3] = -rot.t() @ trans

    return inv_mat