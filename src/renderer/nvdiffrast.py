"""
nvdiffrast.py

Wrappers around nvdiffrast.
"""

from typing import Any, Optional, Tuple, Union

from jaxtyping import Float, Int, Shaped, jaxtyped
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from typeguard import typechecked

from .utils import homogenize_vector, compute_inverse_affine


@jaxtyped(typechecker=typechecked)
def render(
    v: Float[torch.Tensor, "V 3"],
    f: Int[torch.Tensor, "F 3"],
    cam2world: Float[torch.Tensor, "4 4"],
    proj_mat: Float[torch.Tensor, "4 4"],
    img_height: int,
    img_width: int,
    vc: Optional[Float[torch.Tensor, "V 3"]] = None,
    uvs: Optional[Float[torch.Tensor, "* 2"]] = None,
    tex_inds: Optional[Int[torch.Tensor, "F 3"]] = None,
    tex: Optional[Float[torch.Tensor, "H_T W_T 3"]] = None,
    vn: Optional[Float[torch.Tensor, "V 3"]] = None,
    vn_inds: Optional[Int[torch.Tensor, "F 3"]] = None,
    bg_color: Float[torch.Tensor, "3"] = torch.ones(3),
    use_opengl: bool = False,
    device: torch.device = torch.device("cuda"),
    ss_scale: float = 1.0,
) -> Tuple[
    Float[torch.Tensor, "H W 3"],
    Union[
        Float[torch.Tensor, "H W 4"],
        Float[torch.Tensor, "H W 0"],
    ],
]:
    """
    Renders the given geometry via differentiable rasterization.

    Args:
        v: Vertices.
        f: Faces.
        cam2world: Camera extrinsic matrix.
        proj_mat: Projection matrix matrix.
        img_height: Image height.
        img_width: Image width.
        uvs: Per-vertex texture coordinates.
        tex_inds: Per-face texture indices.
        tex: Texture image.
        n: Per-vertex normals.
        n_inds: Per-face normal indices.
        bg_color: Background color.
        use_opengl: Flag for using OpenGL backend.
        device: Device where the computation takes place.
        ss_scale: Supersampling scale.
    """
    # initialize rasterize context
    ctx = dr.RasterizeCudaContext(device)
    if use_opengl:
        ctx = dr.RasterizeGLContext(device)

    img_height_ss = img_height
    img_width_ss = img_width
    if ss_scale > 1.0:
        img_height_ss = int(img_height * ss_scale)
        img_width_ss = int(img_width * ss_scale)
    
    # apply view and projection matrices
    v = v[None, ...]
    if not vn is None:
        vn = vn[None, ...]
    v_clip, v_eye, vn_eye, light_eye = apply_view_projection_matrices(
        v, cam2world, proj_mat, vn,  # TODO: add light
    )

    # rasterize
    rast_out, img_derivative = dr.rasterize(
        ctx, v_clip, f, resolution=(img_height_ss, img_width_ss)
    )

    # TODO: add Phong shading when necessary
    has_tex = all([not uvs is None, not tex_inds is None, not tex is None])
    if has_tex:
        uvs_per_pixel, _ = dr.interpolate(uvs, rast_out, tex_inds)
        img = dr.texture(tex[None, ...], uvs_per_pixel)
    else:
        assert not vc is None, "Vertex colors are required when texture is not provided"
        img, _ = dr.interpolate(vc, rast_out, f)

    # anti-aliasing
    img = dr.antialias(img, rast_out, v_clip, f)

    # set backgrouond color
    bg_color = bg_color.to(img.device)
    fg_mask = (rast_out[..., 3:4] > 0.0).float()
    img = img * fg_mask + bg_color[None, ...] * (1.0 - fg_mask)

    # flip images vertically
    # nvdiffrast renders images upside down
    # flipping could be done when defining projection matrices
    # but we flip the images to keep the convention tidy
    # Refer to: https://github.com/NVlabs/nvdiffrast/issues/44
    img = torch.flip(img, dims=[1])
    img_derivative = torch.flip(img_derivative, dims=[1])

    # truncate batch dimension
    img = img[0, ...]
    img_derivative = img_derivative[0, ...]

    # downsample
    if ss_scale > 1.0:
        img = img.permute(2, 0, 1)
        img_derivative = img_derivative.permute(2, 0, 1)

        img = F.interpolate(
            img[None, ...],
            (img_height, img_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, ...]
        img_derivative = F.interpolate(
            img_derivative[None, ...],
            (img_height, img_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, ...]

        img = img.permute(1, 2, 0)
        img_derivative = img_derivative.permute(1, 2, 0)

    return img, img_derivative

@jaxtyped(typechecker=typechecked)
def render_depth(
    v: Float[torch.Tensor, "V 3"],
    f: Int[torch.Tensor, "F 3"],
    cam2world: Float[torch.Tensor, "4 4"],
    proj_mat: Float[torch.Tensor, "4 4"],
    img_height: int,
    img_width: int,
    use_opengl: bool = False,
    device: torch.device = torch.device("cuda"),
    ss_scale: float = 1.0,
) -> Tuple[
    Float[torch.Tensor, "H W 3"],
    Union[
        Float[torch.Tensor, "H W 4"],
        Float[torch.Tensor, "H W 0"],
    ],
    Float[torch.Tensor, "H W 1"],
]:
    """
    Renders the given geometry via differentiable rasterization.

    Args:
        v: Vertices.
        f: Faces.
        cam2world: Camera extrinsic matrix.
        proj_mat: Projection matrix matrix.
        img_height: Image height.
        img_width: Image width.
        uvs: Per-vertex texture coordinates.
        tex_inds: Per-face texture indices.
        tex: Texture image.
        n: Per-vertex normals.
        n_inds: Per-face normal indices.
        bg_color: Background color.
        use_opengl: Flag for using OpenGL backend.
        device: Device where the computation takes place.
        ss_scale: Supersampling scale.
    """
    # initialize rasterize context
    ctx = dr.RasterizeCudaContext(device)
    if use_opengl:
        ctx = dr.RasterizeGLContext(device)

    img_height_ss = img_height
    img_width_ss = img_width
    if ss_scale > 1.0:
        img_height_ss = int(img_height * ss_scale)
        img_width_ss = int(img_width * ss_scale)
    
    # apply view and projection matrices
    v = v[None, ...]
    v_clip, v_eye, vn_eye, light_eye = apply_view_projection_matrices(
        v, cam2world, proj_mat
    )

    # rasterize
    rast_out, dmap_derivative = dr.rasterize(
        ctx, v_clip, f, resolution=(img_height_ss, img_width_ss)
    )

    # interpolate depth values
    v_clip_z = v_clip[..., 2:3]
    dmap, _ = dr.interpolate(
        v_clip_z.contiguous(), rast_out, f
    )

    # anti-aliasing
    dmap = dr.antialias(dmap, rast_out, v_clip, f)

    # set backgrouond color
    # FIXME: Is normalizing depth maps after assigning
    # background colors a good idea?
    bg_color = torch.zeros(3).to(dmap.device)
    fg_mask = (rast_out[..., 3:4] > 0.0).float()
    dmap = dmap * fg_mask + bg_color[None, ...] * (1.0 - fg_mask)

    # flip images vertically
    # nvdiffrast renders images upside down
    # flipping could be done when defining projection matrices
    # but we flip the images to keep the convention tidy
    # Refer to: https://github.com/NVlabs/nvdiffrast/issues/44
    dmap = torch.flip(dmap, dims=[1])
    dmap_derivative = torch.flip(dmap_derivative, dims=[1])
    fg_mask = torch.flip(fg_mask, dims=[1])

    # truncate batch dimension
    dmap = dmap[0, ...]
    dmap_derivative = dmap_derivative[0, ...]
    fg_mask = fg_mask[0, ...]

    # downsample
    if ss_scale > 1.0:
        dmap = dmap.permute(2, 0, 1)
        dmap_derivative = dmap_derivative.permute(2, 0, 1)

        dmap = F.interpolate(
            dmap[None, ...],
            (img_height, img_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, ...]
        dmap_derivative = F.interpolate(
            dmap_derivative[None, ...],
            (img_height, img_width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )[0, ...]

        dmap = dmap.permute(1, 2, 0)
        dmap_derivative = dmap_derivative.permute(1, 2, 0)

    return dmap, dmap_derivative, fg_mask

@jaxtyped(typechecker=typechecked)
def apply_view_projection_matrices(
    v: Union[
        Float[torch.Tensor, "batch_size num_vertices 3"],
        Float[torch.Tensor, "batch_size num_vertices 4"]
    ],
    cam2world: Float[torch.Tensor, "4 4"],
    proj_mat: Float[torch.Tensor, "4 4"],
    vn: Optional[Union[
            Float[torch.Tensor, "batch_size num_normal 3"],
            Float[torch.Tensor, "batch_size num_normal 4"],
        ]] = None,
    light: Any = None,
) -> Tuple[
        Float[torch.Tensor, "batch_size num_vertices 4"],
        Float[torch.Tensor, "batch_size num_vertices 4"],
        Optional[Float[torch.Tensor, "batch_size num_normal 3"]],
        Any,  # TODO: Add appropriate type annotation to 'light'
    ]:
    """
    Transforms the vertices from world space to clip space by
    applying the view and projection matrices.

    Args:
        vertices: The vertices to be transformed.
        view_matrix: The view matrix.
        projection_matrix: The projection matrix.
        vertex_normals: The vertex normals. If provided, the vertex normals are also transformed.
        light: The light source. If provided, the light source is also transformed.

    Returns:
        vertices_clip: The transformed vertices in clip space.
        vertex_normals_eye: The transformed vertex normals in the camera frame.
        light_eye: The light source whose position vector is transformed to the camera frame.
    """
    # transform vertices
    if v.shape[-1] == 3:
        v = homogenize_vector(v, 1.0)
    v_eye = v @ compute_inverse_affine(cam2world).t()
    v_clip = v_eye @ proj_mat.t()

    # transform vertex normals
    vn_eye = None
    if vn is not None:
        if vn.shape[-1] == 3:
            vn = homogenize_vector(vn, 0.0)
        vn_eye = vn @ cam2world
        vn_eye = vn_eye[..., :3]

    # transform light source
    light_eye = None
    if light is not None:
        light_pos = light.position[None, ...]
        light_pos = homogenize_vector(light_pos, 1.0)
        light_position_eye = light_pos @ compute_inverse_affine(cam2world).t()
        light_eye = light.copy()
        light_eye.position = light_position_eye[0, :3]

    return v_clip, v_eye, vn_eye, light_eye
