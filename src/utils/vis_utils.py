"""
vis_utils.py

A collection of utility functions for visualization.
"""

import os
os.environ["PYOPENGL_PLATFORM"] = "egl"  # Enable headless rendering

from typing import List, Optional

from jaxtyping import Shaped, jaxtyped
import pyrender
import trimesh
from typeguard import typechecked

import numpy as np
import torch

from ..renderer.camera import (
    compute_lookat_mat,
    compute_proj_mat,
    sample_trajectory_along_upper_hemisphere,
)
from ..renderer.nvdiffrast import render, render_depth


@jaxtyped(typechecker=typechecked)
def render_mesh_360(
    v: Shaped[torch.Tensor, "V 3"],
    f: Shaped[torch.Tensor, "F 3"],
    uvs: Optional[Shaped[torch.Tensor, "..."]] = None,
    tex_inds: Optional[Shaped[torch.Tensor, "..."]] = None,
    tex: Optional[Shaped[torch.Tensor, "..."]] = None,
    vc: Optional[Shaped[torch.Tensor, "..."]] = None,
    origin: Shaped[torch.Tensor, "3"] = torch.tensor([0.0, 0.0, 0.0]),
    aspect_ratio: float = 1.0,
    fov: float = 53.14,
    near: float = 0.1,
    far: float = 10.0,
    img_height: int = 512,
    img_width: int = 512,
    n_step: int = 4,
    radius: float = 1.0,
    elev: float = 0.0,
    bg_color: Shaped[torch.Tensor, "3"] = torch.ones(3)
) -> List[Shaped[np.ndarray, "H W C"]]:
    """
    Renders 360 views of a mesh.

    By default, assumes that the mesh is at the origin.
    """
    device = v.device

    cam_locs = sample_trajectory_along_upper_hemisphere(
        radius, elev, n_step
    )
    cam_locs = cam_locs.to(v.device)

    imgs = []
    
    for cam_loc in cam_locs:
        # compute camera matrices
        cam2world = compute_lookat_mat(
            cam_loc.type(torch.float32).to(device),
            origin.type(torch.float32).to(device),
        )
        proj_mat = compute_proj_mat(
            aspect_ratio, fov, near, far, device=device
        )

        # render
        with torch.no_grad():
            img, _ = render(
                v, f, cam2world, proj_mat, img_height, img_width,
                vc=vc, uvs=uvs, tex_inds=tex_inds, tex=tex, ss_scale=4.0,
                bg_color=bg_color.to(v.device),
            )
            img = (img.detach().cpu().numpy() * 255.0).astype(np.uint8)
        imgs.append(img)

    return imgs

@jaxtyped(typechecker=typechecked)
def render_mesh_depth_360(
    v: Shaped[torch.Tensor, "V 3"],
    f: Shaped[torch.Tensor, "F 3"],
    origin: Shaped[torch.Tensor, "3"] = torch.tensor([0.0, 0.0, 0.0]),
    aspect_ratio: float = 1.0,
    fov: float = 53.14,
    near: float = 0.1,
    far: float = 10.0,
    img_height: int = 512,
    img_width: int = 512,
    n_step: int = 4,
    radius: float = 1.0,
    elev: float = 0.0,
) -> List[Shaped[np.ndarray, "H W C"]]:
    """
    Renders 360 views of a mesh.

    By default, assumes that the mesh is at the origin.
    """
    device = v.device

    cam_locs = sample_trajectory_along_upper_hemisphere(
        radius, elev, n_step
    )
    cam_locs = cam_locs.to(v.device)

    depths = []
    
    for cam_loc in cam_locs:
        # compute camera matrices
        cam2world = compute_lookat_mat(
            cam_loc.type(torch.float32).to(device),
            origin.type(torch.float32).to(device),
        )
        proj_mat = compute_proj_mat(
            aspect_ratio, fov, near, far, device=device
        )

        # render
        with torch.no_grad():
            depth, _, _ = render_depth(
                v, f, cam2world, proj_mat, img_height, img_width,
            )
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = (depth.detach().cpu().numpy() * 255.0).astype(np.uint8)
        depths.append(depth)

    return depths

@jaxtyped(typechecker=typechecked)
def render_mesh_with_markers_360(
    v: Shaped[torch.Tensor, "V 3"],
    f: Shaped[torch.Tensor, "F 3"],
    handle_inds: Shaped[torch.Tensor, "N"],
    handle_pos: Shaped[torch.Tensor, "N 3"],
    origin: Shaped[torch.Tensor, "3"] = torch.tensor([0.0, 0.0, 0.0]),
    aspect_ratio: float = 1.0,
    fov: float = 53.14,
    near: float = 0.1,
    far: float = 10.0,
    img_height: int = 512,
    img_width: int = 512,
    n_step: int = 4,
    radius: float = 1.0,
    elev: float = 0.0,
) -> List[Shaped[np.ndarray, "H W C"]]:
    """
    Renders 360 views of a mesh.

    By default, assumes that the mesh is at the origin.

    Additionally, it supports rendering markers located at deformation
    handles and their target positions for visualiation.
    """
    device = v.device

    cam_locs = sample_trajectory_along_upper_hemisphere(
        radius, elev, n_step
    )
    cam_locs = cam_locs.to(device)

    imgs = []

    # create scene
    scene = pyrender.Scene()
    mesh = trimesh.Trimesh(
        v.detach().cpu().numpy(),
        f.detach().cpu().numpy(),
    )
    mesh = pyrender.Mesh.from_trimesh(mesh)
    scene.add(mesh)

    # create pin-hole camera
    cam = pyrender.PerspectiveCamera(
        yfov=(fov * np.pi / 180.0),
        aspectRatio=aspect_ratio,
        znear=near,
        zfar=far,
    )

    for cam_loc in cam_locs:

        # compute camera matrices
        cam2world = compute_lookat_mat(
            cam_loc.type(torch.float32).to(device),
            origin.type(torch.float32).to(device),
        )

        # add camera
        cam_ = scene.get_nodes(name="cam")
        if len(cam_) > 0:
            assert len(cam_) == 1, "More than one camera in the scene!"
            scene.remove_node(list(cam_)[0])
        scene.add(
            cam,
            pose=cam2world.detach().cpu().numpy(),
            name="cam",
        )

        # add light
        light_ = scene.get_nodes(name="light")
        if len(light_) > 0:
            assert len(light_) == 1, "More than one light in the scene!"
            scene.remove_node(list(light_)[0])
        light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=1.0)
        light_pose = np.eye(4)
        light_pose[:3, 3] = cam2world.detach().cpu().numpy()[:3, 3]
        scene.add(light, pose=light_pose, name="light")

        # add markers
        assert not handle_inds is None, "handle_inds must be provided!"
        assert not handle_pos is None, "handle_pos must be provided!"

        # place markers at handle
        marker_h_ = scene.get_nodes(name="marker_h")
        if len(marker_h_) > 0:
            assert len(marker_h_) == 1, "More than one marker_h in the scene!"
            scene.remove_node(list(marker_h_)[0])
        marker_h = trimesh.creation.uv_sphere(radius=0.02)
        marker_h.visual.vertex_colors = [1.0, 0.0, 0.0, 1.0]
        h_transforms = np.tile(
            np.eye(4), (len(handle_inds), 1, 1)
        )
        h_transforms[:, :3, 3] = v[
            handle_inds.detach().cpu().numpy(), ...
        ].detach().cpu().numpy()
        marker_h = pyrender.Mesh.from_trimesh(
            marker_h, poses=h_transforms
        )
        scene.add(marker_h, name="marker_h")

        # place markers at target position
        marker_t_ = scene.get_nodes(name="marker_t")
        if len(marker_t_) > 0:
            assert len(marker_t_) == 1, "More than one marker_t in the scene!"
            scene.remove_node(list(marker_t_)[0])
        marker_t = trimesh.creation.uv_sphere(radius=0.02)
        marker_t.visual.vertex_colors = [0.0, 1.0, 0.0, 1.0]
        t_transforms = np.tile(
            np.eye(4), (len(handle_inds), 1, 1)
        )
        t_transforms[:, :3, 3] = handle_pos.detach().cpu().numpy()
        marker_t = pyrender.Mesh.from_trimesh(
            marker_t, poses=t_transforms
        )
        scene.add(marker_t, name="marker_t")

        # render
        renderer = pyrender.OffscreenRenderer(img_width, img_height)
        try:
            img, _ = renderer.render(
                scene,
                # flags=pyrender.constants.RenderFlags.ALL_WIREFRAME | pyrender.RenderFlags.FLAT,
                flags=pyrender.constants.RenderFlags.ALL_SOLID,
            )
        except Exception as e:
            print(str(e))
            img = np.zeros((img_height, img_width, 3), dtype=np.float32)
        img = (img * 255.0).astype(np.uint8)
        imgs.append(img)

        # cleanup
        renderer.delete()

    return imgs