"""
geometry_utils.py

Utility functions for geometry processing.
"""

from PIL import Image
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fpsample
import igl
import imageio
from jaxtyping import Int32, Float, Float32, Int32, Shaped, jaxtyped
import numpy as np
from numpy import ndarray
import pymeshlab
import scipy
import torch
import trimesh
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def deform_arap(
    v: Shaped[np.ndarray, "V 3"],
    f: Shaped[np.ndarray, "F 3"],
    handle_inds: Int32[np.ndarray, "N_H"],
    handle_pos: Shaped[np.ndarray, "N_H 3"],
    initial_guess: Optional[Shaped[np.ndarray, "V 3"]] = None,
    n_max_iter: int = 1,
    energy_type: int = igl.ARAP_ENERGY_TYPE_SPOKES,
) -> Shaped[np.ndarray, "V 3"]:
    """
    A simple wrapper around libigl's ARAP implementation.

    Eliminates duplicate handles to avoid factorization issues.
    """
    # remove duplicate keypoints
    handle_inds, unique_idx = np.unique(handle_inds, return_index=True)
    handle_pos = handle_pos[unique_idx]

    # initialize ARAP system
    arap = igl.ARAP(
        v, f, 3, handle_inds, energy_type, max_iter=n_max_iter
    )

    if initial_guess is None:
        v_ = arap.solve(handle_pos, v)
    else:
        v_ = arap.solve(handle_pos, initial_guess)

    return v_

@jaxtyped(typechecker=typechecked)
def deform_biharmonics(
    v: Shaped[np.ndarray, "V 3"],
    f: Shaped[np.ndarray, "F 3"],
    handle_inds: Int32[np.ndarray, "N_H"],
    handle_pos: Shaped[np.ndarray, "N_H 3"],
) -> Shaped[np.ndarray, "V 3"]:
    """Deforms input meshes using biharmonic coordinates."""
    
    # remove duplicate keypoints
    handle_inds, unique_idx = np.unique(handle_inds, return_index=True)
    handle_pos = handle_pos[unique_idx]

    # compute biharmonic weights
    S = [[x] for x in handle_inds.tolist()]
    W = igl.biharmonic_coordinates(
        v.astype(np.float64), f, S, k=2
    )

    v_ = W @ handle_pos.astype(np.float64)

    return v_

@jaxtyped(typechecker=typechecked)
def propagate_quats(
    v: Shaped[np.ndarray, "V 3"],
    f: Shaped[np.ndarray, "F 3"],
    handle_inds: Shaped[np.ndarray, "N_H"],
    anchor_inds: Shaped[np.ndarray, "N_A"],
    handle_quat: Shaped[np.ndarray, "4"],  # TODO: Support distinct quaternion for each handle
    anchor_quat: Shaped[np.ndarray, "4"],  # TODO: Support distinct quaternion for each anchor
    geo_h2all: Optional[Any] = None,
    geo_a2all: Optional[Any] = None,
) -> Tuple[
   Shaped[np.ndarray, "F 4"],
   Shaped[np.ndarray, "F"],
   Shaped[np.ndarray, "F1"],
   Shaped[np.ndarray, "F2"],
   Shaped[np.ndarray, "F3"],
]:
    """
    Propagates the rotation of the handle vertices to the rest of the mesh
    by interpolating the quaternions assigned to the handles and anchors.
    """

    f_inds = set(np.arange(f.shape[0]))

    handle_f = []
    for handle_ind in handle_inds:
        handle_f.append(set(np.where(f == handle_ind)[0]))
    handle_f = set.union(*handle_f)

    anchor_f = []
    for anchor_ind in anchor_inds:
        anchor_f.append(set(np.where(f == anchor_ind)[0]))
    anchor_f = set.union(*anchor_f)

    free_f = f_inds - handle_f - anchor_f

    f_inds = np.array(list(f_inds), dtype=np.int32)
    handle_f = np.array(list(handle_f), dtype=np.int32)
    anchor_f = np.array(list(anchor_f), dtype=np.int32)
    free_f = np.array(list(free_f), dtype=np.int32)
    assert len(f_inds) == (len(handle_f) + len(anchor_f) + len(free_f)), (
        f"""{len(f_inds)} != {len(handle_f)} + {len(anchor_f)} + {len(free_f)}
            = {len(handle_f) + len(anchor_f) + len(free_f)}"""
    )

    # compute handle to all faces
    if geo_h2all is None:
        geo_h2all = igl.exact_geodesic(v, f, handle_inds, handle_inds, ft=f_inds)
    geo_h2all = geo_h2all[len(handle_inds):]

    # compute anchor to all faces
    if geo_a2all is None:
        geo_a2all = igl.exact_geodesic(v, f, anchor_inds, anchor_inds, ft=f_inds)
    geo_a2all = geo_a2all[len(anchor_inds):]

    # interpolate quaternions
    weights = geo_a2all / (geo_a2all + geo_h2all + 1e-8)
    interp_quat = (1 - weights)[:, None] * anchor_quat[None] + weights[:, None] * handle_quat[None]
    interp_quat = interp_quat / np.linalg.norm(interp_quat, axis=1, keepdims=True)

    return interp_quat, f_inds, handle_f, anchor_f, free_f

@jaxtyped(typechecker=typechecked)
def propagate_quats_torch(
    v: Shaped[torch.Tensor, "V 3"],
    f: Shaped[torch.Tensor, "F 3"],
    handle_inds: Shaped[torch.Tensor, "N_H"],
    anchor_inds: Shaped[torch.Tensor, "N_A"],
    handle_quat: Shaped[torch.Tensor, "4"],  # TODO: Support distinct quaternion for each handle
    anchor_quat: Shaped[torch.Tensor, "4"],  # TODO: Support distinct quaternion for each anchor
    geo_h2all: Optional[Shaped[torch.Tensor, "..."]] = None,  # TODO: Type this argument
    geo_a2all: Optional[Shaped[torch.Tensor, "..."]] = None,  # TODO: Type this argument
) -> Tuple[
   Shaped[torch.Tensor, "F 4"],
   Shaped[torch.Tensor, "F"],
   Shaped[torch.Tensor, "F1"],
   Shaped[torch.Tensor, "F2"],
   Shaped[torch.Tensor, "F3"],
]:
    """
    Propagates the rotation of the handle vertices to the rest of the mesh
    by interpolating the quaternions assigned to the handles and anchors.
    """

    f_inds = set(torch.arange(f.shape[0]).tolist())

    handle_f = []
    for handle_ind in handle_inds:
        handle_f.append(set(torch.where(f == handle_ind)[0].tolist()))
    handle_f = set.union(*handle_f)

    anchor_f = []
    for anchor_ind in anchor_inds:
        anchor_f.append(set(torch.where(f == anchor_ind)[0].tolist()))
    anchor_f = set.union(*anchor_f)

    free_f = f_inds - handle_f - anchor_f

    f_inds = torch.tensor(list(f_inds), dtype=torch.int32)
    handle_f = torch.tensor(list(handle_f), dtype=torch.int32)
    anchor_f = torch.tensor(list(anchor_f), dtype=torch.int32)
    free_f = torch.tensor(list(free_f), dtype=torch.int32)
    assert len(f_inds) == (len(handle_f) + len(anchor_f) + len(free_f)), (
        f"""{len(f_inds)} != {len(handle_f)} + {len(anchor_f)} + {len(free_f)}
            = {len(handle_f) + len(anchor_f) + len(free_f)}"""
    )

    # compute handle to all faces
    if geo_h2all is None:
        geo_h2all = igl.exact_geodesic(
            v.clone().detach().cpu().numpy(),
            f.clone().detach().cpu().numpy(),
            handle_inds.clone().detach().cpu().numpy(),
            handle_inds.clone().detach().cpu().numpy(),
            ft=f_inds.clone().detach().cpu().numpy(),
        )
        geo_h2all = torch.from_numpy(geo_h2all).to(v.device)
        geo_h2all = geo_h2all[len(handle_inds):]
    assert len(geo_h2all) == len(f_inds), (
        f"{len(geo_h2all)} != {len(f_inds)}"
    )

    # compute anchor to all faces
    if geo_a2all is None:
        geo_a2all = igl.exact_geodesic(
            v.clone().detach().cpu().numpy(),
            f.clone().detach().cpu().numpy(),
            anchor_inds.clone().detach().cpu().numpy(),
            anchor_inds.clone().detach().cpu().numpy(),
            ft=f_inds.clone().detach().cpu().numpy(),
        )
        geo_a2all = torch.from_numpy(geo_a2all).to(v.device)
        geo_a2all = geo_a2all[len(anchor_inds):]
    assert len(geo_a2all) == len(f_inds), (
        f"{len(geo_a2all)} != {len(f_inds)}"
    )

    # interpolate quaternions
    weights = geo_a2all / (geo_a2all + geo_h2all + 1e-8)

    interp_quat = (1 - weights)[:, None] * anchor_quat[None] + weights[:, None] * handle_quat[None]
    interp_quat = interp_quat / torch.norm(interp_quat, dim=1, keepdim=True)

    return interp_quat, f_inds, handle_f, anchor_f, free_f

@jaxtyped(typechecker=typechecked)
def fps_pointcloud(
    p: Shaped[np.ndarray, "N 3"],
    n_sample: int,
) -> Tuple[Shaped[np.ndarray, "N_S 3"], Int32[np.ndarray, "N_S"]]:
    """
    Samples a point cloud using farthest point sampling.
    """
    inds = fpsample.bucket_fps_kdline_sampling(
        p, n_sample, h=3
    ).astype(np.int32)
    pts = p[inds]

    return pts, inds

@jaxtyped(typechecker=typechecked)
def load_obj(path: Path):
    """
    Loads an .obj file.
    """
    assert path.suffix == ".obj", f"Not an .obj file. Got {path.suffix}"
    assert path.exists(), f"File not found: {str(path)}"

    v = []
    f = []
    vc = None
    uvs = []
    vns = []
    tex_inds = []
    vn_inds = []
    tex = None

    with open(path, "r") as file:

        for line in file.readlines():

            # ===================================================================
            # parse vertex coordinates
            if line.startswith("v "):
                vertices_ = _parse_vertex(line)
                v.append(vertices_)
            # ===================================================================

            # ===================================================================
            # parse faces
            elif line.startswith("f "):
                (
                    face_indices_,
                    tex_coord_indices_,
                    vertex_normal_indices_
                ) = _parse_face(line)
                f.append(face_indices_)
                tex_inds.append(tex_coord_indices_)
                vn_inds.append(vertex_normal_indices_)
            # ===================================================================

            # ===================================================================
            # parse texture coordinates
            elif line.startswith("vt "):
                tex_coordinates_ = _parse_tex_coordinates(line)
                uvs.append(tex_coordinates_)
            # ===================================================================

            # ===================================================================
            # parse vertex normals
            elif line.startswith("vn "):
                vertex_normals_ = _parse_vertex_normal(line)
                vns.append(vertex_normals_)
            # ===================================================================

            else:
                pass  # ignore
    v = np.array(v, dtype=np.float32)
    f = np.array(f, dtype=np.int32)

    # ==========================================================================
    # load texture
    tex = _load_texture_image(path)
    tex_exists = (
        len(uvs) > 0 \
        and tex_inds[0][0] is not None \
        and tex is not None
    )
    if tex_exists:
        uvs = np.array(uvs, dtype=np.float32)
        tex_inds = np.array(tex_inds, dtype=np.int32)
    else:
        uvs = None
        tex_inds = None
        vc = np.ones_like(v) * 0.75
    # ==========================================================================

    # ==========================================================================
    # load vertex normals
    if len(vns) > 0:
        vns = np.array(vns, dtype=np.float32)
    else:
        vns = None

    if vn_inds[0][0] is not None:
        vn_inds = np.array(vn_inds, dtype=np.int32)
    else:
        vn_inds = None
    # ==========================================================================
    
    return v, f, vc, uvs, vns, tex_inds, vn_inds, tex

@typechecked
def _parse_vertex(line: str) -> List[float]:
    coords = [float(x) for x in line.split()[1:]]
    coords = coords[:3]  # ignore the rest
    return coords

@typechecked
def _parse_face(
    line: str,
) -> Tuple[
    List[int],
    Union[List[int], List[None]],
    Union[List[int], List[None]],
]:
    """
    Parses a line starts with 'f' that contains face information.

    NOTE: face indices must be offset by 1 because OBJ files are 1-indexed.
    """

    space_splits = line.split()[1:]

    face_indices = []
    tex_coord_indices = []
    vertex_normal_indices = []

    for space_split in space_splits:
        slash_split = space_split.split("/")

        if len(slash_split) == 1:  # f v1 v2 v3 ...
            face_indices.append(int(slash_split[0]) - 1)
            tex_coord_indices.append(None)
            vertex_normal_indices.append(None)

        elif len(slash_split) == 2:  # f v1/vt1 v2/vt2 v3/vt3 ...
            face_indices.append(int(slash_split[0]) - 1)
            tex_coord_indices.append(int(slash_split[1]) - 1)
            vertex_normal_indices.append(None)

        elif len(slash_split) == 3:  # f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3 ...
            face_indices.append(int(slash_split[0]) - 1)
            tex_coord_index = None
            if slash_split[1].isnumeric():
                tex_coord_index = int(slash_split[1]) - 1
            tex_coord_indices.append(tex_coord_index)
            vertex_normal_indices.append(int(slash_split[2]) - 1)

        else:
            raise NotImplementedError("Unsupported feature")

    return (
        face_indices,
        tex_coord_indices,
        vertex_normal_indices,
    )

@typechecked
def _parse_tex_coordinates(line: str) -> List[float]:
    return [float(x) for x in line.split()[1:]]

@typechecked
def _parse_vertex_normal(line: str) -> List[float]:
    return [float(x) for x in line.split()[1:]]

@jaxtyped(typechecker=typechecked)
def _load_texture_image(
    obj_path: Path,
    vertical_flip: bool = True,
    default_bg_color: Float[ndarray, "3"] = np.zeros(3, dtype=np.float32),
) -> Optional[Float[np.ndarray, "image_height image_width 3"]]:
    """
    Loads the texture image associated with the given .obj file.

    Args:
        obj_path: Path to the .obj file whose texture is being loaded.
        vertical_flip: Whether to flip the texture image vertically.
            This is necessary for rendering systems following OpenGL conventions.
        default_bg_color: The default background color to use 
            if the loaded image has an alpha channel.
            
    Returns:
        A texture image if it exists, otherwise None.
    """
    img_path = obj_path.parent / f"{obj_path.stem}_texture.png"
    tex = None
    if img_path.exists():
        tex = imageio.imread(img_path)

        num_channel = tex.shape[-1]
        if num_channel == 4:  # RGBA
            tex, alpha = np.split(tex, [3], axis=2)
            alpha_mask = (alpha > 0.0).astype(np.float32)
        elif num_channel == 3:  # RGB
            alpha_mask = np.ones([*tex.shape[:2], 1], dtype=np.float32)
        else:
            raise AssertionError(f"Invalid texture image shape: {tex.shape}")

        tex = tex.astype(np.float32) / 255.0
        tex = alpha_mask * tex + \
            (1 - alpha_mask) * default_bg_color

        if vertical_flip:
            tex = np.flip(tex, axis=0).copy()
    return tex

@jaxtyped(typechecker=typechecked)
def save_obj(
    out_path: Path,
    v: Shaped[np.ndarray, "V 3"],
    f: Shaped[np.ndarray, "F 3"],
    vc: Optional[Shaped[np.ndarray, "* 3"]] = None,
    uvs: Optional[Shaped[np.ndarray, "..."]] = None,  # FIXME: Add type annotation
    vns: Optional[Shaped[np.ndarray, "* 3"]] = None,
    tex_inds: Optional[Shaped[np.ndarray, "..."]] = None,  # FIXME: Add type annotation
    vn_inds: Optional[Shaped[np.ndarray, "* 3"]] = None,
    tex: Optional[Shaped[np.ndarray, "..."]] = None,  # FIXME: Add type annotation
    flip_tex: bool = True,
) -> None:
    """
    Saves a triangle mesh as an .obj file.
    """

    assert out_path.parent.exists(), f"Directory not found: {str(out_path.parent)}"
    assert out_path.suffix == ".obj", f"Not an .obj file. Got {out_path.suffix}"

    with open(out_path, "w") as obj_file:
        v_ = v.tolist()
        f_ = f.tolist()

        # check whether mesh has texture
        if (not tex_inds is None) and (not tex is None) and (not uvs is None):
            tex_ = tex
            tex_inds_ = tex_inds.tolist()
            uvs_ = uvs.tolist()
        else:
            tex_ = None
            uvs_ = None
            tex_inds_ = None
        
        # check whether mesh has per-vertex normal
        if (not vn_inds is None) and (not vns is None):
            vns_ = vns.tolist()
            vn_inds_ = vn_inds.tolist()
        else:
            vns_ = None
            vn_inds_ = None 

        obj_file.write("\n# vertices\n")
        for vertex in v_:
            obj_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

        if (not tex_inds is None) and (not tex is None) and (not uvs is None):
            obj_file.write("\n# texture coordinates\n")
            for uv in uvs_:
                obj_file.write(f"vt {uv[0]} {uv[1]}\n")

            if flip_tex:
                tex_ = np.flip(tex_, axis=0)
            if np.max(tex_) <= 1.0:
                tex_ = (tex_ * 255.0).astype(np.uint8)
            imageio.imwrite(
                out_path.parent / f"{out_path.stem}_texture.png",
                tex_,
            )

        if (not vn_inds is None) and (not vns is None):
            obj_file.write("\n# vertex normals\n")
            for vn in vns_:
                obj_file.write(
                    f"vn {vn[0]} {vn[1]} {vn[2]}\n"
                )

        obj_file.write("\n# faces\n")
        for face_index, face in enumerate(f_):
            face_i_str = str(face[0] + 1)
            face_j_str = str(face[1] + 1)
            face_k_str = str(face[2] + 1)

            # TODO: Clean up this code
            if (not tex_inds is None) and (not tex is None) and (not uvs is None):
                face_i_str += f"/{tex_inds_[face_index][0] + 1}"
                face_j_str += f"/{tex_inds_[face_index][1] + 1}"
                face_k_str += f"/{tex_inds_[face_index][2] + 1}"

                if (not vn_inds is None) and (not vns is None):
                    face_i_str += "/"
                    face_j_str += "/"
                    face_k_str += "/"
            else:
                if (not vn_inds is None) and (not vns is None):
                    face_i_str += "//"
                    face_j_str += "//"
                    face_k_str += "//"

            if (not vn_inds is None) and (not vns is None):
                face_i_str += f"{vn_inds_[face_index][0] + 1}"
                face_j_str += f"{vn_inds_[face_index][1] + 1}"
                face_k_str += f"{vn_inds_[face_index][2] + 1}"

            obj_file.write(f"f {face_i_str} {face_j_str} {face_k_str}\n")

@jaxtyped(typechecker=typechecked)
def normalize_mesh(
    v: Union[Shaped[np.ndarray, "V 3"], Shaped[torch.Tensor, "V 3"]],
) -> Union[Shaped[np.ndarray, "V 3"], Shaped[torch.Tensor, "V 3"]]:
    """
    Normalizes a mesh to fit into a unit cube centered at the origin.
    """
    # normalize the vertices
    if isinstance(v, np.ndarray):
        v_min = np.min(v, axis=0, keepdims=True)
        v_max = np.max(v, axis=0, keepdims=True)
        v_range = v_max - v_min
        v_center = (v_min + v_max) / 2
        v_normalized = (v - v_center) / v_range.max()
    else:
        v_min = torch.min(v, dim=0, keepdim=True).values
        v_max = torch.max(v, dim=0, keepdim=True).values
        v_range = v_max - v_min
        v_center = (v_min + v_max) / 2
        v_normalized = (v - v_center) / v_range.max()
 
    return v_normalized

@jaxtyped(typechecker=typechecked)
def cleanup_mesh(
    v: Shaped[ndarray, "V 3"],
    f: Shaped[ndarray, "F 3"],
) -> Tuple[
    Float32[ndarray, "V_ 3"],
    Int32[ndarray, "F_ 3"],
]:
    """
    Applies a series of filters to the input mesh.

    For instance,
    - Duplicate vertex removal
    - Unreference vertex removal
    - Remove isolated pieces
    """

    # create PyMeshLab MeshSet
    mesh = pymeshlab.Mesh(
        vertex_matrix=v.astype(np.float64),
        face_matrix=f.astype(int),
    )
    meshset = pymeshlab.MeshSet()
    meshset.add_mesh(mesh)

    # remove duplicate vertices
    meshset.meshing_remove_duplicate_vertices()

    # remove unreferenced vertices
    meshset.meshing_remove_unreferenced_vertices()

    # remove isolated pieces
    meshset.meshing_remove_connected_component_by_diameter()

    # extract the processed mesh
    mesh_new = meshset.current_mesh()
    vertices_proc = mesh_new.vertex_matrix().astype(np.float32)
    faces_proc = mesh_new.face_matrix()

    return vertices_proc, faces_proc

@jaxtyped(typechecker=typechecked)
def run_loop_subdivision(
    v: Shaped[ndarray, "V 3"],
    f: Shaped[ndarray, "F 3"],
    n_subdiv: int,
):
    v_new, f_new = igl.upsample(v, f, n_subdiv)
    return v_new, f_new
