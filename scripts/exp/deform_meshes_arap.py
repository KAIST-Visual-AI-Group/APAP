"""
deform_meshes_arap.py

A script for deforming meshes with ARAP.
"""

from dataclasses import dataclass, fields
import json
from PIL import Image
from pathlib import Path

import imageio
from jaxtyping import jaxtyped
import numpy as np
import torch
from typeguard import typechecked
import tyro
import wandb

from src.utils.geometry_utils import (
    deform_arap,
    load_obj,
    save_obj,
)
from src.utils.random_utils import seed_everything
from src.utils.vis_utils import (
    render_mesh_360,
    render_mesh_with_markers_360,
)


@dataclass
class Args:

    mesh_file: Path
    """A mesh file to deform"""
    handle_file: Path
    """A file holding handle indices and target positions"""
    anchor_file: Path
    """A file holding anchor indices and target positions"""
    out_dir: Path
    """Output directory"""
    wandb_grp: str = "deform_meshes_arap"
    """W&B group name"""

    # Visualization config
    radius: float = 2.0
    """Camera radius"""

    # Experiment config
    n_step: int = 1
    """Number of intermediate steps between the source and deformed states"""
    device: torch.device = torch.device("cuda")
    """Device to use"""
    seed: int = 2024
    """Random seed"""


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    # reproducibility
    seed_everything(args.seed)

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {str(args.out_dir.resolve())}")

    # Save experiment config
    with open(args.out_dir / "config.txt", mode="w") as file:
        for item in fields(args):
            name = item.name
            value = getattr(args, name)
            file.write(f"{name}: {value}\n")

    # Load mesh & metadata
    assert args.mesh_file.exists(), f"Not found: {str(args.mesh_file)}"
    (
        v, f, vcs, uvs, vns, tex_inds, vn_inds, tex
    ) = load_obj(args.mesh_file)
    v = v.astype(np.float32)
    f = f.astype(np.int32)

    with open(args.mesh_file.parent / "metadata.json") as file:
        mesh_metadata = json.load(file)

    # Load handles
    handle_inds, handle_pos = [], []
    with open(args.handle_file, "r") as file:
        for l in file.readlines():
            ind, x, y, z = l.split()
            handle_inds.append(int(ind))
            handle_pos.append([float(x), float(y), float(z)])
    handle_inds = np.array(handle_inds, dtype=np.int32)
    handle_pos = np.array(handle_pos, dtype=np.float32)

    # Load anchors
    anchor_inds, anchor_pos = [], []
    with open(args.anchor_file, "r") as file:
        for l in file.readlines():
            ind, x, y, z = l.split()
            anchor_inds.append(int(ind))
            anchor_pos.append([float(x), float(y), float(z)])
    anchor_inds = np.array(anchor_inds, dtype=np.int32)
    anchor_pos = np.array(anchor_pos, dtype=np.float32)

    # Combine handles and anchors
    const_inds = np.concatenate([handle_inds, anchor_inds])
    const_pos = np.concatenate([handle_pos, anchor_pos], axis=0)

    # Initial positions of handles and anchors
    init_const_pos = v[const_inds.tolist(), ...]

    # Initialize W&B
    handle_id = str(args.handle_file.parent.stem)
    anchor_id = str(args.anchor_file.parent.stem)
    wandb_name = (
        f"mesh-{mesh_metadata['object_name']}_handle-{handle_id}_anchor-{anchor_id}"
    )
    wandb.init(
        project="apap_pp",
        group=args.wandb_grp,
        name=wandb_name,
        save_code=True,
    )

    # Initialize video writer
    img_dir = args.out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    vid_dir = args.out_dir / "video"
    vid_dir.mkdir(parents=True, exist_ok=True)
    vid_writer = imageio.get_writer(
        vid_dir / "deform.mp4",
        mode="I",
        format="FFMPEG",
        fps=12,
        macro_block_size=1,
    )

    # Deform the mesh with given constraints
    for i in range(args.n_step + 1):
        
        # Compute current constraint positions
        curr_const_pos = (1 - i / args.n_step) * init_const_pos + (i / args.n_step) * const_pos

        # Deform the mesh
        v_new = deform_arap(
            v, f, const_inds, curr_const_pos
        )
        v_new = v_new.astype(np.float32)

        # Render the current frame
        curr_rgbs = render_mesh_360(
            torch.from_numpy(v_new).to(args.device),
            torch.from_numpy(f).to(args.device),
            torch.from_numpy(uvs).to(args.device),
            torch.from_numpy(tex_inds).to(args.device),
            torch.from_numpy(tex).to(args.device),
            n_step=4,
            radius=args.radius,
        )
        curr_rgbs = np.concatenate(curr_rgbs, axis=1)
        curr_depths = render_mesh_with_markers_360(
            torch.from_numpy(v_new).to(args.device),
            torch.from_numpy(f).to(args.device),
            torch.from_numpy(const_inds).to(args.device),
            torch.from_numpy(curr_const_pos).to(args.device),
            n_step=4,
            radius=args.radius,
        )
        curr_depths = np.concatenate(curr_depths, axis=1)
        curr_img = np.concatenate([curr_rgbs, curr_depths], axis=0)

        # Write rendered frame
        vid_writer.append_data(curr_img)
        curr_img = Image.fromarray(curr_img)
        curr_img.save(img_dir / f"frame_{i:04d}.png")
        wandb.log(
            {
                f"eval/frames": wandb.Image(curr_img),
            }
        )

    vid_writer.close()

    # Log video
    wandb.log(
        {
            "eval/video": wandb.Video(str(vid_dir / "deform.mp4")),
        }
    )

    # =================================================================================
    # Visualize
    
    # Initial mesh
    init_imgs = render_mesh_360(
        torch.from_numpy(v).to(args.device),
        torch.from_numpy(f).to(args.device),
        torch.from_numpy(uvs).to(args.device),
        torch.from_numpy(tex_inds).to(args.device),
        torch.from_numpy(tex).to(args.device),
        n_step=4,
        radius=args.radius,
    )
    init_imgs = np.concatenate(init_imgs, axis=1)

    # Initial mesh with markers
    init_marker_imgs = render_mesh_with_markers_360(
        torch.from_numpy(v).to(args.device),
        torch.from_numpy(f).to(args.device),
        torch.from_numpy(const_inds).to(args.device),
        torch.from_numpy(const_pos).to(args.device),
        n_step=4,
        radius=args.radius,
    )
    init_marker_imgs = np.concatenate(init_marker_imgs, axis=1)

    init_imgs = np.concatenate([init_imgs, init_marker_imgs], axis=0)

    # Deformed mesh
    deform_imgs = render_mesh_360(
        torch.from_numpy(v_new).to(args.device),
        torch.from_numpy(f).to(args.device),
        torch.from_numpy(uvs).to(args.device),
        torch.from_numpy(tex_inds).to(args.device),
        torch.from_numpy(tex).to(args.device),
        n_step=4,
        radius=args.radius,
    )
    deform_imgs = np.concatenate(deform_imgs, axis=1)

    # Deformed mesh with markers
    deform_marker_imgs = render_mesh_with_markers_360(
        torch.from_numpy(v_new).to(args.device),
        torch.from_numpy(f).to(args.device),
        torch.from_numpy(const_inds).to(args.device),
        torch.from_numpy(const_pos).to(args.device),
        n_step=4,
        radius=args.radius,
    )
    deform_marker_imgs = np.concatenate(deform_marker_imgs, axis=1)

    deform_imgs = np.concatenate([deform_imgs, deform_marker_imgs], axis=0)

    init_imgs = Image.fromarray(init_imgs)
    deform_imgs = Image.fromarray(deform_imgs)
    init_imgs.save(args.out_dir / "init_imgs.png")
    deform_imgs.save(args.out_dir / "deform_imgs.png")
    
    wandb.log(
        {
            "eval/init_img": wandb.Image(init_imgs),    
            "eval/deform_img": wandb.Image(deform_imgs),
        }
    )
    # =================================================================================

    # =================================================================================
    # Save deformed mesh
    mesh_dir = args.out_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / "deformed.obj"
    save_obj(
        mesh_path,
        v_new,
        f,
        uvs=uvs,
        tex_inds=tex_inds,
        tex=tex,
    )
    print(f"Saved mesh at: {str(mesh_path)}")

    print(f"Done! Logs can be found at: {str(args.out_dir.resolve())}")


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
