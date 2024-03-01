"""
render_mesh.py

A script for rendering meshes.
"""

from dataclasses import dataclass, field, fields
import json
from PIL import Image
from pathlib import Path
from typing import Any, List, Optional, Union

import imageio
from jaxtyping import jaxtyped
import numpy as np
import torch
from typeguard import typechecked
import tyro
import wandb

from src.utils.geometry_utils import load_obj
from src.utils.vis_utils import render_mesh_360


@dataclass
class Args:

    mesh_file: Path
    """A mesh file to render"""
    out_dir: Path
    """Output directory"""
    device: torch.device = torch.device("cuda:0")
    """Device to use"""
    wandb_grp: str = "render_mesh"
    """Wandb group name"""

    n_step: int = 4
    """Number of steps to rotate the camera"""
    radius: float = 1.5
    """Radius of the camera"""


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:
    
    assert args.mesh_file.exists(), f"Mesh file {args.mesh_file} does not exist."
    (
        v, f, vcs, uvs, vns, tex_inds, vn_inds, tex
    ) = load_obj(args.mesh_file)
    v = torch.from_numpy(v).to(args.device)
    f = torch.from_numpy(f).to(args.device)
    uvs = torch.from_numpy(uvs).to(args.device)
    tex_inds = torch.from_numpy(tex_inds).to(args.device)
    tex = torch.from_numpy(tex).to(args.device)
    
    # lookup mesh name
    metadata_file = args.mesh_file.parent / "metadata.json"
    if metadata_file.exists():
        with open(args.mesh_file.parent / "metadata.json", "r") as f:
            mesh_metadata = json.load(f)
            mesh_name = mesh_metadata["object_name"]
    else:
        # TODO: remove this line after rendering legacy meshes
        mesh_name = args.mesh_file.parents[1].stem

    # create output directory
    args.out_dir.mkdir(exist_ok=True, parents=True)
    print(f"Output directory: {args.out_dir}")

    wandb_name = f"mesh-{mesh_name}"
    wandb.init(
        project="apap_pp",
        group=args.wandb_grp,
        name=wandb_name,
        save_code=True,
    )

    #  render and save images
    imgs = render_mesh_360(
        v, f, uvs, tex_inds, tex,
        n_step=args.n_step, radius=args.radius,
    )
    imgs_cat = np.concatenate(imgs, axis=1)

    for i, img in enumerate(imgs):
        img = Image.fromarray(img)
        img.save(args.out_dir / f"render-{i:03d}.png")
    imgs_cat = Image.fromarray(imgs_cat)
    imgs_cat.save(args.out_dir / "render-all-views.png")
    wandb.log({"render": wandb.Image(imgs_cat)})


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
