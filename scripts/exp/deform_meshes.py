"""
deform_meshes.py

A script for mesh deformation experiments.
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

from configs.deform_meshes.pipeline import (
    StableDiffusion2DPipeConfig,
    StableDiffusion3DPipeConfig,
)
from src.geometry.poisson_system import PoissonSystem
from src.guidance.stable_diffusion import StableDiffusionGuidance
from src.renderer.camera import compute_lookat_mat, compute_proj_mat
from src.renderer.nvdiffrast import render, render_depth
from src.utils.geometry_utils import load_obj, save_obj
from src.utils.math_utils import (
    normalize_vs,
    quat_to_mat_torch,
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
    wandb_grp: str = "deform_meshes"
    """W&B group name"""
    lora_dir: Optional[Path] = None
    """Path to LoRA checkpoint"""

    pipe_cfg: Union[
        StableDiffusion2DPipeConfig,
        StableDiffusion3DPipeConfig,
    ] = StableDiffusion3DPipeConfig()
    """Config for the experiment pipeline"""


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    # reproducibility
    seed_everything(args.pipe_cfg.seed)

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {str(args.out_dir.resolve())}")

    # Save experiment config
    # TODO: recursively save guidance config
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
    v = torch.from_numpy(v).to(args.pipe_cfg.device)
    f = torch.from_numpy(f).to(args.pipe_cfg.device)
    uvs = torch.from_numpy(uvs).to(args.pipe_cfg.device)
    tex_inds = torch.from_numpy(tex_inds).to(args.pipe_cfg.device)
    tex = torch.from_numpy(tex).to(args.pipe_cfg.device)
    with open(args.mesh_file.parent / "metadata.json") as file:
        mesh_metadata = json.load(file)

    # Load handles
    handle_inds, handle_pos = [], []
    with open(args.handle_file, "r") as file:
        for l in file.readlines():
            ind, x, y, z = l.split()
            handle_inds.append(int(ind))
            handle_pos.append([float(x), float(y), float(z)])
    handle_inds = torch.tensor(handle_inds).to(args.pipe_cfg.device)
    handle_pos = torch.tensor(handle_pos, dtype=torch.float32).to(args.pipe_cfg.device)

    # Load anchors
    anchor_inds, anchor_pos = [], []
    with open(args.anchor_file, "r") as file:
        for l in file.readlines():
            ind, x, y, z = l.split()
            anchor_inds.append(int(ind))
            anchor_pos.append([float(x), float(y), float(z)])
    anchor_inds = torch.tensor(anchor_inds).to(args.pipe_cfg.device)
    anchor_pos = torch.tensor(anchor_pos, dtype=torch.float32).to(args.pipe_cfg.device)

    # Initialize W&B
    handle_id = str(args.handle_file.parent.stem)
    anchor_id = str(args.anchor_file.parent.stem)
    wandb_name = (
        f"mesh-{mesh_metadata['object_name']}_handle-{handle_id}_anchor-{anchor_id}"
    )
    wandb.init(
        project="apap",
        group=args.wandb_grp,
        name=wandb_name,
        save_code=True,
    )

    # Initialize video writer
    vid_dir = args.out_dir / "video"
    vid_dir.mkdir(parents=True, exist_ok=True)
    vid_writer = imageio.get_writer(
        vid_dir / "optim.mp4",
        format="FFMPEG",
        mode="I",
        fps=24,
        macro_block_size=1,
    )

    # Initialize guidance
    guidance = StableDiffusionGuidance(
        device=args.pipe_cfg.device,
        lora_path=args.lora_dir,
        lora_scale=args.pipe_cfg.lora_scale,
        grad_clamp_val=args.pipe_cfg.clamp_val,
    )
    print(f"Loaded guidance: {type(guidance)}")

    # Determine guidance prompt
    prompt = f"a photo of {mesh_metadata['object_name']}"
    if not args.lora_dir is None:
        ckpt_name = args.lora_dir.stem
        with open(args.lora_dir / "metadata.json") as file:
            lora_metadata = json.load(file)
        prompt = (
            f"a photo of {lora_metadata['special_token']} {lora_metadata['object_name']}"
        )
    print(f"Guidance Prompt: {str(prompt)}")

    # Initialize Poisson System
    poisson = PoissonSystem(
        v, f,
        args.pipe_cfg.device,
        train_J=args.pipe_cfg.train_J,
        anchor_inds=anchor_inds,
    )
    print(f"Initialized Poisson system")

    # Initialize auxiliary learnable parameters
    quat_field = torch.zeros(
        (f.shape[0], 4),
        dtype=torch.float64,
        device=args.pipe_cfg.device,
    )
    quat_field[:, 0] = 1.0  # identity rotation
    if args.pipe_cfg.train_quat:
        quat_field.requires_grad_(True)
    # TODO: Add scaling field if necessary

    # =================================================================================
    # Optimization loop begins

    # Stage 1
    optim_vars = []
    if args.pipe_cfg.train_J:
        assert poisson.J.requires_grad, "Jacobian field must be trainable"
        optim_vars.append(poisson.J)
    assert len(optim_vars) > 0, "No trainable variables found"

    optim = torch.optim.Adam(optim_vars, lr=args.pipe_cfg.lr_stage_1)
    for i in range(args.pipe_cfg.n_kp_iter):

        curr_v, curr_f = poisson.get_current_mesh(
            anchor_pos,
            trans_mats=quat_to_mat_torch(normalize_vs(quat_field)),
        )

        loss = torch.sum(
            (curr_v[handle_inds, ...] - handle_pos) ** 2
        ) / len(handle_inds)
        
        optim.zero_grad()
        loss.backward()
        optim.step()

        wandb.log(
            {
                "train/loss_total": loss.item(),
                "train/loss_guidance": 0.0,
                "train/loss_keypoint": loss.item(),
            },
        )

        if (i + 1) % args.pipe_cfg.vis_every == 0:
            optim_img_dir = args.out_dir / "optim_imgs"
            optim_img_dir.mkdir(parents=True, exist_ok=True)

            vis_imgs = render_mesh_360(
                curr_v, curr_f, uvs, tex_inds, tex, n_step=4, radius=1.5,
            )
            vis_imgs = np.concatenate(vis_imgs, axis=1) 

            # log video
            vid_writer.append_data(vis_imgs)

            # log image
            vis_imgs = Image.fromarray(vis_imgs)
            vis_imgs.save(optim_img_dir / f"{i:04d}.png")

            wandb.log({"eval/image": wandb.Image(vis_imgs)})

    # Stage 2
    optim_vars = []
    if args.pipe_cfg.train_J:
        assert poisson.J.requires_grad, "Jacobian field must be trainable"
        optim_vars.append(poisson.J)
    if args.pipe_cfg.train_quat:
        assert quat_field.requires_grad, "Quaternion field must be trainable"
        optim_vars.append(quat_field)
    assert len(optim_vars) > 0, "No trainable variables found"

    optim = torch.optim.Adam(optim_vars, lr=args.pipe_cfg.lr_stage_2)
    for i in range(args.pipe_cfg.n_iter - args.pipe_cfg.n_kp_iter):

        # Get current mesh
        curr_v, curr_f = poisson.get_current_mesh(
            anchor_pos,
            trans_mats=quat_to_mat_torch(normalize_vs(quat_field)),
        )

        # Compute camera parameters
        if args.pipe_cfg.cam_schedule == "random":
            view_idx = int(np.random.randint(len(args.pipe_cfg.cam_locs)))
        elif args.pipe_cfg.cam_schedule == "sequential":
            view_idx = int(i % len(args.pipe_cfg.cam_locs))
        else:
            raise ValueError(f"Unknown camera schedule: {args.pipe_cfg.cam_schedule}")
        cam_loc = args.pipe_cfg.cam_locs[view_idx]
        cam2world = compute_lookat_mat(
            cam_loc.type(torch.float32).to(args.pipe_cfg.device),
            args.pipe_cfg.origin.type(torch.float32).to(args.pipe_cfg.device),
        )
        proj_mat = compute_proj_mat(
            args.pipe_cfg.aspect_ratio,
            args.pipe_cfg.fov,
            args.pipe_cfg.near,
            args.pipe_cfg.far,
            device=args.pipe_cfg.device,
        )

        # Render RGB
        bg_color = torch.ones(3, device=args.pipe_cfg.device)
        if args.pipe_cfg.use_random_bg:
            bg_color = torch.rand(3, device=args.pipe_cfg.device)
        img, img_grad = render(
            curr_v, curr_f,
            cam2world,
            proj_mat,
            args.pipe_cfg.img_height,
            args.pipe_cfg.img_width,
            uvs=uvs, tex_inds=tex_inds, tex=tex, ss_scale=4.0, bg_color=bg_color,
        )

        if (i + 1) % args.pipe_cfg.vis_every == 0:
            optim_img_dir = args.out_dir / "optim_imgs"
            optim_img_dir.mkdir(parents=True, exist_ok=True)

            vis_imgs = render_mesh_360(
                curr_v, curr_f, uvs, tex_inds, tex, n_step=4, radius=1.5,
            )
            vis_imgs = np.concatenate(vis_imgs, axis=1) 

            # log video
            vid_writer.append_data(vis_imgs)

            # log image
            vis_imgs = Image.fromarray(vis_imgs)
            vis_imgs.save(optim_img_dir / f"{i:04d}.png")
            wandb.log({"eval/image": wandb.Image(vis_imgs)})

        # =================================================================================
        # Compute losses
        l_guidance = torch.tensor(0.0, device=args.pipe_cfg.device)
        if args.pipe_cfg.w_guidance > 0.0:  # Compute guidance loss after keypoint only optimization  
            l_guidance = guidance(
                prompt,
                image=img[None].permute(0, 3, 1, 2),
                cfg_scale=args.pipe_cfg.cfg_scale,
            )

        l_kp = torch.sum(
            (curr_v[handle_inds, ...] - handle_pos) ** 2
        ) / len(handle_inds)
        
        l_quat_reg = torch.tensor(0.0, device=args.pipe_cfg.device)
        if args.pipe_cfg.w_quat_reg > 0.0:
            quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=args.pipe_cfg.device)
            l_quat_reg = torch.sum((quat_field - quat_id[None]) ** 2)

        l_total = (
            args.pipe_cfg.w_guidance * l_guidance \
            + args.pipe_cfg.w_kp * l_kp \
            + args.pipe_cfg.w_quat_reg * l_quat_reg \
        )

        # Log losses
        wandb.log(
            {
                "train/loss_total": l_total.item(),
                "train/loss_guidance": l_guidance.item(),
                "train/loss_keypoint": l_kp.item(),
                "train/loss_quat_reg": l_quat_reg.item(),
            },
        )
        # =================================================================================

        optim.zero_grad()
        l_total.backward()
        optim.step()

    # Clean up
    vid_writer.close()

    # Optimization loop ends
    # =================================================================================
        
    # Save results
    mesh_dir = args.out_dir / "mesh"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_path = mesh_dir / "deformed.obj"
    save_obj(
        mesh_path,
        curr_v.detach().cpu().numpy(),
        curr_f.detach().cpu().numpy(),
        uvs=uvs.detach().cpu().numpy(),
        tex_inds=tex_inds.detach().cpu().numpy(),
        tex=tex.detach().cpu().numpy(),
    )
    print(f"Saved mesh at: {str(mesh_path)}")

    final_imgs = render_mesh_360(
        curr_v, curr_f, uvs, tex_inds, tex, n_step=4, radius=1.5,
    )
    final_imgs = np.concatenate(final_imgs, axis=1)
    final_imgs_ = Image.fromarray(final_imgs)
    final_imgs_.save(args.out_dir / "deformed_mesh.png")

    final_marker_imgs = render_mesh_with_markers_360(
        curr_v,
        curr_f,
        torch.cat([handle_inds, anchor_inds], dim=0),
        torch.cat([handle_pos, anchor_pos], dim=0),
        n_step=4,
        radius=1.5,
    )
    final_marker_imgs = np.concatenate(final_marker_imgs, axis=1)
    final_marker_imgs_ = Image.fromarray(final_marker_imgs)
    final_marker_imgs_.save(args.out_dir / "deformed_mesh_markers.png")

    final_imgs = np.concatenate([final_imgs, final_marker_imgs], axis=0)
    final_imgs_ = Image.fromarray(final_imgs)
    final_imgs_.save(args.out_dir / "deformed_mesh_summary.png")

    wandb.log(
        {
            "eval/video": wandb.Video(str(vid_dir / "optim.mp4")),
            "eval/final_rendering": wandb.Image(final_imgs_),
        }
    )

    print(f"Done! Logs can be found at: {str(args.out_dir.resolve())}")


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
