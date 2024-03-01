"""
sd_configs.py

Config objects for experiments using Stable Diffusion guidance.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from jaxtyping import Shaped
import torch


@dataclass
class StableDiffusion3DPipeConfig:

    # Guidance parameters
    guidance_type: Literal["sd"] = "sd"
    """Type of guidance to use"""
    lora_scale: float = 1.0
    """LoRA scale used for the experiment"""
    cfg_scale: float = 100.0
    """CFG scale used for the experiment"""
    clamp_val: float = 2.0
    """Clamping value for the guidance"""

    # Poisson parameters
    train_J: bool = True
    """Whether to train source Jacobian field"""
    train_quat: bool = False
    """Whether to train quaternion field"""

    # Renderer parameters
    cam_schedule: Literal["random", "sequential"] = "random"
    """Camera scheduler"""
    cam_locs: Shaped[torch.Tensor, "* 3"] = torch.tensor(
        [
            [0.0, 0.0, 1.5],
            [1.5, 0.0, 0.0],
            [-1.5, 0.0, 0.0],
            [0.0, 0.0, -1.5],
        ]
    )
    """Camera locations"""
    origin: Shaped[torch.Tensor, "3"] = torch.tensor([0.0, 0.0, 0.0])
    """Origin of the scene"""
    img_height: int = 512
    """Height of the rendered image"""
    img_width: int = 512
    """Width of the rendered image"""
    aspect_ratio: float = 1.0
    """Aspect ratio of the rendered image"""
    fov: float = 53.14
    """Field of view of the rendered image"""
    near: float = 1e-1
    """Near plane of the rendered image"""
    far: float = 1e10
    """Far plane of the rendered image"""
    use_random_bg: bool = True
    """Whether to use random background"""

    # Loss configs
    w_guidance: float = 1.0
    """Weight for guidance loss"""
    w_kp: float = 1.0
    """Weight for keypoint matching loss"""
    w_quat_reg: float = 0.0
    """Weight for quaternion regularization loss"""

    # Experiment configs
    lr_stage_1: float = 1e-3
    """Optimizer learning rate for stage 1"""
    lr_stage_2: float = 1e-3
    """Optimizer learning rate for stage 2"""
    n_iter: int = 1300
    """Number of optimization iterations"""
    n_kp_iter: int = 300
    """Number of keypoint optimization only iterations"""
    vis_every: int = 10
    """Visualize every n iterations"""
    device = torch.device("cuda")
    """Device to use for optimization"""
    seed: int = 2024
    """Random seed"""

@dataclass
class StableDiffusion2DPipeConfig:

    # Guidance parameters
    guidance_type: Literal["sd"] = "sd"
    """Type of guidance to use"""
    lora_dir: Optional[Path] = None
    """Path to LoRA checkpoint"""
    lora_scale: float = 1.0
    """LoRA scale used for the experiment"""
    cfg_scale: float = 100.0
    """CFG scale used for the experiment"""
    clamp_val: float = 2.0
    """Clamping value for the guidance"""

    # Poisson parameters
    train_J: bool = True
    """Whether to train source Jacobian field"""
    train_quat: bool = False
    """Whether to train quaternion field"""

    # Renderer parameters
    cam_schedule: Literal["random", "sequential"] = "random"
    """Camera scheduler"""
    cam_locs: Shaped[torch.Tensor, "* 3"] = torch.tensor(
        [
            [0.0, 0.0, 1.0],
        ]
    )
    """Camera locations"""
    origin: Shaped[torch.Tensor, "3"] = torch.tensor([0.0, 0.0, 0.0])
    """Origin of the scene"""
    img_height: int = 512
    """Height of the rendered image"""
    img_width: int = 512
    """Width of the rendered image"""
    aspect_ratio: float = 1.0
    """Aspect ratio of the rendered image"""
    fov: float = 53.14
    """Field of view of the rendered image"""
    near: float = 1e-1
    """Near plane of the rendered image"""
    far: float = 1e10
    """Far plane of the rendered image"""
    use_random_bg: bool = True
    """Whether to use random background"""

    # Loss configs
    w_guidance: float = 1.0
    """Weight for guidance loss"""
    w_kp: float = 1.0
    """Weight for keypoint matching loss"""
    w_quat_reg: float = 0.0
    """Weight for quaternion regularization loss"""

    # Experiment configs
    lr_stage_1: float = 1e-3
    """Optimizer learning rate for stage 1"""
    lr_stage_2: float = 1e-3
    """Optimizer learning rate for stage 2"""
    n_iter: int = 1300
    """Number of optimization iterations"""
    n_kp_iter: int = 300
    """Number of keypoint optimization only iterations"""
    vis_every: int = 10
    """Visualize every n iterations"""
    device = torch.device("cuda")
    """Device to use for optimization"""
    seed: int = 2024
    """Random seed"""
