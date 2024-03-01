"""
stable_diffusion.py

A wrapper around Stable Diffusion for computing SDS loss and its variants.

Inspired by an implementation from threestudio:
https://github.com/threestudio-project/threestudio/blob/main/threestudio/models/guidance/stable_diffusion_guidance.py
"""

from pathlib import Path
from typing import List, Literal, Optional, Union

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
)
from jaxtyping import Float, Int64, jaxtyped
import torch
import torch.nn.functional as F
from typeguard import typechecked

from .base_guidance import Guidance


class StableDiffusionGuidance(Guidance):
    
    model_type: Literal["stabilityai/stable-diffusion-2-1-base"]
    """Type of the Stable Diffusion model to use."""
    min_step_t: float
    """Minimum diffusion timestep."""
    max_step_t: float
    """Maximum diffusion timestep."""
    use_half_precision: bool
    """Whether to use half-precision floating point."""
    device: torch.device
    """Device to use for computation."""
    lora_path: Optional[Union[str, Path]]
    """Path to the pre-trained LoRA weights."""
    lora_scale: float
    """Scaling factor for the LoRA weights."""
    grad_clamp_val: Optional[float]
    """Value to clamp the gradients to."""

    
    @jaxtyped(typechecker=typechecked)
    def __init__(
        self,
        model_type: Literal[
            "stabilityai/stable-diffusion-2-1-base",
        ] = "stabilityai/stable-diffusion-2-1-base",
        min_step_t: float = 0.02,
        max_step_t: float = 0.98,
        use_half_precision: bool = False,
        device: torch.device = torch.device("cuda"),
        lora_path: Optional[Union[str, Path]] = None,
        lora_scale: float = 1.0,
        grad_clamp_val: Optional[float] = None,
    ) -> None:
        """Constructor"""
        super().__init__()

        # register variables
        self.model_type = model_type
        self.min_step_t = min_step_t
        self.max_step_t = max_step_t
        self.use_half_precision = use_half_precision
        self.device = device
        self.lora_path = lora_path
        self.lora_scale = lora_scale
        self.grad_clamp_val = grad_clamp_val

        # load Stable Diffusion
        self._build_model()

    @jaxtyped(typechecker=typechecked)
    def __call__(
        self,
        prompt: str,
        image: Float[torch.Tensor, "1 num_channel height width"] = None,
        latent: Optional[Float[torch.Tensor, "1 4 64 64"]] = None,
        cfg_scale: float = 7.5,
        step: Optional[Int64[torch.Tensor, "1"]] = None,
        grad_type: Literal["sds", "dds", "nfsd"] = "sds",
    ):
        # encode images
        if latent is None:
            assert image is not None, "Either image or latent must be specified."
            image = F.interpolate(
                image,
                (512, 512),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            latent = self.encode_images(image)
        batch_size = latent.shape[0]

        # sample diffusion timestep ~ U(t_min, t_max)
        if step is None:
            step = torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch_size,),
                dtype=torch.long,
                device=self.device,
            )
        assert step.min() >= self.min_step and step.max() <= self.max_step, (
            f"step must be in [{self.min_step}, {self.max_step}]: {step.min()}, {step.max()}"
        )

        # predict gradient
        prompt = [prompt]
        grad = self.compute_img_grad(latent, prompt, cfg_scale, step, grad_type)
        grad = torch.nan_to_num(grad)
        if not self.grad_clamp_val is None:
            grad = torch.clamp(
                grad,
                -self.grad_clamp_val,
                self.grad_clamp_val,
            )
        
        # compute loss
        target = (latent - grad).detach()
        loss = (0.5 / batch_size) * F.mse_loss(
            latent,
            target,
            reduction="mean",
        )

        return loss

    @jaxtyped(typechecker=typechecked)
    def compute_img_grad(
        self,
        latent: Float[torch.Tensor, "1 4 64 64"],
        prompt: List[str],
        cfg_scale: float,
        step: Int64[torch.Tensor, "1"],
        grad_type: Literal["sds", "dds", "nfsd"] = "sds",
    ) -> Float[torch.Tensor, "1 4 64 64"]:
        """
        Computes image-space gradient for the given latent and prompt.
        """
        if grad_type == "sds":
            return self._compute_img_grad_sds(
                latent, prompt, cfg_scale, step
            )
        elif grad_type == "nfsd":
            return self._compute_img_grad_nfsd(
                latent, prompt, cfg_scale, step
            )            
        else:
            raise NotImplementedError()

    @jaxtyped(typechecker=typechecked)
    def _compute_img_grad_sds(
        self,
        latent: Float[torch.Tensor, "1 4 64 64"],
        prompt: List[str],
        cfg_scale: float,
        step: Int64[torch.Tensor, "1"],
    ) -> Float[torch.Tensor, "1 4 64 64"]:
        """
        Computes Score Distillation Sampling gradient.
        """
        # encode prompts
        batch_size = latent.shape[0]
        assert len(prompt) == batch_size, f"{len(prompt)} != {batch_size}"
        text_embeddings = self.encode_prompts(prompt)

        # predict noise
        with torch.no_grad():
            noise = torch.randn_like(latent)
            latents_noisy = self.scheduler.add_noise(latent, noise, step)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            eps = self.forward_unet(
                latent_model_input,
                torch.cat([step] * 2),
                encoder_hidden_states=text_embeddings,
            )

        # compute the Classifier-Free Guidance
        # NOTE: the order of unconditional and conditional
        # noise prediction is often reversed in other codebases
        # https://github.com/threestudio-project/threestudio/blob/5e29759db7762ec86f503f97fe1f71a9153ce5d9/threestudio/models/guidance/stable_diffusion_guidance.py#L248
        eps_uncond, eps_text = eps.chunk(2)
        eps = eps_text + cfg_scale * (eps_text - eps_uncond)

        # compute SDS gradient
        weight = (1 - self.alphas[step]).view(-1, 1, 1, 1)
        grad = weight * (eps - noise)

        return grad

    @jaxtyped(typechecker=typechecked)
    def _compute_img_grad_nfsd(
        self,
        latent: Float[torch.Tensor, "1 4 64 64"],
        prompt: List[str],
        cfg_scale: float,
        step: Int64[torch.Tensor, "1"],
    ):
        # encode prompts
        batch_size = latent.shape[0]
        assert len(prompt) == batch_size, f"{len(prompt)} != {batch_size}"
        text_embeddings = self.encode_prompts(prompt)

        # ===============================================================
        # predict delta_c, delta_d
        with torch.no_grad():
            noise = torch.randn_like(latent)
            latents_noisy = self.scheduler.add_noise(latent, noise, step)
            latent_model_input = torch.cat([latents_noisy] * 2, dim=0)
            eps_c = self.forward_unet(
                latent_model_input,
                torch.cat([step] * 2),
                encoder_hidden_states=text_embeddings,
            )

        # compute the Classifier-Free Guidance
        # NOTE: the order of unconditional and conditional
        # noise prediction is often reversed in other codebases
        # https://github.com/threestudio-project/threestudio/blob/5e29759db7762ec86f503f97fe1f71a9153ce5d9/threestudio/models/guidance/stable_diffusion_guidance.py#L248
        eps_uncond, eps_text = eps_c.chunk(2)
        delta_c = eps_text - eps_uncond

        delta_d = eps_uncond
        if step.item() >= 200:
            with torch.no_grad():
                neg_prompt = ["unrealistic, blurry, low quality, out of focus, ugly, low contrast, dull, dark, low-resolution, gloomy"]
                neg_prompt_emb = self.encode_prompts(neg_prompt)[1:2]
                eps_d = self.forward_unet(
                    latents_noisy,
                    step,
                    encoder_hidden_states=neg_prompt_emb,
                )
            delta_d -= eps_d
        # ===============================================================

        # compute NFSD gradient
        weight = (1 - self.alphas[step]).view(-1, 1, 1, 1)
        grad = weight * (delta_d + cfg_scale * delta_c)

        return grad

    @torch.no_grad()
    @jaxtyped(typechecker=typechecked)
    def edit_img(
        self,
        prompt: str,
        img_init: Float[torch.Tensor, "1 num_channel height width"],
        cfg_scale: float = 7.5,
        start_t: int = 500,
        num_inf_step: int = 20,
        method: Literal["sdedit"] = "sdedit",
    ):
        assert start_t >= 0 and start_t <= self.num_train_timesteps, (
            f"start_t must be in [0, {self.num_train_timesteps}]: {start_t}"
        )

        # encode images
        assert img_init is not None, "Either image or latent must be specified."
        img_init = F.interpolate(
            img_init,
            (512, 512),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        latent_init = self.encode_images(img_init)
        batch_size = latent_init.shape[0]

        # encode prompts
        prompt = [prompt]
        assert len(prompt) == batch_size, f"{len(prompt)} != {batch_size}"
        txt_embeds = self.encode_prompts(prompt)

        # set scheduler timesteps
        self.scheduler.set_timesteps(num_inf_step, device=self.device)
        print(f"Using {num_inf_step} inference steps")
        # print(self.scheduler.timesteps)
        ts = self.scheduler.timesteps
        step_ratio = self.scheduler.num_train_timesteps // num_inf_step
        start_t_idx = len(ts) - start_t // step_ratio - 1
        assert start_t_idx >= 0 and start_t_idx < len(ts), (
            f"start_t_idx must be in [0, {len(ts)}): {start_t_idx}"
        )

        # add noise to image
        eps = torch.randn_like(latent_init)
        x_t = self.scheduler.add_noise(
            latent_init,
            eps,
            ts[start_t_idx],
        )

        # run reverse process        
        for t in ts[start_t_idx:]:
            eps = torch.zeros_like(latent_init)
            if t > 0:
                eps = torch.randn_like(latent_init)

            # predict noise
            eps_theta = self.forward_unet(
                torch.cat([x_t] * 2, dim=0),
                torch.tensor([t, t], dtype=torch.long, device=self.device),
                encoder_hidden_states=txt_embeds,
            )
            eps_uncond, eps_text = eps_theta.chunk(2)
            eps_theta = eps_text + cfg_scale * (eps_text - eps_uncond)

            # denoise
            x_t = self.scheduler.step(eps_theta, t, x_t, return_dict=False)[0]

        # decode latents
        img_out = self.decode_latents(x_t)

        return img_out

    @jaxtyped(typechecker=typechecked)
    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self,
        image: Float[torch.Tensor, "1 3 512 512"]
    ) -> Float[torch.Tensor, "1 4 64 64"]:
        """
        Encodes RGB images into latents using pre-trained VAE.
        """
        input_dtype = image.dtype
        image = image * 2.0 - 1.0
        posterior = self.vae.encode(image.to(self.weight_dtype)).latent_dist
        latent = posterior.sample() * self.vae.config.scaling_factor
        return latent.to(input_dtype)
    
    @jaxtyped(typechecker=typechecked)
    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self,
        latent: Float[torch.Tensor, "1 4 64 64"],
    ) -> Float[torch.Tensor, "1 3 512 512"]:
        """
        Decodes latents into RGB images using pre-trained VAE.
        """
        input_dtype = latent.dtype
        latent = 1 / self.vae.config.scaling_factor * latent
        image = self.vae.decode(latent.to(self.weight_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    @jaxtyped(typechecker=typechecked)
    @torch.no_grad()
    @torch.cuda.amp.autocast(enabled=False)
    def encode_prompts(
        self,
        prompt: List[str],
    ) -> Float[torch.Tensor, "* sequence_len embedding_dim"]:
        """
        Encodes the input text prompt.
        """
        lora_scale = self.lora_scale
        if self.lora_path is None:
            lora_scale = None
        text_embedding = self.pipeline._encode_prompt(
            prompt,
            self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=[""] * len(prompt),
            lora_scale=lora_scale,
        )

        return text_embedding

    @jaxtyped(typechecker=typechecked)
    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latent: Float[torch.Tensor, "* 4 64 64"],
        step: Int64[torch.Tensor, "*"],
        encoder_hidden_states: Float[torch.Tensor, "* sequence_len embedding_dim"],
    ) -> Float[torch.Tensor, "* 4 64 64"]:
        """
        Forward pass of Stable Diffusion U-Net.
        """
        input_dtype = latent.dtype
        cross_attention_kwargs = None
        if self.lora_path is not None:
            cross_attention_kwargs = {
                "scale": self.lora_scale,
            }
        unet_output = self.unet(
            latent.to(self.weight_dtype),
            step.to(self.weight_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weight_dtype),
            cross_attention_kwargs=cross_attention_kwargs,
        ).sample.to(input_dtype)

        return unet_output

    @jaxtyped(typechecker=typechecked)
    def _build_model(self) -> None:
        
        # set precision
        self.weight_dtype = torch.float32
        if self.use_half_precision:
            self.weight_dtype = torch.float16

        # load Stable Diffusion
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_type,
            torch_dtype=self.weight_dtype,
            resume_download=True,
        ).to(self.device)

        # configure scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            self.model_type,
            subfolder="scheduler",
            torch_dtype=self.weight_dtype,
        )

        # configure timestep scheduling
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.min_step_t * self.num_train_timesteps)
        self.max_step = int(self.max_step_t * self.num_train_timesteps)
        self.alphas: Float[torch.Tensor, "num_train_timesteps"] = (
            self.scheduler.alphas_cumprod.to(self.device)
        )

        # load pre-trained LoRA weights if specified
        if self.lora_path is not None:
            assert self.lora_scale >= 0.0 and self.lora_scale <= 1.0, (
                f"LoRA scale must be in [0, 1]: {self.lora_scale}"
            )
            self._load_lora_weights()

        # extract modules from Stable Diffusion pipeline
        self.vae = self.pipeline.vae.eval()
        self.unet = self.pipeline.unet.eval()

        # freeze the modules
        for param in self.vae.parameters():
            param.requires_grad_(False)
        for param in self.unet.parameters():
            param.requires_grad_(False)

        print("="*30)
        print("[!] Built Stable Diffusion Guidance")
        print("="*30)

    @jaxtyped(typechecker=typechecked)
    def _load_lora_weights(self) -> None:
        if isinstance(self.lora_path, str):
            self.lora_path = Path(self.lora_path)
        assert self.lora_path.exists(), (
            f"LoRA checkpoint does not exist: {str(self.lora_path)}"
        )
        assert self.pipeline is not None, (
            "Pipeline must be initialized before loading LoRA weights"
        )
        assert isinstance(self.pipeline, StableDiffusionPipeline), (
            "Pipeline must be an instance of StableDiffusionPipeline"
        )

        # replace the scheduler
        self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipeline.scheduler.config,
            use_karras_sigmas=True,
        )

        # load and attach LoRA to the pipeline
        self.pipeline.load_lora_weights(str(self.lora_path))
        self.pipeline = self.pipeline.to(self.device)

        # override the scheduler config variables
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.min_step_t * self.num_train_timesteps)
        self.max_step = int(self.max_step_t * self.num_train_timesteps)
        self.alphas: Float[torch.Tensor, "num_train_timesteps"] = (
            self.scheduler.alphas_cumprod.to(self.device)
        )
        
        print("="*30)
        print("[!] Loaded LoRA weights")
        print("="*30)


if __name__ == "__main__":
    guidance = StableDiffusionGuidance()