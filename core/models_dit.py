'''
-----------------------------------------------------------------------------
Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
-----------------------------------------------------------------------------
'''

import numpy as np
import random
import trimesh
from functools import partial
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler, DDIMScheduler
from diffusers.training_utils import compute_snr

import kiui
from core.options import Options
from core.transformer.dit import DiT


class MDiT(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()

        self.opt = opt

        self.dit = DiT(
            hidden_dim=opt.dit_hidden_dim,
            num_heads=opt.dit_num_heads,
            latent_size=opt.point_latent_size,
            latent_dim=opt.point_latent_dim,
            num_layers=opt.dit_num_layers,
            gradient_checkpointing = opt.checkpointing,
        )

        # load pretrained CLIP model and freeze 
        self.normalize_image = partial(TF.normalize, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # ref: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/preprocessor_config.json#L6
        self.image_encoder = CLIPVisionModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K').eval().half()
        self.image_encoder.requires_grad_(False)
        cond_dim = 1280
    
        # condition adaptor (out of dit)
        self.proj_cond = nn.Linear(cond_dim, opt.dit_hidden_dim)
        self.norm_cond = nn.LayerNorm(opt.dit_hidden_dim)

        # since the point encoder is cheap, we can do on-the-fly encoding!
        # NOTE: must be pretrained, use --resume or --resume2 to load from LMM checkpoint
        if opt.point_encoder_mode == 'downsample':
            from core.transformer.point import PointEncoder
        elif opt.point_encoder_mode == 'embed':
            from core.transformer.point import PointEncoderEmbed as PointEncoder
        
        self.point_encoder = PointEncoder(
            hidden_dim=opt.point_hidden_dim, 
            num_heads=opt.point_num_heads, 
            latent_size=opt.point_latent_size, 
            latent_dim=opt.point_latent_dim, 
            gradient_checkpointing=False,
        ).eval().half()
        self.point_encoder.requires_grad_(False)

        # we are not loading pretrained mesh decoder as it's too heavy... 
        
        # scheduler
        self.noise_scheduler = DDPMScheduler(
            prediction_type = opt.noise_scheduler_predtype,
            num_train_timesteps = 1000,
            beta_schedule = "scaled_linear",
            beta_start = 0.00085,
            beta_end = 0.012,
            clip_sample = False,
            thresholding = False,
            timestep_spacing = "leading",
        )
        self.scheduler = DDIMScheduler(
            prediction_type = opt.noise_scheduler_predtype,
            num_train_timesteps = 1000,
            beta_schedule = "scaled_linear",
            beta_start = 0.00085,
            beta_end = 0.012,
            clip_sample = False,
            thresholding = False,
            timestep_spacing = "leading",
            set_alpha_to_one = False,
            steps_offset = 1,
        )


    def get_cond(self, inputs):
        # inputs: images [B, 3, H, W], torch tensor in [0, 1] or text [B], str

        with torch.no_grad():
            images_clip = self.normalize_image(inputs)
            images_clip = F.interpolate(images_clip, (224, 224), mode='bilinear', align_corners=False)
            images_clip = images_clip.to(device=self.image_encoder.device)
            cond = self.image_encoder(images_clip).last_hidden_state # [B, 257, 1280]
                
        cond = self.norm_cond(self.proj_cond(cond))
            
        return cond
        
     
    # training step
    def forward(self, data, step_ratio=1):
        # data: output of the dataloader
        # return: loss

        results = {}

        inputs = data['cond'] # [B, 3, H, W], torch tensor in [0, 1] or [B] str
        points = data['points'] # surface points, [B, N, 3]
        
        B = points.shape[0]

        cond = self.get_cond(inputs)

        # random CFG dropout
        if self.training:
            mask = torch.rand((B, 1, 1), device=cond.device, dtype=cond.dtype) >= 0.1
            cond = cond * mask

        with torch.no_grad():
            
            # on-the-fly point encoding
            posterior = self.point_encoder(points)
            latents = posterior.mode()

            latents = latents.nan_to_num_(0)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=latents.device).long()

            noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # forward unet
        model_pred = self.dit(
            x=noisy_latents, # [B, N, C]
            c=cond, # [B, M, C] 
            t=timesteps, # [B,]
        )

        # loss
        if self.noise_scheduler.config.prediction_type == "epsilon": # default
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        
        if self.opt.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.opt.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
            if self.noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

        results['loss'] = loss

        return results
    
    # denoise sampling
    def run(
        self, 
        inputs, # [B, 3, H, W] torch tensor, [0, 1] or [B] str
        num_inference_steps=100,
        guidance_scale=7.5,
        num_repeat=1,
        latents=None,
        strength=0.5, # if latents is not None, add level of noise.
    ):

        device = next(self.dit.parameters()).device

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        
        cond = self.get_cond(inputs)
        batch_size = cond.shape[0]

        cond = cond.repeat_interleave(num_repeat, dim=0)

        if latents is None:
            init_step = 0
            latents = torch.randn(batch_size * num_repeat, self.opt.point_latent_size, self.opt.point_latent_dim, device=device, dtype=torch.float32)
        else:
            init_step = int(num_inference_steps * strength)
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[init_step])
        
        cond = torch.cat([torch.zeros_like(cond), cond], dim=0)

        for i, t in enumerate(self.scheduler.timesteps[init_step:]):
            latent_model_input = torch.cat([latents] * 2, dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            t_input = torch.tensor([t] * batch_size * num_repeat * 2, device=device, dtype=latents.dtype)

            noise_pred = self.dit(
                x=latent_model_input, # [B, N, C]
                c=cond, # [B, M, C]
                t=t_input, # [B,]
            )

            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        return latents # [B, N, C]