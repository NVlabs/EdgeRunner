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

from transformers import CLIPVisionModel, CLIPImageProcessor

import kiui
from core.options import Options
from core.provider import save_mesh
from core.utils import quantize_num_faces

# large mesh model
class LMM(nn.Module):
    def __init__(self, opt: Options):
        super().__init__()

        self.opt = opt

        ### conditioner
        if opt.cond_mode == 'image':
            self.normalize_image = partial(TF.normalize, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)) # ref: https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K/blob/main/preprocessor_config.json#L6
            self.image_encoder = CLIPVisionModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K')
            if opt.freeze_encoder:
                self.image_encoder = self.image_encoder.eval().half()
                self.image_encoder.requires_grad_(False)
            self.proj_cond = nn.Linear(1280, opt.hidden_dim)
            self.norm_cond = nn.LayerNorm(opt.hidden_dim)

        elif opt.cond_mode == 'point':
            if opt.point_encoder_mode == 'downsample':
                from core.transformer.point import PointEncoder
            elif opt.point_encoder_mode == 'embed':
                from core.transformer.point import PointEncoderEmbed as PointEncoder
            
            assert not opt.freeze_encoder

            self.point_encoder = PointEncoder(
                hidden_dim=opt.point_hidden_dim, 
                num_heads=opt.point_num_heads, 
                latent_size=opt.point_latent_size, 
                latent_dim=opt.point_latent_dim, 
                gradient_checkpointing=opt.checkpointing,
            )

            self.proj_cond = nn.Linear(opt.point_latent_dim, opt.hidden_dim)
            self.norm_cond = nn.LayerNorm(opt.hidden_dim)
        
        elif opt.cond_mode == 'point_latent':
            # in such case, the latent is produced by a separate diffusion model
            self.proj_cond = nn.Linear(opt.point_latent_dim, opt.hidden_dim)
            self.norm_cond = nn.LayerNorm(opt.hidden_dim)

        if opt.use_num_face_cond:
            self.embed_num_face = nn.Embedding(10, opt.hidden_dim)

        ### mesh decoder

        # vocab size (hard-coded for each case)
        if opt.use_meto:
            if opt.meto_backend == 'LR':
                self.vocab_size = 2 * opt.discrete_bins + 3 + 3
            elif opt.meto_backend == 'LR_ABSCO':
                self.vocab_size = opt.discrete_bins + 3 + 3
        else:
            self.vocab_size = opt.discrete_bins + 3

        from core.transformer.modeling_opt import ShapeOPTConfig, ShapeOPT
        self.config = ShapeOPTConfig(
            vocab_size=self.vocab_size,
            hidden_dim=opt.hidden_dim,
            intermediate_dim=opt.hidden_dim * 4 if opt.intermediate_dim is None else opt.intermediate_dim,
            num_hidden_layers=opt.num_layers,
            num_attention_heads=opt.num_heads,
            max_position_embeddings=opt.max_seq_length + opt.num_cond_tokens + 10, # pos embedding size
            num_cond_tokens=opt.num_cond_tokens,
        )
        self.mesh_decoder = ShapeOPT(self.config)

        if opt.checkpointing:
            self.mesh_decoder.model.gradient_checkpointing_enable()
        
    def encode_cond(self, conds, num_faces):

        results = {}

        grad_ctx = torch.no_grad if self.opt.freeze_encoder else nullcontext
        if self.opt.cond_mode == 'image':
            with grad_ctx():
                images_clip = self.normalize_image(conds)
                images_clip = F.interpolate(images_clip, (224, 224), mode='bilinear', align_corners=False)
                images_clip = images_clip.to(device=self.image_encoder.device)
                cond_embeds = self.image_encoder(images_clip).last_hidden_state # [B, 257, 1280]
            cond_embeds = self.norm_cond(self.proj_cond(cond_embeds))

        elif self.opt.cond_mode == 'point':
            with grad_ctx():
                posterior = self.point_encoder(conds)
                results['posterior'] = posterior
            if self.training:
                cond_embeds = posterior.sample() # [B, 2048, 64]
            else:
                cond_embeds = posterior.mode()
            if not self.training:
                kiui.lo(cond_embeds, verbose=True)
            cond_embeds = self.norm_cond(self.proj_cond(cond_embeds))
        
        elif self.opt.cond_mode == 'point_latent':
            # no encoder (preprocessed)
            cond_embeds = self.proj_cond(conds)
            cond_embeds = self.norm_cond(cond_embeds)
        
        elif self.opt.cond_mode == 'none': # will ignore conds
            cond_embeds = None

        # encode num_faces
        if self.opt.use_num_face_cond:
            num_faces = quantize_num_faces(num_faces)
            num_face_embeds = self.embed_num_face(num_faces).unsqueeze(1) # [B, 1, C]
            if cond_embeds is not None:
                cond_embeds = torch.cat((cond_embeds, num_face_embeds), dim=1)
            else:
                cond_embeds = num_face_embeds

        results['cond_embeds'] = cond_embeds
        return results


    def forward(self, data, step_ratio=1):

        results = {}

        conds = data['conds'] # image [B, 3, H, W] or point [B, N, 3] or None
        tokens = data['tokens'] # tokens [B, 1+M+1], long
        labels = data['labels'] # labels [B, C+1+M+1], long
        masks = data['masks'] # attn masks [B, C+1+M+1], bool
        num_faces = data['num_faces'] # num_faces [B], long
        num_tokens = data['num_tokens'] # num_tokens [B], long

        B = labels.shape[0]

        # random num_faces dropout
        unprog_mask = None
        if self.training and self.opt.use_num_face_cond:
            unprog_mask = torch.rand((B,), device=conds.device) < self.opt.nof_dropout_ratio
            num_faces[unprog_mask] = -1
        
        # encode conds
        results_cond = self.encode_cond(conds, num_faces) # [B, N, C]
        cond_embeds = results_cond['cond_embeds']

        # encode tokens
        token_embeds = self.mesh_decoder.model.embd(tokens)

        # insert cond embeds
        if cond_embeds is not None:
            inputs_embeds = torch.cat((cond_embeds, token_embeds), dim=1)
        else:
            inputs_embeds = token_embeds
        
        # call decoder
        kwargs = {
            'inputs_embeds': inputs_embeds,
            'labels': labels,
            'attention_mask': masks,
            'num_tokens': num_tokens,
        }

        outputs = self.mesh_decoder(**kwargs)

        results['loss_ce'] = outputs.loss
        loss = outputs.loss

        # optional kl loss
        if 'posterior' in results_cond:
            posterior = results_cond['posterior']
            kl_loss = posterior.kl().mean()
            results['loss_kl'] = kl_loss
            loss = loss + self.opt.kl_weight * kl_loss
       
        results['loss'] = loss
        results['logits'] = outputs.logits # [B, 1+C+M+1, V]

        return results

    @torch.no_grad()
    def generate(
            self,
            conds,
            num_faces=1000,
            resume_ids=None,
            tokenizer=None,
            max_new_tokens=None,
            clean=True,
        ):
            
            B = conds.shape[0]
            assert B == 1, 'Batch size must be 1 for generation.'

            # encode input_embeds (only COND)
            cond_num_faces = torch.full((B,), num_faces, dtype=torch.long, device=conds.device)
            results_cond = self.encode_cond(conds, cond_num_faces) # [B, N, C]
            cond_embeds = results_cond['cond_embeds']

            # BOS input_ids to start generation
            input_ids = torch.full((B, 1), self.opt.bos_token_id, dtype=torch.long, device=conds.device) # BOS token
            if resume_ids is not None:
                input_ids = torch.cat((input_ids, resume_ids), dim=1)

            tokens_embeds = self.mesh_decoder.model.embd(input_ids) # [B, 1, C]

            if cond_embeds is not None:
                inputs_embeds = torch.cat((cond_embeds, tokens_embeds), dim=1)
            else:
                inputs_embeds = tokens_embeds

            # constraint function
            if tokenizer is None:
                def prefix_allowed_tokens_fn(batch_id, input_ids):
                    candidates = list(range(3, self.vocab_size))
                    # BOS is already provided as the first input token 
                    if input_ids.shape[0] % 9 == 1:
                        candidates.append(self.opt.eos_token_id) # EOS is only allowed at 1 + 9 * N position (each triangle must be complete)
                    return candidates
            else:
                # special rules for meto sequence
                if self.opt.meto_backend in ['LR', 'LR_ABSCO']: 
                    def prefix_allowed_tokens_fn_with_state(batch_id, input_ids, state: dict):
                        idx = input_ids.shape[0]
                        # print(f'=== prefix idx: {idx} ===')

                        # BOS is always provided, so the first token must be BOM
                        # 0=PAD, 1=BOS, 2=EOS, 3=L, 4=R, 5=BOM, 6~=coords
                        if idx == 0: return [5]

                        # update state based on the last token
                        if input_ids[-1] == 5:
                            state['counter'] = 9 # after BOM, there must be 9 coords tokens
                        elif input_ids[-1] in [3, 4]:
                            state['counter'] = 3 # after LR, there must be 3 coords tokens
                        elif input_ids[-1] >= 6:
                            state['counter'] -= 1 # after coords, counter -1
                       
                        # set rules for the next token
                        # counter > 0 means there are still coords to be filled
                        if state['counter'] > 0:
                            return list(range(6, self.vocab_size))
                        # otherwise, it could be L/R/BOM/EOS
                        else:
                            return [3, 4, 5, self.opt.eos_token_id]
                    # keep a persistent state during generation
                    state = { 'counter': 0 }
                    prefix_allowed_tokens_fn = partial(prefix_allowed_tokens_fn_with_state, state=state)

                else:
                    print('[WARN] prefix_allowed_tokens_fn is not defined for meto backend:', self.opt.meto_backend)
                    prefix_allowed_tokens_fn = None

            # call generate
            max_new_tokens = self.opt.max_seq_length if max_new_tokens is None else max_new_tokens

            # un-face-num-conditioned generation
            if num_faces < 0:
                num_tokens = torch.full((B,), -1, dtype=torch.long, device=conds.device)
            else:
                num_tokens = torch.full((B,), num_faces * 4 + self.opt.num_cond_tokens, dtype=torch.long, device=conds.device)

            kwargs = {
                # 'input_ids': input_ids,
                'inputs_embeds': inputs_embeds,
                'num_tokens': num_tokens,
                'pad_token_id': self.opt.pad_token_id,
                'bos_token_id': self.opt.bos_token_id,
                'eos_token_id': self.opt.eos_token_id,
                'max_new_tokens': max_new_tokens,
                'prefix_allowed_tokens_fn': prefix_allowed_tokens_fn, # after converging we don't actually need this.
            }

            if self.opt.generate_mode == 'greedy':
                kwargs['num_beams'] = 1
            elif self.opt.generate_mode == 'sample':
                kwargs['do_sample'] = True
                kwargs['top_k'] = 10

            output_ids = self.mesh_decoder.generate(**kwargs) # [B, 1+C+M+1]

            # batch detokenize to meshes
            meshes = []
            all_tokens = []
            for b in range(B):
                tokens = output_ids[b].detach().cpu().numpy()
                if resume_ids is not None:
                    tokens = np.concatenate((resume_ids[b].detach().cpu().numpy(), tokens), axis=0)
                kiui.lo(tokens) # not including the COND and BOS tokens
                # print(tokens[:13])
                # print(tokens[-13:])
                mesh = save_mesh(tokens, self.opt, tokenizer=tokenizer, clean=clean, verbose=True) # discard BOS
                meshes.append(mesh)
                all_tokens.append(tokens)

            return meshes, all_tokens