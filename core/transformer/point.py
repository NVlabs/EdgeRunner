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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

from core.transformer.attention import SelfAttention, CrossAttention

import kiui
    
class DummyLatent:
    def __init__(self, mean):
        self.mean = mean
    
    def sample(self):
        return self.mean

    def mode(self):
        return self.mean

    def kl(self):
        # just an l2 penalty
        return 0.5 * torch.sum(torch.pow(self.mean, 2))
    
class PointEmbed(nn.Module):
    def __init__(self, dim=512, freq_embed_dim=48):
        super().__init__()

        # frequency embedding
        assert freq_embed_dim % 6 == 0
        self.freq_embed_dim = freq_embed_dim
        e = torch.pow(2, torch.arange(self.freq_embed_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.freq_embed_dim // 6), torch.zeros(self.freq_embed_dim // 6)]),
            torch.cat([torch.zeros(self.freq_embed_dim // 6), e, torch.zeros(self.freq_embed_dim // 6)]),
            torch.cat([torch.zeros(self.freq_embed_dim // 6), torch.zeros(self.freq_embed_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # [3, 48]

        self.mlp = nn.Linear(self.freq_embed_dim+3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum('bnd,de->bne', input, basis.to(input.dtype))
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.embed(input, self.basis) # B x N x C
        embed = torch.cat([embed, input], dim=2).to(input.dtype)
        embed = self.mlp(embed) # B x N x C
        return embed


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)


class ResAttBlock(nn.Module):
    def __init__(self, dim, num_heads, gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ln1 = nn.LayerNorm(dim)
        self.att = SelfAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)
    
    def forward(self, x):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, use_reentrant=False)
        else:
            return self._forward(x)
        
    def _forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ResCrossAttBlock(nn.Module):
    def __init__(self, dim, num_heads, gradient_checkpointing=True):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.ln1 = nn.LayerNorm(dim)
        self.att = CrossAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim)
    
    def forward(self, x, c):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, x, c, use_reentrant=False)
        else:
            return self._forward(x, c)
        
    def _forward(self, x, c):
        x = x + self.att(self.ln1(x), c)
        x = x + self.mlp(self.ln2(x))
        return x


class PointEncoder(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=16, latent_size=2048, latent_dim=64, gradient_checkpointing=True):
        super().__init__()

        self.latent_size = latent_size        
        # self.query_embed = nn.Parameter(torch.randn(1, latent_size, hidden_dim) / hidden_dim ** 0.5)

        self.point_embed = PointEmbed(dim=hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.cross_att = ResCrossAttBlock(hidden_dim, num_heads, gradient_checkpointing)

        self.linear = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, pc):
        # pc: [B, N, 3]
        # return: latent [B, L, D]

        B, N, C = pc.shape

        # embed
        x = self.ln(self.point_embed(pc)) # [B, N, D], condition (kv)
     
        ###### fps
        import torch_cluster
        pc_flattened = pc.view(B*N, C)
        batch_indices = torch.arange(B, device=pc.device).repeat_interleave(N)
        fps_indices = torch_cluster.fps(pc_flattened, batch_indices, ratio=self.latent_size / N)
        query_pc = pc_flattened[fps_indices].view(B, self.latent_size, C)
        q = self.point_embed(query_pc)
        ######

        # att
        l = self.cross_att(q, x)

        # out
        l = self.linear(l) # [B, L, D]

        posterior = DummyLatent(l)

        return posterior
    

class PointEncoderEmbed(nn.Module):
    def __init__(self, hidden_dim=1024, num_heads=16, latent_size=2048, latent_dim=64, gradient_checkpointing=True):
        super().__init__()

        self.latent_size = latent_size        
        self.query_embed = nn.Parameter(torch.randn(1, latent_size, hidden_dim) / hidden_dim ** 0.5)

        self.point_embed = PointEmbed(dim=hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)

        self.cross_att = ResCrossAttBlock(hidden_dim, num_heads, gradient_checkpointing)

        self.linear = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, x):
        # x: [B, N, 3]
        # return: latent [B, L, D]

        B, N, C = x.shape

        # embed
        x = self.ln(self.point_embed(x)) # [B, N, D], condition (kv)

        # downsample x to q
        q = self.query_embed.repeat(B, 1, 1) # query
      
        # att
        l = self.cross_att(q, x)

        # out
        l = self.linear(l) # [B, L, D]

        posterior = DummyLatent(l)

        return posterior