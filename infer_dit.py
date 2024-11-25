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

import os
import tyro
import glob
import time
import numpy as np
import rembg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file

import kiui
from kiui.op import recenter

from core.options import AllConfigs, Options
from core.models import LMM
from core.models_dit import MDiT
from core.utils import get_tokenizer
from core.utils import monkey_patch_transformers

monkey_patch_transformers()

opt = tyro.cli(AllConfigs)

kiui.seed_everything(opt.seed)

# model
model = LMM(opt)

# resume pretrained checkpoint
if opt.resume is not None:
    if opt.resume.endswith('safetensors'):
        ckpt = load_file(opt.resume, device='cpu')
    else:
        ckpt = torch.load(opt.resume, map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume}')
else:
    print(f'[WARN] model randomly initialized, are you sane?')

# fix options
opt.cond_mode = 'point_latent'

# dit model
model_dit = MDiT(opt)

# resume pretrained checkpoint
if opt.resume2 is not None:
    if opt.resume2.endswith('safetensors'):
        ckpt = load_file(opt.resume2, device='cpu')
    else:
        ckpt = torch.load(opt.resume2, map_location='cpu')
    model_dit.load_state_dict(ckpt, strict=False)
    print(f'[INFO] Loaded checkpoint from {opt.resume2}')

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.half().eval().to(device)
model_dit = model_dit.half().eval().to(device)

# load rembg
bg_remover = rembg.new_session()

# tokenizer
tokenizer, _ = get_tokenizer(opt)


# process function
def process(opt: Options, path):
    name = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(opt.workspace, exist_ok=True)

    input_image = kiui.read_image(path, mode='uint8', order='RGBA')

    # bg removal if there is no alpha channel
    if input_image.shape[-1] == 3:
        input_image = rembg.remove(input_image, session=bg_remover) # [H, W, 4]

    mask = input_image[..., -1] > 0
    image = recenter(input_image, mask, border_ratio=0.2)
    image = image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
    kiui.write_image(os.path.join(opt.workspace, name + '.jpg'), image)

    image = torch.from_numpy(image).permute(2, 0, 1).contiguous().unsqueeze(0).float().to(device)
    cond = F.interpolate(image, (512, 512), mode='bilinear', align_corners=False) # match training data and DINO.

    cond_points = None

    for i in range(opt.test_repeat):

        for num_faces in opt.test_num_face:
            t0 = time.time()

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    # first run dit
                    latents = model_dit.run(cond)                    
                    # then run lmm
                    meshes, tokens = model.generate(latents, num_faces=num_faces, max_new_tokens=opt.test_max_seq_length, tokenizer=tokenizer, clean=True)
            
            # single batch
            mesh = meshes[0]
            tokens = tokens[0]
            
            # post process tokens
            eos_idx = np.nonzero(tokens == 2)[0]
            if len(eos_idx) > 0:
                tokens = tokens[:eos_idx[0]]
            tokens -= 3

            # write output
            filename = f'{name}_{i}'
            if opt.use_num_face_cond:
                filename += f'_{num_faces}f'
            mesh.export(f'{opt.workspace}/{filename}.obj')
            np.save(f'{opt.workspace}/{filename}_tokens.npy', tokens)

            # timing
            torch.cuda.synchronize()
            t1 = time.time()
            print(f'[INFO] Processing {path} --> {filename}.obj, time = {t1 - t0:.4f}s')
    

assert opt.test_path is not None
if os.path.isdir(opt.test_path):
    file_paths = glob.glob(os.path.join(opt.test_path, "*"))
else:
    file_paths = [opt.test_path]
for path in file_paths:
    process(opt, path)
