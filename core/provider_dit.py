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
import cv2
import json
import glob
import random
import trimesh
import numpy as np
import pandas as pd
import megfile
import tarfile

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader

import kiui
from kiui.mesh_utils import clean_mesh, decimate_mesh
from kiui.op import recenter
from core.options import Options
from core.utils import load_mesh, normalize_mesh, get_tokenizer

class ObjaverseDataset(Dataset):
    def __init__(self, opt: Options, training=True, tokenizer=None):
        
        self.opt = opt
        self.training = training

        # data list
        metadata = kiui.read_json('data_list/objaverse_wface.json')
        self.items = []
        for item in metadata:
            if item[1] < opt.max_face_length: # allow dynamic adjustment
                self.items.append(item[0])
        self.obj_path = 's3://objaverse_ply/'

        # gobj for image
        self.gobj_path = 's3://gobjaverse/'

        # load obj to gobj mapping
        gobj_to_obj = kiui.read_json('data/gobjaverse_280k_index_to_objaverse.json')
        self.obj_to_gobj = {v.replace('.glb', ''): k for k, v in gobj_to_obj.items()} # 000-xxx/bbb --> cc/dd

        if self.training:
            self.items = self.items[:-self.opt.testset_size]
        else:
            self.items = self.items[-self.opt.testset_size:]
        
        # gobj vid candidates
        self.vids = list(range(33, 40)) + list(range(12, 24))
        self.resolution = 512

        # tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        results = {}

        path = self.items[idx]

        while True:
            try:
                ### no scale augmentation for image condition
                bound = 0.95
                border_ratio = 0.2
             
                ### rotation augmentation
                if self.training:
                    # image cond align with rendered data
                    vid = np.random.choice(self.vids, 1)[0] # 7 front-ish views
                    if vid > 27:
                        azimuth = ((vid - 27) * 30 + 90) % 360 # [0-90, 270-360]
                    else:
                        azimuth = ((vid - 0) * 15 + 90) % 360
                else:
                    vid = 36
                    azimuth = 0

                ### load cond
            
                gobj_uid = self.obj_to_gobj[path]
                tar_path = os.path.join(self.gobj_path, gobj_uid + '.tar')
                uid_last = gobj_uid.split('/')[1]
                
                image_path = os.path.join(uid_last, 'campos_512_v4', f"{vid:05d}/{vid:05d}.png")

                with megfile.smart_open(tar_path, 'rb') as f:
                    with tarfile.open(fileobj=f, mode='r') as tar:
                        with tar.extractfile(image_path) as f:
                            image = np.frombuffer(f.read(), np.uint8)
            
                image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255 # [512, 512, 4] in [0, 1]
                mask = image[..., 3:4] > 0.5 # [512, 512, 1]
                # augment image (recenter)
                image = recenter(image, mask, border_ratio=border_ratio)
                image = image[..., :3] * image[..., 3:] + (1 - image[..., 3:]) # [512, 512, 3], to white bg
                image = image[..., [2,1,0]] # bgr to rgb

                image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
                cond = image
                
                ### load mesh
                mesh_path = os.path.join(self.obj_path, path + '.ply')
                v, f = load_mesh(mesh_path)
                # already cleaned
                # v, f = clean_mesh(v, f, min_f=0, min_d=0, remesh=False, verbose=False)

                # face may exceed max_face_length, stats maybe inaccurate...
                if f.shape[0] > self.opt.max_face_length:
                    raise ValueError(f"{f.shape[0]} exceeds face limit.")

                # rotate augmentation
                if azimuth != 0:
                    roty = np.stack([
                        [np.cos(np.radians(-azimuth)), 0, np.sin(np.radians(-azimuth))],
                        [0, 1, 0],
                        [-np.sin(np.radians(-azimuth)), 0, np.cos(np.radians(-azimuth))],
                    ])
                    v = v @ roty.T

                # normalize after rotation in case of oob (augment scale)
                v = normalize_mesh(v, bound=bound)

                mesh = trimesh.Trimesh(vertices=v, faces=f)
                points = mesh.sample(self.opt.point_num)
                # perturbation as augmentation
                # if self.training and random.random() < 0.5:
                #     points += np.random.randn(*points.shape) * 0.01
                points = torch.from_numpy(points).float() # [N, 3]

                break

            except Exception as e:
                # print(f'[WARN] {path}: {e}') 
                # raise e # DANGEROUS, may cause infinite loop
                idx = np.random.randint(0, len(self.items))
                path = self.items[idx]

        results['points'] = points # [N, 3]
        results['cond'] = cond # [3, 512, 512] or str
        results['azimuth'] = azimuth # [1]
        results['path'] = path

        # a custom collate_fn is needed for padding and masking
        
        return results


if __name__ == "__main__":
    import tyro
    from core.options import AllConfigs
    from functools import partial
    
    opt = tyro.cli(AllConfigs)
    kiui.seed_everything(opt.seed)

    # tokenizer
    tokenizer, _ = get_tokenizer(opt)

    dataset = ObjaverseDataset(opt, training=True, tokenizer=tokenizer)
    print(len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    for i in range(5):
        results = next(iter(dataloader))

        # kiui.lo(results['points'], results['image'], results['azimuth'])

        # restore mesh
        for b in range(len(results['points'])):
            
            # kiui.lo(tokens, faces)
            print(results['path'][b])

            kiui.write_image(f'{i}_{b}.png', results['cond'][b].numpy().transpose(1, 2, 0))
        
            pc = results['points'][b].numpy()
            pc = trimesh.PointCloud(pc)
            pc.export(f'{i}_{b}.obj')