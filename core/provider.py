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


def save_mesh(tokens, opt: Options, path=None, tokenizer=None, clean=True, verbose=False):
    # tokens: [M], only for single-batch!

    # trim EOS and make divisible
    eos_idx = (tokens == opt.eos_token_id).nonzero()[0]
    if len(eos_idx) > 0:
        tokens = tokens[:eos_idx[0]]

    vertices, faces = detokenize_mesh(tokens, opt.discrete_bins, tokenizer=tokenizer)

    if verbose:
        print(f'[INFO] vertices: {vertices.shape[0]}, faces: {faces.shape[0]}')

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # fix flipped faces and merge close vertices
    if clean:
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.fix_normals()

        if verbose:
            print(f'[INFO] cleaned vertices: {mesh.vertices.shape[0]}, faces: {mesh.faces.shape[0]}')

    if path is None:
        return mesh
    else:
        mesh.export(path)


def tokenize_mesh(vertices, faces, discrete_bins, tokenizer=None):
    # vertices: [N, 3]
    # faces: [M, 3]

    # encode mesh into tokens using different tokenizers
    if tokenizer is None:
        # sort vertices
        sort_inds = np.lexsort(vertices.T) # [N]
        vertices = vertices[sort_inds]

        # xyz to zyx
        vertices = vertices[:, [2, 1, 0]] # [N, 3]

        # re-index faces
        inv_inds = np.argsort(sort_inds)
        faces = inv_inds[faces]

        # cyclically permute each face's 3 vertices, and place the lowest vertex first
        start_inds = faces.argmin(axis=1) # [M]
        all_inds = start_inds[:, None] + np.arange(3)[None, :] # [M, 3]
        faces = np.concatenate([faces, faces[:, :2]], axis=1) # [M, 5], ABCAB
        faces = np.take_along_axis(faces, all_inds, axis=1) # [M, 3]

        # sort among faces (faces.sort(0) will break each face, so we have to sort as list)
        faces = faces.tolist()
        faces.sort()
        faces = np.array(faces)

        # flatten face to vertices
        verts_per_face = vertices[faces] # [M, 3, 3]
        # discretize
        coords = ((verts_per_face + 1) * 0.5 * discrete_bins).clip(0, discrete_bins - 1).astype(np.int32)
        # tokenize
        tokens = coords.reshape(-1)
    else:
        # meto tokenizer (no need to sort)
        tokens, _, _ = tokenizer.encode(vertices, faces)
    
    # offset special tokens
    tokens = tokens + 3 # [M]

    return tokens

def detokenize_mesh(tokens, discrete_bins=None, tokenizer=None):
    # tokens: [M]

    tokens = tokens - 3

    if tokenizer is None:
        # after decoding, the tokens should be multiples of 9
        if len(tokens) % 9 != 0:
            print(f'[WARN] tokens len is {len(tokens)} % 9 != 0, trimming...')
            tokens = tokens[:-(len(tokens) % 9)]

        # all special tokens are treated as invalid triangles
        invalid_mask = tokens < 0
        # tokens = tokens[~invalid_mask] # just remove those bad tokens...
        invalid_mask = invalid_mask.reshape(-1, 9).any(axis=1)

        coords = tokens.reshape(-1, 3)

        # renormalize to [-1, 1]
        if discrete_bins is None:
            vertices = coords / coords.max() * 2 - 1
        else:
            vertices = (coords + 0.5) / discrete_bins * 2 - 1

        faces = np.arange(len(vertices)).reshape(-1, 3)
        faces = faces[~invalid_mask]

        vertices = vertices[:, [2, 1, 0]] # zyx to xyz
    else:
        # meto tokenizer
        vertices, faces, face_type = tokenizer.decode(tokens)
        # kiui.lo(vertices, faces)

    # vertices and faces still need to be deduplicated and reindexed

    return vertices, faces


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
        # self.vids = list(range(28, 40)) + list(range(0, 24))
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
                ### scale augmentation (not for image condition)
                if self.opt.use_scale_aug and self.training and self.opt.cond_mode != 'image':
                # if False:
                    bound = np.random.uniform(0.75, 0.95)
                    border_ratio = 0.2 + 0.95 - bound
                else:
                    bound = 0.95
                    border_ratio = 0.2
             
                ### none condition
                if self.opt.cond_mode == 'none':
                    cond = torch.zeros((1, 0), dtype=torch.float32) # dummy cond, we need the batch info during generation

                ### rotation augmentation
                if self.training:
                    # image cond align with rendered data
                    if self.opt.cond_mode == 'image':
                        vid = np.random.choice(self.vids, 1)[0] # 7 front-ish views
                        if vid > 27:
                            azimuth = ((vid - 27) * 30 + 90) % 360 # [0-90, 270-360]
                        else:
                            azimuth = ((vid - 0) * 15 + 90) % 360
                    # point/uncond
                    else:
                        vid = 36 # no use
                        azimuth = np.random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], 1)[0]
                else:
                    vid = 36
                    azimuth = 0

                ### load image
                if self.opt.cond_mode == 'image':
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

                    cond = torch.from_numpy(image).permute(2, 0, 1).contiguous()
                
                ### load mesh
                mesh_path = os.path.join(self.obj_path, path + '.ply')
                v, f = load_mesh(mesh_path)
                # already cleaned
                # v, f = clean_mesh(v, f, min_f=0, min_d=0, remesh=False, verbose=False)

                # face may exceed max_face_length, stats maybe inaccurate...
                if f.shape[0] > self.opt.max_face_length:
                    raise ValueError(f"{f.shape[0]} exceeds face limit.")

                # decimate mesh augmentation
                if self.opt.use_decimate_aug and self.training:
                    if f.shape[0] >= 200 and random.random() < 0.5:
                        # at most decimate to 25% of original faces.
                        target = np.random.randint(max(100, f.shape[0] // 4), f.shape[0])
                        # print(f'[INFO] decimating {f.shape[0]} to {target} faces...')
                        v, f = decimate_mesh(v, f, target=target, verbose=False)

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

                # point cloud cond
                if self.opt.cond_mode == 'point':
                    mesh = trimesh.Trimesh(vertices=v, faces=f)
                    points = mesh.sample(self.opt.point_num)
                    # perturbation as augmentation
                    if self.training and random.random() < 0.5:
                        points += np.random.randn(*points.shape) * 0.01
                    cond = torch.from_numpy(points) # [N, 3]

                coords = tokenize_mesh(v, f, self.opt.discrete_bins, self.tokenizer) # [M]

                # rare cases that relative coordinate encoding is out-of-bound
                if (coords - 3 < 0).any():
                    raise Exception(f'Invalid token range: {coords.min() - 3} - {coords.max() - 3}')

                # truncate to max length instead of dropping
                if coords.shape[0] > self.opt.max_seq_length:
                    # print(f'[WARN] {path}: coords.shape[0] > {self.opt.max_seq_length}, truncating...')
                    # coords = coords[:self.opt.max_seq_length]
                    raise ValueError(f"{coords.shape[0]} exceeds token limit.")
                    
                break

            except Exception as e:
                # print(f'[WARN] {path}: {e}') 
                # raise e # DANGEROUS, may cause infinite loop
                idx = np.random.randint(0, len(self.items))
                path = self.items[idx]

        results['cond'] = cond # [3, H, W] for image, [N, 6] for point
        results['coords'] = coords # [M]
        results['len'] = coords.shape[0] # [1]
        results['num_faces'] = f.shape[0] # [1]
        results['azimuth'] = azimuth # [1]
        results['path'] = path

        # a custom collate_fn is needed for padding and masking
        
        return results

class GithubDataset(Dataset):
    def __init__(self, opt: Options, training=True, tokenizer=None):
        
        self.opt = opt
        self.training = training

        assert opt.cond_mode != 'image', 'GithubDataset does not support image condition'

        # load items
        self.items = []
        metadata = kiui.read_json('data/github_cleaned_wface.json') # 48k
        for k, v in metadata.items():
            if v < opt.max_face_length:
                self.items.append(k)
        metadata_fbx = kiui.read_json('data/github_fbx_wface.json') # 30K
        for k, v in metadata_fbx.items():
            if v < opt.max_face_length:
                self.items.append(k)
        self.obj_path = 's3://github_ply/'
      
        # tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):

        results = {}

        path = self.items[idx]

        while True:
            try:
                # print(idx, path)

                ### scale augmentation (not for image condition)
                if self.opt.use_scale_aug and self.training and self.opt.cond_mode != 'image':
                # if False:
                    bound = np.random.uniform(0.75, 0.95)
                else:
                    bound = 0.95
             
                ### none condition
                if self.opt.cond_mode == 'none':
                    cond = torch.zeros((1, 0), dtype=torch.float32) # dummy cond, we need the batch info during generation

                ### rotation augmentation
                if self.training:
                    azimuth = np.random.choice([0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330], 1)[0]
                else:
                    azimuth = 0

                mesh_path = os.path.join(self.obj_path, path)
                v, f = load_mesh(mesh_path)

                # github data still need to be cleaned... otherwise may cause random segfault
                v, f = clean_mesh(v, f, min_f=0, min_d=0, remesh=False, verbose=False)
            
                # decimate mesh augmentation
                if self.opt.use_decimate_aug and self.training:
                    if f.shape[0] >= 200 and random.random() < 0.5:
                        # at most decimate to 10% of original faces.
                        target = np.random.randint(max(100, f.shape[0] // 4), f.shape[0])
                        # print(f'[INFO] decimating {f.shape[0]} to {target} faces...')
                        v, f = decimate_mesh(v, f, target=target, verbose=False)

                # rotate augmentation
                if azimuth != 0:
                    roty = np.stack([
                        [np.cos(np.radians(azimuth)), 0, np.sin(np.radians(azimuth))],
                        [0, 1, 0],
                        [-np.sin(np.radians(azimuth)), 0, np.cos(np.radians(azimuth))],
                    ])
                    v = v @ roty.T

                # normalize after rotation in case of oob (augment scale)
                v = normalize_mesh(v, bound=bound)

                # point cloud cond
                if self.opt.cond_mode == 'point':
                    mesh = trimesh.Trimesh(vertices=v, faces=f)
                    points = mesh.sample(self.opt.point_num)
                    # perturbation as augmentation
                    if self.training and random.random() < 0.5:
                        points += np.random.randn(*points.shape) * 0.01
                    cond = torch.from_numpy(points) # [N, 3]

                coords = tokenize_mesh(v, f, self.opt.discrete_bins, self.tokenizer) # [M]

                # rare cases that relative coordinate encoding is out-of-bound
                if (coords - 3 < 0).any():
                    raise Exception(f'Invalid token range: {coords.min() - 3} - {coords.max() - 3}')

                # truncate to max length instead of dropping
                if coords.shape[0] > self.opt.max_seq_length:
                    coords = coords[:self.opt.max_seq_length]
                    
                break

            except Exception as e:
                print(f'[WARN] {path}: {e}') 
                # raise e # DANGEROUS, may cause infinite loop
                idx = np.random.randint(0, len(self.items))
                path = self.items[idx]

        results['cond'] = cond # [3, H, W] for image, [N, 6] for point
        results['coords'] = coords # [M]
        results['len'] = coords.shape[0] # [1]
        results['num_faces'] = f.shape[0] # [1]
        results['azimuth'] = azimuth # [1]
        results['idx'] = idx
        results['path'] = path

        # a custom collate_fn is needed for padding and masking
        
        return results

class MixedDataset(Dataset):
    def __init__(self, opt: Options, training=True, tokenizer=None):
        
        self.opt = opt
        self.training = training

        assert self.training, 'MixedDataset only supports training mode'
        assert opt.cond_mode != 'image', 'MixedDataset does not support image condition'

        self.datasets = [
            ObjaverseDataset(opt, training=training, tokenizer=tokenizer),
            GithubDataset(opt, training=training, tokenizer=tokenizer),
        ]

        self.lens = [len(dataset) for dataset in self.datasets]
        # print(f'[INFO] MixedDataset: {self.lens}')

        # tokenizer
        self.tokenizer = tokenizer

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
            
        for i, dataset in enumerate(self.datasets):
            if idx < len(dataset):
                return dataset[idx]
            else:
                idx -= len(dataset)

        raise Exception('Invalid index')


def collate_fn(batch, opt: Options):

    # conds
    conds = [item['cond'] for item in batch]
    num_faces = [item['num_faces'] for item in batch]
    azimuths = [item['azimuth'] for item in batch]

    # get max len of this batch
    max_len = max([item['len'] for item in batch])
    max_len = min(max_len, opt.max_seq_length)

    # num cond tokens (may add face conds)
    num_cond_tokens = opt.num_cond_tokens
    
    # pad or truncate to max_len, and prepare masks
    tokens = []
    labels = []
    masks = []
    num_tokens = []
    for item in batch:
        
        if max_len >= item['len']:
            pad_len = max_len - item['len']

            tokens.append(np.concatenate([
                # COND tokens will be inserted here later
                np.full((1,), opt.bos_token_id), # BOS
                item['coords'], # mesh tokens
                np.full((1,), opt.eos_token_id), # EOS
                np.full((pad_len,), opt.pad_token_id), # padding
            ], axis=0)) # [1+M+1]

            labels.append(np.concatenate([
                np.full((num_cond_tokens + 1), -100), # condition & BOS don't need to be supervised
                item['coords'], # tokens to be supervised
                np.full((1,), opt.eos_token_id), # EOS to be supervised
                np.full((pad_len,), -100), # padding
            ], axis=0)) # [C+1+M+1]

            masks.append(np.concatenate([
                np.ones(num_cond_tokens + 1 + item['len'] + 1), 
                np.zeros(pad_len)
            ], axis=0)) # [C+1+M+1]

            num_tokens.append(num_cond_tokens + 1 + item['len'] + 1)
        else:
            tokens.append(np.concatenate([
                # COND tokens will be inserted here later
                np.full((1,), opt.bos_token_id), # BOS
                item['coords'][:max_len], # mesh tokens
                # no EOS as it's truncated
            ], axis=0))

            labels.append(np.concatenate([
                np.full((num_cond_tokens + 1), -100), # condition & BOS don't need to be supervised
                item['coords'][:max_len], # tokens to be supervised
                # no EOS as it's truncated
            ], axis=0))

            masks.append(np.ones(num_cond_tokens + 1 + max_len))
            num_tokens.append(num_cond_tokens + 1 + max_len)

    results = {}
    results['conds'] = torch.from_numpy(np.stack(conds, axis=0)).float()
    results['num_faces'] = torch.from_numpy(np.stack(num_faces, axis=0)).long()
    results['num_tokens'] = torch.from_numpy(np.stack(num_tokens, axis=0)).long()
    results['azimuths'] = torch.from_numpy(np.stack(azimuths, axis=0)).long()
    results['tokens'] = torch.from_numpy(np.stack(tokens, axis=0)).long()
    results['labels'] = torch.from_numpy(np.stack(labels, axis=0)).long()
    results['masks'] = torch.from_numpy(np.stack(masks, axis=0)).bool()
    results['paths'] = [item['path'] for item in batch]

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
        collate_fn=partial(collate_fn, opt=opt),
    )

    for i in range(5):
        results = next(iter(dataloader))

        kiui.lo(results['conds'], results['tokens'], results['azimuths'])

        # restore mesh
        for b in range(len(results['masks'])):
            masks = results['masks'][b].numpy()
            tokens = results['labels'][b].numpy()[masks][1+opt.num_cond_tokens:-1]

            # write obj using the original order to check face orientation
            vertices, faces = detokenize_mesh(tokens, opt.discrete_bins, tokenizer=tokenizer)
            with open(f'{i}_{b}.obj', 'w') as f:
                for v in vertices:
                    f.write(f'v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n')
                for face in faces:
                    f.write(f'f {" ".join([str(v+1) for v in face])}\n')

            # kiui.lo(tokens, faces)
            print(results['paths'][b])
            print(f'[INFO] tokens: {tokens.shape[0]}, faces: {faces.shape[0]}, ratio={100 * tokens.shape[0] / (9 * faces.shape[0]):.2f}%')

            if opt.cond_mode == 'image':
                kiui.write_image(f'{i}_{b}.png', results['conds'][b].numpy().transpose(1, 2, 0))