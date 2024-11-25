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
import pickle
import trimesh
from typing import Literal

# cpp extension (named in setup.py and PYBIND11_MODULE)
import _meto

class Engine:
    def __init__(self, discrete_bins, verbose=False, backend: Literal['CLERS', 'LR', 'LR_ABSCO'] = 'LR_ABSCO'):
        self.discrete_bins = discrete_bins
        self.verbose = verbose
        # the cpp impl
        if backend == 'CLERS':
            self.impl = _meto.Engine_CLERS(discrete_bins, verbose)
            self.num_base_tokens = discrete_bins * 2
            self.num_special_tokens = 7
        elif backend == 'LR':
            self.impl = _meto.Engine_LR(discrete_bins, verbose)
            self.num_base_tokens = discrete_bins * 2
            self.num_special_tokens = 3
        elif backend == 'LR_ABSCO':
            self.impl = _meto.Engine_LR_ABSCO(discrete_bins, verbose)
            self.num_base_tokens = discrete_bins
            self.num_special_tokens = 3

        self.num_tokens = self.num_base_tokens + self.num_special_tokens

    def encode(self, vertices, faces):
        # vertices: [N, 3], float
        # faces: [M, 3], int
        tokens, face_order, face_type = self.impl.encode(vertices, faces)
        return np.asarray(tokens), np.asarray(face_order), np.asarray(face_type)

    def decode(self, tokens):
        # tokens: [N], int
        vertices, faces, face_type = self.impl.decode(tokens)
        return np.asarray(vertices), np.asarray(faces), np.asarray(face_type)
    

# helper functions
def normalize_mesh(vertices, bound=0.95):
    vmin = vertices.min(0)
    vmax = vertices.max(0)
    ori_center = (vmax + vmin) / 2
    ori_scale = 2 * bound / np.max(vmax - vmin)
    vertices = (vertices - ori_center) * ori_scale
    return vertices


def load_mesh(path, bound=0.95, clean=True):
    # use trimesh to load glb
    _data = trimesh.load(path)
    # always convert scene to mesh, and apply all transforms...
    if isinstance(_data, trimesh.Scene):
        # print(f"[INFO] load trimesh: concatenating {len(_data.geometry)} meshes.")
        _concat = []
        # loop the scene graph and apply transform to each mesh
        scene_graph = _data.graph.to_flattened() # dict {name: {transform: 4x4 mat, geometry: str}}
        for k, v in scene_graph.items():
            name = v['geometry']
            if name in _data.geometry and isinstance(_data.geometry[name], trimesh.Trimesh):
                transform = v['transform']
                _concat.append(_data.geometry[name].apply_transform(transform))
        _mesh = trimesh.util.concatenate(_concat)
    else:
        _mesh = _data
    
    vertices = _mesh.vertices
    faces = _mesh.faces

    # normalize
    vertices = normalize_mesh(vertices, bound=bound)

    # clean
    if clean:
        from kiui.mesh_utils import clean_mesh
        # only merge close vertices
        vertices, faces = clean_mesh(vertices, faces, v_pct=1, min_f=0, min_d=0, remesh=False)

    return vertices, faces


def sort_mesh(vertices, faces):
    # sort vertices
    sort_inds = np.lexsort((vertices[:, 0], vertices[:, 2], vertices[:, 1])) # [N], sort in y-z-x order (last key is first sorted)
    vertices = vertices[sort_inds]

    # re-index faces
    inv_inds = np.argsort(sort_inds)
    faces = inv_inds[faces]

    # cyclically permute each face's 3 vertices, and place the lowest vertex first
    start_inds = faces.argmin(axis=1) # [M]
    all_inds = start_inds[:, None] + np.arange(3)[None, :] # [M, 3]
    faces = np.concatenate([faces, faces[:, :2]], axis=1) # [M, 5], ABC --> ABCAB
    faces = np.take_along_axis(faces, all_inds, axis=1) # [M, 3]

    # sort among faces (faces.sort(0) will break each face, so we have to sort as list)
    faces = faces.tolist()
    faces.sort()
    faces = np.array(faces)

    return vertices, faces