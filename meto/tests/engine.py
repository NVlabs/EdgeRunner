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

import sys
sys.path.append('.')

import os
import trimesh
import numpy as np
import argparse

import kiui
from meto import Engine, load_mesh, normalize_mesh, sort_mesh

def write_obj(vertices, faces, filename):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

parser = argparse.ArgumentParser()
parser.add_argument('mesh', type=str, help='path to the mesh file')
parser.add_argument('--backend', type=str, default='LR_ABSCO', help='engine backend')
parser.add_argument('--verbose', action='store_true', help='print verbose output')
parser.add_argument('--output', type=str, default='output.obj', help='path to the output file')
opt = parser.parse_args()


if opt.mesh == 'plane':        
    # plane of two triangles
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
elif opt.mesh == 'tetrahedron':
    # tetrhedron
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 0.5, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int32)
elif opt.mesh == 'cube':
    # cube
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0], [4, 7, 6], [4, 6, 5]], dtype=np.int32)
elif opt.mesh == 'see':
    # a simple case that encodes to SEE
    vertices = np.array([[0.5, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 4, 1]], dtype=np.int32)
elif opt.mesh == 'lrlre':
    # a simple case that encodes to LRLRE
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0], [-1, 2, 0], [-2, 2, 0]], dtype=np.float32)
    vertices = normalize_mesh(vertices)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [4, 3, 5], [5, 4, 6]], dtype=np.int32)
elif opt.mesh == 'lRlre':
    # flip the second triangle of the previous case, this will lead to inconsistent face orientation
    # but our algorithm should be able to detect and correct it!
    vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0], [-1, 2, 0], [-2, 2, 0]], dtype=np.float32)
    vertices = normalize_mesh(vertices)
    faces = np.array([[0, 1, 2], [0, 3, 2], [0, 3, 4], [4, 3, 5], [5, 4, 6]], dtype=np.int32)
elif opt.mesh == 'mtype':
    # m-type
    vertices = np.array([[1, 0, 0], [3, 0, 0], [2, 1, 0], [4, 1, 0], [3, 2, 0], [4, 3, 0], [2, 3, 0], [1, 2, 0], [0, 3, 0], [0, 1, 0]])
    vertices = normalize_mesh(vertices)
    faces = np.array([[0, 1, 2], [1, 3, 2], [4, 2, 3], [5, 4, 3], [6, 4, 5], [6, 7, 4], [8, 7, 6], [8, 9, 7], [7, 9, 2], [9, 0, 2]], dtype=np.int32)
elif opt.mesh == 'mtype_fake':
    # m-type
    vertices = np.array([[1, 0, 0], [3, 0, 0], [2, 1, 0], [4, 1, 0], [3, 2, 0], [4, 3, 0], [2, 3, 0], [1, 2, 0], [0, 3, 0], [0, 1, 0]])
    vertices = normalize_mesh(vertices)
    faces = np.array([[7, 2, 4], [0, 1, 2], [1, 3, 2], [4, 2, 3], [5, 4, 3], [6, 4, 5], [6, 7, 4], [8, 7, 6], [8, 9, 7], [7, 9, 2], [9, 0, 2]], dtype=np.int32)
elif opt.mesh == 'mtype2':
    # m'-type
    vertices = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1], [1, 0, 1], [2, 1, 1], [2, 0, 1]])
    vertices = normalize_mesh(vertices)
    faces = np.array([[1, 0, 2], [2, 0 ,3], [2, 3, 4], [4, 3 ,5], [4, 5, 1], [1, 5, 0]], dtype=np.int32)
elif opt.mesh == 'torus':
    # m'-type
    vertices = np.array([
        [2, 0, 0], [2, 1, 0], [4, 1, 0], [3, 2, 0], [4, 3, 0], [2, 3, 0], [1, 2, 0], [0, 3, 0], [0, 1, 0],
        [2, 0, 1], [2, 1, 1], [4, 1, 1], [3, 2, 1], [4, 3, 1], [2, 3, 1], [1, 2, 1], [0, 3, 1], [0, 1, 1],
    ])
    vertices = normalize_mesh(vertices)
    faces = np.array([
        [1, 2, 0], [2, 1, 3], [2, 3, 4], [4, 3, 5], [3, 6, 5], [5, 6, 7], [6, 8, 7], [1, 8, 6], [1, 0, 8],
        [9, 11, 10], [12, 10, 11], [13, 12, 11], [14, 12, 13], [14, 15, 12], [16, 15, 14], [16, 17, 15], [15, 17, 10], [17, 9, 10],
        [8, 0, 17], [9, 17, 0], [9, 0, 2], [11, 9, 2], [11, 2, 4], [13, 11, 4], [13, 4, 5], [14, 13, 5], [14, 5, 7], [16, 14, 7], [16, 7, 8], [17, 16, 8],
        [10, 1, 6], [15, 10, 6], [12, 3, 1], [10, 12, 1], [15, 6, 3], [12, 15, 3],
    ], dtype=np.int32)
elif opt.mesh == 'torus_fake':
    # m'-type fake
    vertices = np.array([
        [2, 0, 0], [2, 1, 0], [4, 1, 0], [3, 2, 0], [4, 3, 0], [2, 3, 0], [1, 2, 0], [0, 3, 0], [0, 1, 0],
        [2, 0, 1], [2, 1, 1], [4, 1, 1], [3, 2, 1], [4, 3, 1], [2, 3, 1], [1, 2, 1], [0, 3, 1], [0, 1, 1],
    ])
    vertices = normalize_mesh(vertices)
    faces = np.array([
        [0, 2, 1], [3, 1, 2], [4, 3, 2], [5, 3, 4], [5, 6, 3], [7, 6, 5], [7, 8, 6], [6, 8, 1], [8, 0, 1],
        [9, 11, 10], [12, 10, 11], [13, 12, 11], [14, 12, 13], [14, 15, 12], [16, 15, 14], [16, 17, 15], [15, 17, 10], [17, 9, 10],
        [0, 8, 17], [0, 17, 9], [2, 0, 9], [2, 9, 11], [4, 2, 11], [4, 11, 13], [5, 4, 13], [5, 13, 14], [7, 5, 14], [7, 14, 16], [8, 7, 16], [8, 16, 17],
        # [6, 1, 10], [6, 10, 15], [1, 3, 12], [1, 12, 10], [3, 6, 15], [3, 15, 12],
    ], dtype=np.int32)
elif opt.mesh == 'sphere':
    # sphere
    mesh = trimesh.creation.icosphere(subdivisions=2)
    vertices = mesh.vertices
    vertices = normalize_mesh(vertices)
    faces = mesh.faces
elif opt.mesh == 'annulus':
    # annulus
    mesh = trimesh.creation.annulus(0.5, 1, 1)
    vertices = mesh.vertices
    vertices = normalize_mesh(vertices)
    faces = mesh.faces
else:
    # load from file
    vertices, faces = load_mesh(opt.mesh, clean=True)

# sort mesh
# vertices, faces = sort_mesh(vertices, faces) # sort in cpp now

# engine
engine = Engine(2048, verbose=opt.verbose, backend=opt.backend)
write_obj(vertices, faces, os.path.join('data', os.path.basename(opt.mesh) + '.obj'))
tokens, face_order, face_type = engine.encode(vertices, faces)
vertices2, faces2, face_type2 = engine.decode(tokens)
write_obj(vertices2, faces2, opt.output)

print(f"[INFO] input vertices: {vertices.shape[0]}, faces: {faces.shape[0]}")
print(f"[INFO] encoded tokens: {len(tokens)}, ratio = {100 * len(tokens) / (9 * faces.shape[0]):.2f} %")
print(f"[INFO] decoded vertices: {vertices2.shape[0]}, faces: {faces2.shape[0]}")
kiui.lo(tokens)
