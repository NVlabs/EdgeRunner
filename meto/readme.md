# Meto

This repo implements a mesh tokenization algorithm modified from [EdgeBreaker](https://faculty.cc.gatech.edu/~jarek/papers/EdgeBreaker.pdf), as described in the [EdgeRunner](https://research.nvidia.com/labs/dir/edgerunner/) paper.

The core functions are implemented in c++ and binded to python.

Typical compression ratio compared to flattening the triangles (9 floats per triangle) can be 50% (4-5 integers per triangle).


https://github.com/user-attachments/assets/b0132687-69ab-40a6-86b3-935fa5ef0f62


## Install

```bash
# locally
pip install -e .
```

## Usage

To encode (tokenize) or decode (de-tokenize) a mesh:

```python
from meto import Engine, load_mesh

### load mesh
# clean: merge close/duplicate vertices (which is common for uv-unwrapped models!)
vertices, faces = load_mesh('mesh.obj', clean=True)

### initialize 
# discrete_bins: quantize vertex coordinates to 512 discrete values
# verbose: print detailed logs
# backend: choose from ['CLERS', 'LR', 'LR_ABSCO'], see below for explanation
engine = Engine(discrete_bins=512, verbose=True, backend='LR_ABSCO')

### encode
# tokens: [N], int, encoded tokens (both face and vertex tokens)
# face_order: [M], int, order of face in traversal
# face_type: [M], int, op type of face in traversal (only face tokens)
tokens, face_order, face_type = engine.encode(vertices, faces)

### decode
# vertices2: [V, 3], float, decoded vertices
# faces2: [F, 3], int, decoded faces
# face_type2: [F], int, op type of face in traversal
# note that the decoded mesh still requires cleaning to remove duplicate vertices/faces.
vertices2, faces2, face_type2 = engine.decode(tokens)
```

We provide some examples in the `tests` folder, including a GUI to visualize the encoding process:
```bash
### run encode and decode on a given mesh
python tests/engine.py mesh.obj

### open a GUI to visualize the encoding process (press left/right arrow key to show progress)
# extra dep: torch, dearpygui, nvdiffrast, opencv-python
python tests/gui.py --mesh mesh.obj
# visualize the decoding process
python tests/gui.py --mesh mesh.obj --decode 
```

## Backends

We have implemented multiple variants of the original [EdgeBreaker](https://faculty.cc.gatech.edu/~jarek/papers/EdgeBreaker.pdf) algorithm, which can be configured using the `backend` parameter in the `Engine` class. The available backends are:
* **`CLERS`**: most similar to the original algorithm, using 7 special tokens (C, L, E, R, S, BOM, EOM) and relative coordinate.
* **`LR`**: simplified variant using 3 special tokens (L, R, BOM) and relative coordinate.
* **`LR_ABSCO`**: `LR` variant using absolute coordinate. **This is the default backend and also the algorithm used in EdgeRunner paper.**


## Note
* Input: **duplicate (or close) vertices** should be merged together for building a correct graph.
* `discrete_bins` is the quantization resolution of the coordinates. By default we use 512, but it may not be enough for high-resolution meshes.

## Citation

```
@article{rossignac1999edgebreaker,
  title={Edgebreaker: Connectivity compression for triangle meshes},
  author={Rossignac, Jarek},
  journal={IEEE transactions on visualization and computer graphics},
  volume={5},
  number={1},
  pages={47--61},
  year={1999},
  publisher={IEEE}
}

@article{tang2024edgerunner,
  title={EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation},
  author={Tang, Jiaxiang and Li, Zhaoshuo and Hao, Zekun and Liu, Xian and Zeng, Gang and Liu, Ming-Yu and Zhang, Qinsheng},
  journal={arXiv preprint arXiv:2409.18114},
  year={2024}
}
```
