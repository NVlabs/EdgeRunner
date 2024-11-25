
# EdgeRunner

This is the official implementation of *EdgeRunner: Auto-regressive Auto-encoder for Efficient Mesh Generation*.

### [Project Page](https://research.nvidia.com/labs/dir/edgerunner/) | [Arxiv](https://arxiv.org/abs/2409.18114) | [Mesh Tokenizer](https://github.com/NVlabs/EdgeRunner/tree/main/meto)


https://github.com/user-attachments/assets/b3444733-720e-4514-afb9-4db46e1ee202


- [x] Release training and inference code.
- [ ] Release pretrained checkpoints.

## Install

Make sure `torch` with CUDA is correctly installed.
For training, we rely on `flash-attn` (requires at least Ampere GPUs like A100). For inference, older GPUs like V100 are also supported, although slower.

```bash
# clone
git clone https://github.com/NVlabs/EdgeRunner
cd EdgeRunner

# install flash-attn
pip install flash-attn --no-build-isolation

# install meto the mesh tokenizer
pip install -e ./meto

# install other dependencies
pip install -r requirements.txt
```

We also provide a [Dockerfile](./Dockerfile) for easy setup.


## Training

**NOTE**: 
Since the dataset used in our training is based on AWS, it cannot be directly used for training in a new environment.
We provide the necessary training code framework, please check and modify the [dataset](./core/provider.py) implementation!

```bash
# debug training
accelerate launch --config_file acc_configs/gpu1.yaml main.py ArAE --workspace workspace_train
accelerate launch --config_file acc_configs/gpu1.yaml main_dit.py DiT --workspace workspace_train_dit

# single-node training (use slurm for multi-nodes training)
accelerate launch --config_file acc_configs/gpu8.yaml main.py ArAE --workspace workspace_train
accelerate launch --config_file acc_configs/gpu8.yaml main_dit.py DiT --workspace workspace_train_dit
```

**Training details**:
* We train the ArAE model on 64 A100 (80GB) GPUs for approximately one week. At a batch size of 4, each training iteration takes about 4 seconds. Expected training loss at convergence should be around `0.315`.
* We train the DiT model on 16 A100 (40GB) GPUs for approximately one week. At a batch size of 32, each training iteration takes about 7 seconds. Expected training loss at convergence should be around `0.0018`.

## Inference

Inference takes about 16GB GPU memory.

```bash
### point cloud conditioned 
# --workspace: path to save outputs.
# --resume: path to pretrained ArAE checkpoint.
# --test_path: can be either a directory or a single file of mesh. We will randomly sample surface points from it.
# --generate_mode: choose from ['greedy', 'sample'], strategy for auto-regressive generation.
# --test_num_face: targeted number of face to generate, choose from [-1, 1000, 2000, 4000], usually 1000 gives most robust results.
# --test_repeat: number of times to repeat the inference with different random seeds.
# --seed: initial random seed.
python infer.py ArAE --workspace workspace --resume pretrained/ArAE.safetensors --test_path data_mesh/ --generate_mode sample --test_num_face 1000 --test_repeat 3 --seed 42

### image conditioned
# --resume2: path to pretrained DiT checkpoint.
# --test_path: can be either a directory or a single file of image.
python infer_dit.py DiT --workspace workspace --resume pretrained/ArAE.safetensors --resume2 pretrained/DiT.safetensors  --test_path data_images/ --generate_mode sample --test_num_face 1000 --test_repeat 3 
```

## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [MeshAnything](https://github.com/buaacyw/MeshAnything)
- [transformers](https://github.com/huggingface/transformers)
- [tyro](https://github.com/brentyi/tyro)

## Citation

```
@article{tang2024edgerunner,
  title={EdgeRunner: Auto-regressive Auto-encoder for Artistic Mesh Generation},
  author={Tang, Jiaxiang and Li, Zhaoshuo and Hao, Zekun and Liu, Xian and Zeng, Gang and Liu, Ming-Yu and Zhang, Qinsheng},
  journal={arXiv preprint arXiv:2409.18114},
  year={2024}
}
```
