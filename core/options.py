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

import tyro
from dataclasses import dataclass
from typing import Tuple, Literal, Dict, Optional, List

@dataclass
class Options:

    ### tokenizer
    # coord discrete bins (also the number of basic tokens)
    discrete_bins: int = 512
    # enable meto
    use_meto: bool = True
    # meto backend
    meto_backend: Literal['LR', 'LR_ABSCO'] = 'LR_ABSCO'
    # special tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    ### point vae
    # number of samples
    point_num: int = 8192
    # hidden size
    point_hidden_dim: int = 1024
    # number of heads
    point_num_heads: int = 16
    # latent size
    point_latent_size: int = 2048
    # latent dim
    point_latent_dim: int = 64

    # number of decoder layers
    point_num_layers: int = 24
    # number of query points per training iter
    point_query_num: int = 81920
    # encoder mode
    point_encoder_mode: Literal['downsample', 'embed'] = 'embed'
    # kl weight
    kl_weight: float = 1e-8

    ### dit
    # dit hidden size
    dit_hidden_dim: int = 1024
    # dit number of heads
    dit_num_heads: int = 16
    # dit number of layers
    dit_num_layers: int = 24
    # diffusion snr gamma
    snr_gamma: Optional[float] = 5.0
    # diffusion scheduler predtype
    noise_scheduler_predtype: Literal["epsilon", "v_prediction"] = "v_prediction"
    
    ### lmm
    # freeze encoder
    freeze_encoder: bool = True
    # max sequence length (excluding BOS, EOS, and COND)
    max_seq_length: int = 10240 # for naive tokenizer, use max_face_length * 9; for meto, use max_face_length * 9 * 0.5
    # hidden size
    hidden_dim: int = 1024
    # intermediate mlp size
    intermediate_dim: Optional[int] = None
    # layers
    num_layers: int = 24
    # num head
    num_heads: int = 16
    # conditions
    cond_mode: Literal['none', 'image', 'point', 'point_latent'] = 'image'
    # length of condition tokens
    num_cond_tokens: int = 257
    # generate mode
    generate_mode: Literal['greedy', 'sample'] = 'sample'
    # num face condition
    use_num_face_cond: bool = False
    # number of face dropout ratio
    nof_dropout_ratio: float = 0.2

    ### dataset
    # max face length
    max_face_length: int = 1000
    # data set
    dataset: Literal['obj', 'objxl'] = 'obj'
    # num workers
    num_workers: int = 64
    # testset size
    testset_size: int = 32 # only if image cond now
    # decimate aug
    use_decimate_aug: bool = True
    # scale aug
    use_scale_aug: bool = True
    
    ### training
    # workspace
    workspace: str = './workspace'
    # resume ckpt path
    resume: Optional[str] = None
    resume2: Optional[str] = None
    # resume step_ratio
    resume_step_ratio: float = 0
    # pos embd align
    align_posemb: Literal['left', 'right'] = 'right'
    # batch size (per-GPU)
    batch_size: int = 4
    # gradient accumulation
    gradient_accumulation_steps: int = 1
    # training epochs
    num_epochs: int = 100
    # gradient clip
    gradient_clip: float = 1.0
    # mixed precision
    mixed_precision: Literal['no', 'fp8', 'fp16', 'fp32'] = 'bf16'
    # learning rate
    lr: float = 1e-4
    # gradient checkpointing
    checkpointing: bool = True
    # random seed
    seed: int = 0
    # use deepspeed
    use_deepspeed: bool = False
    # evaluate mode
    eval_mode: Literal['none', 'loss', 'generate'] = 'loss'
    # debug eval in train (skip training and only do evaluation)
    debug_eval: bool = False
    # lr warmup ratio
    warmup_ratio: float = 0.01
    # use wandb
    use_wandb: bool = False
    
    ### testing
    # test image/point path
    test_path: Optional[str] = None
    # test resume tokens
    test_resume_tokens: Optional[str] = None
    # test repeat
    test_repeat: int = 1
    # test targeted num faces (can be a list)
    test_num_face: Tuple[int, ...] = (1000,)
    # test max seq len
    test_max_seq_length: Optional[int] = None

    
# all the default settings
config_defaults: Dict[str, Options] = {}
config_doc: Dict[str, str] = {}

config_doc['default'] = 'the default settings'
config_defaults['default'] = Options()

config_doc['ArAE'] = 'ArAE'
config_defaults['ArAE'] = Options(
    point_encoder_mode='embed',
    kl_weight=1e-8,
    discrete_bins=512,
    use_num_face_cond=True,
    use_decimate_aug=True,
    cond_mode='point',
    num_cond_tokens=2049,
    freeze_encoder=False,
    use_meto=True,
    meto_backend='LR_ABSCO',
    max_face_length=4000,
    max_seq_length=40960,
    align_posemb='right',
    batch_size=4,
    hidden_dim=1536,
    num_heads=16,
    num_layers=24,
    gradient_accumulation_steps=1,
    lr=1e-5,
    warmup_ratio=0,
    num_epochs=100,
    eval_mode='loss',
)

config_doc['DiT'] = 'DiT'
config_defaults['DiT'] = Options(
    point_encoder_mode='embed',
    kl_weight=1e-8,
    max_face_length=8000,
    discrete_bins=512,
    use_num_face_cond=True,
    use_decimate_aug=False,
    cond_mode='point',
    num_cond_tokens=2049,
    freeze_encoder=False,
    use_meto=True,
    meto_backend='LR_ABSCO',
    max_seq_length=40960,
    hidden_dim=1536,
    num_heads=16,
    num_layers=24,
    dit_hidden_dim=1024,
    dit_num_heads=16,
    dit_num_layers=24,
    snr_gamma=5.0,
    noise_scheduler_predtype="v_prediction",
    batch_size=8,
    gradient_accumulation_steps=1,
    lr=1e-5,
    num_epochs=300,
    eval_mode='none',
)

AllConfigs = tyro.extras.subcommand_type_from_defaults(config_defaults, config_doc)