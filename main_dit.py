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
import math
import time
import shutil

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import DummyOptim, DummyScheduler
from safetensors.torch import load_file

from core.options import AllConfigs
from core.models_dit import MDiT
from core.provider_dit import ObjaverseDataset
from core.utils import init_logger

import kiui

# torch.autograd.set_detect_anomaly(True)

def main():    
    opt = tyro.cli(AllConfigs)

    # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        mixed_precision=opt.mixed_precision,
        gradient_accumulation_steps=opt.gradient_accumulation_steps,
        # kwargs_handlers=[ddp_kwargs],
    )

    os.makedirs(opt.workspace, exist_ok=True)
    logfile = os.path.join(opt.workspace, 'log.txt')
    logger = init_logger(logfile)

    # print options
    accelerator.print(opt)
    
    # model
    model = MDiT(opt)

    # resume
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:                    
                    logger.warning(f'mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                logger.warning(f'unexpected param {k}: {v.shape}')
    
    # resume2
    if opt.resume2 is not None:
        if opt.resume2.endswith('safetensors'):
            ckpt = load_file(opt.resume2, device='cpu')
        else:
            ckpt = torch.load(opt.resume2, map_location='cpu')
        
        # tolerant load (only load matching shapes)
        state_dict = model.state_dict()
        for k, v in ckpt.items():
            if k in state_dict: 
                if state_dict[k].shape == v.shape:
                    state_dict[k].copy_(v)
                else:                    
                    logger.warning(f'mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.')
            else:
                logger.warning(f'unexpected param {k}: {v.shape}')
    
    # count params
    num_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f'trainable param num: {num_p/1024/1024:.6f} M, total param num: {total_p/1024/1024:.6f}')

    # data
    train_dataset = ObjaverseDataset(opt, training=True)
    
    logger.info(f'train dataset size: {len(train_dataset)}')

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_dataset = ObjaverseDataset(opt, training=False)

    logger.info(f'test dataset size: {len(test_dataset)}')
    
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # optimizer
    if opt.use_deepspeed:
        # deepspeed will handle optimizer and scheduler (set in acc_configs/zero3_offload.json)
        optimizer = DummyOptim(model.parameters(), lr=opt.lr)
        scheduler = DummyScheduler(optimizer)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.1, betas=(0.9, 0.95))
  
        total_steps = opt.num_epochs * len(train_dataloader) // opt.gradient_accumulation_steps
        def _lr_lambda(current_step, warmup_ratio=opt.warmup_ratio, num_cycles=0.5, min_ratio=0.1):
            progress = current_step / max(1, total_steps)
            if warmup_ratio > 0 and progress < warmup_ratio:
                return progress / warmup_ratio
            progress = (progress - warmup_ratio) / (1 - warmup_ratio)
            return max(min_ratio, min_ratio + (1 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # accelerate
    model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, test_dataloader, scheduler
    )

    # wandb
    if opt.use_wandb and accelerator.is_main_process:
        import wandb # set WAND_API_KEY in env
        wandb.init(project='lmm', name=opt.workspace.replace('workspace_', ''), config=opt)

    # loop
    old_save_dirs = []
    best_loss = 1e9
    for epoch in range(opt.num_epochs):

        save_dir = os.path.join(opt.workspace, f'ep{epoch:04d}')
        os.makedirs(save_dir, exist_ok=True)

        # train
        if not opt.debug_eval:
            model.train()
            total_loss = 0
            t_start = time.time()
            for i, data in enumerate(train_dataloader):
                with accelerator.accumulate(model):

                    optimizer.zero_grad()

                    step_ratio = (epoch + i / len(train_dataloader)) / opt.num_epochs
                    step_ratio = opt.resume_step_ratio + (1 - opt.resume_step_ratio) * step_ratio

                    out = model(data, step_ratio)
                    loss = out['loss']

                    accelerator.backward(loss)

                    # gradient clipping
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), opt.gradient_clip)

                    optimizer.step()
                    scheduler.step()

                    total_loss += out['loss'].detach()

                if accelerator.is_main_process:
                    # logging
                    if i % 10 == 0:
                        mem_free, mem_total = torch.cuda.mem_get_info()
                        log = f"{epoch:03d}:{i}/{len(train_dataloader)} mem: {(mem_total-mem_free)/1024**3:.2f}/{mem_total/1024**3:.2f}G lr: {scheduler.get_last_lr()[0]:.7f} loss: {loss.item():.6f}"
                        logger.info(log)
                        
            total_loss = accelerator.gather_for_metrics(total_loss).mean().item()
            torch.cuda.synchronize()
            t_end = time.time()
            if accelerator.is_main_process:
                total_loss /= len(train_dataloader)
                logger.info(f"Train epoch: {epoch} loss: {total_loss:.6f} time: {(t_end - t_start)/60:.2f}min")
            
                # wandb
                if opt.use_wandb:
                    wandb.log({'train_loss': total_loss})
            
            # checkpoint
            # if epoch % 10 == 0 or epoch == opt.num_epochs - 1:
            accelerator.wait_for_everyone()
            accelerator.save_model(model, save_dir)
            if accelerator.is_main_process:
                # symlink latest checkpoint for linux
                if os.name == 'posix':
                    os.system(f'ln -sf {os.path.join(f"ep{epoch:04d}", "model.safetensors")} {os.path.join(opt.workspace, "model.safetensors")}')
                # copy best checkpoint
                if total_loss < best_loss:
                    best_loss = total_loss
                    shutil.copy(os.path.join(save_dir, 'model.safetensors'), os.path.join(opt.workspace, 'best.safetensors'))
                old_save_dirs.append(save_dir)
                if len(old_save_dirs) > 2: # save at most 2 ckpts
                    shutil.rmtree(old_save_dirs.pop(0))
        else:
            if accelerator.is_main_process:
                logger.info(f"epoch: {epoch} skip training for debug !!!")

        # eval
        if opt.eval_mode == 'loss':
            model.eval()
            with torch.no_grad():
                total_loss = 0
                unwrapped_model = accelerator.unwrap_model(model)
                for i, data in enumerate(test_dataloader):
                    out = model(data)
                    loss = out['loss'] 
                    total_loss += loss.detach()

                total_loss = accelerator.gather_for_metrics(total_loss).mean()
                if accelerator.is_main_process:
                    total_loss /= len(test_dataloader)
                    logger.info(f"Eval epoch: {epoch} loss: {total_loss:.6f}")
    
        else:
            if accelerator.is_main_process:
                logger.info(f"Eval epoch: {epoch} skip evaluation.")
            

if __name__ == "__main__":
    main()
