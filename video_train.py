import gc
import glob
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from collections import deque
from contextlib import nullcontext
from functools import partial
from typing import List, Optional, Tuple

import GPUtil
import colorama
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.profiler import record_function
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm

import dist
from utils import arg_util, misc
from utils.datasets import DataLoader, DistributedSampler, DatasetFromCSV, get_transforms_video
from utils.data_sampler import DistInfiniteBatchSampler
from models import build_video_vae_disc, VQVideoVAE, DinoDisc
from videotrainer import VideoVAETrainer
from utils.amp_opt import AmpOptimizer
from utils.lpips import LPIPS
from utils.lr_control import filter_params
from utils import optimizer


def create_tb_lg(args: arg_util.Args):
    tb_lg: misc.TensorboardLogger
    with_tb_lg = dist.is_master()
    if with_tb_lg:
        os.makedirs(args.tb_log_dir_path, exist_ok=True)
        
    else:   
        tb_lg = misc.DistLogger(None)
    dist.barrier()
    return tb_lg


def maybe_auto_resume(args: arg_util.Args, pattern='ckpt*.pth') -> Tuple[List[str], int, int, str, List[Tuple[float, float]], dict, dict]:
    info = []
    resume = None
    if len(args.resume):
        resume = args.resume
        info.append(f'[auto_resume] load from args.resume @ {resume} ...')
    elif not args.local_debug:
        all_ckpt = lyoko.glob_with_latest_modified_first(os.path.join(args.bed, pattern))
        if len(all_ckpt) == 0:
            resume = resume
            info.append(f'[auto_resume] no ckpt found @ {pattern}')
            info.append(f'[auto_resume quit]')
        else:
            resume = all_ckpt[0]
            info.append(f'[auto_resume] auto load from @ {resume} ...')
        info.append(f'[auto_resume quit]')
    else:
        info.append(f'[auto_resume] disabled')
        info.append(f'[auto_resume quit]')
    
    if resume is None:
        return info, 0, 0, '[no acc str]', [], {}, {}
    
    try:
        ckpt = torch.load(resume, map_location='cpu')
    except Exception as e:
        info.append(f'[auto_resume] failed, {e} @ {resume}')
        return info, 0, 0, '[no acc str]', [], {}, {}
    
    dist.barrier()
    ep, it = (ckpt['epoch'], ckpt['iter']) if 'iter' in ckpt else (ckpt['epoch'] + 1, 0)
    eval_milestone = ckpt.get('milestones', [])
    info.append(f'[auto_resume success] resume from ep{ep}, it{it},    eval_milestone: {eval_milestone}')
    return info, ep, it, ckpt.get('acc_str', '[no acc str]'), eval_milestone, ckpt['trainer'], ckpt['args']
    

def build_things_from_args(args: arg_util.Args):
    # set seed
    auto_resume_info, 
    start_ep, 
    start_it, 
    acc_str, 
    eval_milestone, 
    trainer_state, 
    args_state = maybe_auto_resume(args, 'ckpt*.pth')
    args.load_state_dict_vae_only(args_state)
    args.diffs = ' '.join([f'{d:.3f}'[2:] for d in eval_milestone])
    tb_lg = create_tb_lg(args)
    print(f'global bs={args.bs}, local bs={args.lbs}')
    print(f'initial args:\n{str(args)}')

    if start_ep == args.ep:
        print(f'[vlip] Training finished ({acc_str}), skipping ...\n\n')
        return args, tb_lg
    
    # build video dataset (OpenVid 1M)
    if not args.local_debug:
        print(f'[build video dataset] ...\n')

        dataset_full = DatasetFromCSV(         # Test settings:
            csv_path=args.data,                 # csv_path = r'/projectnb/ec720prj/DenseCap/vaex/OpenVid/OpenVid-1M-108part.csv'
            transform=get_transforms_video(),
            num_frames=args.num_frames,         # num_frames = 16
            frame_interval=args.frame_interval, # frame_interval = 3
            root=args.root,                     # root = r'/projectnb/ec720prj/DenseCap/vaex/OpenVid/video'
        )

        # calculate train_size and val_size
        val_size = int(len(dataset_full) * args.val_ratio)
        train_size = len(dataset_full) - val_size
        
        # split dataset into train and val
        dataset_train, dataset_val = torch.utils.data.random_split(dataset_full, [train_size, val_size])

        sampler_train = DistributedSampler(
            dataset_train,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=1
        )

        sampler_val = DistributedSampler(
            dataset_val,
            num_replicas=1,
            rank=0,
            shuffle=False,
            seed=1
        )

        loader_train = DataLoader(
            dataset_train,
            batch_size=8,
            shuffle=False,
            sampler=sampler_train,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )

        loader_val = DataLoader(
            dataset_val,
            batch_size=8,
            shuffle=False,
            sampler=sampler_val,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        del dataset_train, dataset_val


    # import heavy packages after Dataloader object creation
    from torch.nn.parallel import DistributedDataParallel as DDP
    from videotrainer import VideoVAETrainer
    from utils.amp_opt import AmpOptimizer
    from utils.lr_control import lr_wd_annealing

    # build models
    vae_wo_ddp, disc_wo_ddp = build_video_vae_disc(args)
    vae_wo_ddp: VQVideoVAE
    disc_wo_ddp: DinoDisc

    print(f'[PT] VAE model ({args.vae}) = {vae_wo_ddp}\n')
    if isinstance(disc_wo_ddp, DinoDisc):
        print(f'[PT] Disc model (frozen part) = {disc_wo_ddp.dino_proxy[0]}\n')
    print(f'[PT] Disc model (trainable part) = {disc_wo_ddp}\n\n')

    assert all(p.requires_grad for p in vae_wo_ddp.parameters())
    assert all(p.requires_grad for p in disc_wo_ddp.parameters())
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('VAE', vae_wo_ddp), ('VAE.enc', vae_wo_ddp.encoder), ('VAE.dec', vae_wo_ddp.decoder), ('VAE.quant', vae_wo_ddp.quantize)
    )]))
    print(f'[PT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (
        ('Disc', disc_wo_ddp),
    )]) + '\n\n')

    # build optiimzers
    optimizer: List[AmpOptimizer] = []
    for model_name, model_wo_ddp, opt_beta, lr, wd, clip in (('vae', vae_wo_ddp, args.vae_opt_beta, args.vae_lr, args.vae_wd, args.grad_clip), ('dis', disc_wo_ddp, args.disc_opt_beta, args.disc_lr, args.disc_wd, args.grad_clip)):
        if p.requires_grad:
                dist.broadcast(p.data, src_rank=0)
        ndim_dict = {name: para.ndim for name, para in model_wo_ddp.named_parameters() if para.requires_grad}
    
        # build optimizer
        nowd_keys = {
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'class_emb', 'embedding',
            'norm_scale',
        }
        names, paras, para_groups = filter_params(model_wo_ddp, ndim_dict, nowd_keys=nowd_keys)

        beta1, beta2 = map(float, opt_beta.split('_'))
        opt_clz = {
            'adam':  partial(torch.optim.AdamW, betas=(beta1, beta2), fused=args.fuse_opt),
            'adamw': partial(torch.optim.AdamW, betas=(beta1, beta2), fused=args.fuse_opt),
            'lamb':  partial(optimizer.LAMBtimm, betas=(beta1, beta2), max_grad_norm=clip), # eps=1e-7
            'lion':  partial(optimizer.Lion, betas=(beta1, beta2), max_grad_norm=clip),     # eps=1e-7
        }[args.opt]
        opt_kw = dict(lr=lr, weight_decay=0)
        if args.oeps: opt_kw['eps'] = args.oeps

        print(f'[vlip] optim={opt_clz}, opt_kw={opt_kw}\n')
        optimizers.append(AmpOptimizer(model_name, model_maybe_fsdp=None, fp16=args.fp16, bf16=args.bf16, zero=args.zero, optimizer=opt_clz(params=para_groups, **opt_kw), grad_clip=clip, n_gradient_accumulation=args.grad_accu))
        del names, paras, para_groups
    vae_optim, disc_optim = optimizers[0], optimizers[1]

    vae_wo_ddp, disc_wo_ddp = args.compile_model(vae_wo_ddp, args.compile_vae), args.compile_model(disc_wo_ddp, args.compile_disc)
    lpips_loss: LPIPS = args.compile_model(LPIPS(args.lpips_path).to(args.device), fast=args.compile_lpips)
    
    
    # distributed wrapper
    ddp_class = DDP if dist.initialized() else NullDDP
    vae: DDP = ddp_class(vae_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, static_graph=args.ddp_static, broadcast_buffers=False)
    disc: DDP = ddp_class(disc_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, static_graph=args.ddp_static, broadcast_buffers=False)
    
    vae_optim.model_maybe_fsdp = vae if args.zero else vae_wo_ddp
    disc_optim.model_maybe_fsdp = disc if args.zero else disc_wo_ddp

    # build trainer
    trainer = VideoVAETrainer(
        is_visualizer=dist.is_master(),
        vae=vae, 
        vae_wo_ddp=vae_wo_ddp, 
        disc=disc, 
        disc_wo_ddp=disc_wo_ddp, 
        ema_ratio=args.ema,
        dcrit=args.dcrit, 
        vae_opt=vae_optim, 
        disc_opt=disc_optim,
        daug=args.disc_aug_prob, 
        lpips_loss=lpips_loss, 
        lp_reso=args.lpr, 
        wei_l1=args.l1, 
        wei_l2=args.l2, 
        wei_entropy=args.le, 
        wei_lpips=args.lp, 
        wei_disc=args.ld, 
        adapt_type=args.gada, 
        bcr=args.bcr, 
        bcr_cut=args.bcr_cut, 
        reg=args.reg, 
        reg_every=args.reg_every,
        disc_grad_ckpt=args.disc_grad_ckpt,
        dbg_unused=args.dbg_unused, dbg_nan=args.dbg_nan
    )

    if trainer_state is not None and len(trainer_state):
        trainer.load_state_dict(trainer_state, strict=False)
    del vae, vae_wo_ddp, disc, disc_wo_ddp, vae_optim, disc_optim
    
    iters_train = len(loader_train)
    ld_train = iter(loader_train)
    # visualization to do
    
    del inp, label, val_transform
    return (
        tb_lg, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train,
    )


g_speed_ls = deque(maxlen=128)
def train_one_ep(ep: int, is_first_ep: bool, start_it: int, saver: CKPTSaver, args: arg_util.Args, tb_lg: misc.TensorboardLogger, ld_or_itrt, iters_train: int, trainer, logging_params_milestone):
    """
    Args:
        ep: current epoch
        is_first_ep: whether it is the first epoch
        start_it: start iteration
        saver: checkpoint saver
        args: training parameters
        tb_lg: tensorboard logger
        ld_or_itrt: data loader or iterator
        iters_train: training iterations
        trainer: trainer
        logging_params_milestone: logging parameters milestone
    """
    trainer: VideoVAETrainer

    step_cnt = 0
    me = misc.MetricLogger(delimiter='  ')
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{value:.2g}')) for x in ['glr', 'dlr']]
    [me.add_meter(x, misc.SmoothedValue(window_size=1, fmt='{median:.2f} ({global_avg:.2f})')) for x in ['gnm', 'dnm']]
    for l in ['L1', 'NLL', 'Ld', 'Wg']:
        me.add_meter(l, misc.SmoothedValue(fmt='{median:.3f} ({global_avg:.3f})'))
    header = f'[Ep]: [{ep:4d}/{args.ep}]'
    
    touching_secs = 120
    if is_first_ep:
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('ignore', category=UserWarning)
    g_it, wp_it, max_it = ep * iters_train, args.warmup_ep * iters_train, args.ep * iters_train
    disc_start = args.disc_start_ep * iters_train
    disc_wp_it, disc_max_it = args.disc_warmup_ep * iters_train, max_it - disc_start
    
    doing_profiling = args.prof and is_first_ep and (args.profall or dist.is_master())
    maybe_record_function = record_function if doing_profiling else nullcontext
    trainer.vae_wo_ddp.maybe_record_function = maybe_record_function
    
    if args.zero:
        pref = 'hybrid' if args.hsdp else 'fsdp'
        if args.buck in {'0', '0.0', '0e0', '0.0e0'}:
            parallel = f'ep{ep}_{pref}{args.zero}_module_orig{args.fsdp_orig:d}'
        else:
            parallel = f'ep{ep}_{pref}{args.zero}_buk{args.buck}_orig{args.fsdp_orig:d}'

    if os.getenv('NCCL_CROSS_NIC', '0') == '1':
        parallel += f'_NIC1'
    profiling_name = f'{args.vae}_bs{args.bs}_{parallel}_gradckpt{args.vae_grad_ckpt:d}__GPU{dist.get_rank_str_zfill()}of{dist.get_world_size()}'

    profiler = None
    if doing_profiling:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=40,
                warmup=3,
                active=2,
                repeat=1,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=TraceHandler('./', f'{profiling_name}.pt.trace.json', args.tos_profiler_file_prefix, args.bed)
        )
        profiler.start()

    last_t_perf = time.perf_counter()
    speed_ls: deque = g_speed_ls
    FREQ = min(50, iters_train//2-1)
    NVIDIA_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8))
    PRINTABLE_IT_PLUS_1 = set(FREQ*i for i in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32))
    
    for it, (videos, _) in me.log_every(start_it, iters_train, ld_or_itrt, max(10, iters_train // 1000), header):
        if (it+1) % FREQ == 0:
            # update learning rate
            warmup_disc_schedule = min(1.0, max(0.0, (it - disc_start) / disc_wp_it))
            fade_blur_schedule = min(1.0, max(0.0, (it - disc_start) / disc_wp_it))
            
            # train one step
            loss, recon_loss, vq_loss, commit_loss = trainer.train_step(
                ep=ep,
                it=it,
                g_it=g_it,
                stepping=True,
                regularizing=True,
                metric_lg=me,
                logging_params=logging_params_milestone,
                tb_lg=tb_lg,
                inp=videos,
                warmup_disc_schedule=warmup_disc_schedule,
                fade_blur_schedule=fade_blur_schedule,
                maybe_record_function=maybe_record_function,
                args=args
            )
            
            # record loss
            me.update('L1', recon_loss)
            me.update('NLL', vq_loss)
            me.update('Ld', commit_loss)
            me.update('Wg', loss)
            
            # update learning rate
            if it < wp_it:
                lr_wd_annealing(trainer.vae_opt, it, wp_it, args.vae_lr, args.vae_wd)
            if disc_start <= it < disc_start + disc_wp_it:
                lr_wd_annealing(trainer.disc_opt, it - disc_start, disc_wp_it, args.disc_lr, args.disc_wd)
            
            # record learning rate
            me.update('glr', trainer.vae_opt.param_groups[0]['lr'])
            me.update('dlr', trainer.disc_opt.param_groups[0]['lr'])
            
            # record gradient norm
            me.update('gnm', trainer.vae_wo_ddp.get_grad_norm())
            me.update('dnm', trainer.disc_wo_ddp.get_grad_norm())
            
            # save checkpoint
            if (it+1) % args.save_interval == 0:
                saver.save(ep, it, trainer, args)
            
            # record performance metrics
            if (it+1) in PRINTABLE_IT_PLUS_1:
                t_perf = time.perf_counter()
                speed = (it+1) / (t_perf - last_t_perf)
                speed_ls.append(speed)
                print(f'[Speed] {speed:.2f} it/s, avg: {sum(speed_ls)/len(speed_ls):.2f} it/s')
                last_t_perf = t_perf
            
            # update profiler
            if profiler is not None:
                profiler.step()
    
    # end profiler
    if profiler is not None:
        profiler.stop()
    
    # return training metrics
    return {
        'loss': me.get('Wg'),
        'recon_loss': me.get('L1'),
        'vq_loss': me.get('NLL'),
        'commit_loss': me.get('Ld'),
        'vae_lr': me.get('glr'),
        'disc_lr': me.get('dlr'),
        'vae_grad_norm': me.get('gnm'),
        'disc_grad_norm': me.get('dnm')
    }


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def main_training():
    """Main training function"""
    # initialize distributed training and parameters
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    if args.dbg_unused:
        torch.autograd.set_detect_anomaly(True)
    
    # build training components
    ret = build_things_from_args(args)
    if len(ret) < 8:
        return ret
    (
        tb_lg, trainer,
        start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train,
    ) = ret

    # create checkpoint saver
    saver = CKPTSaver(dist.is_master(), eval_milestone)
    
    # training loop
    for ep in range(start_ep, args.ep):
        # train one epoch
        metrics = train_one_ep(
            ep=ep,
            is_first_ep=(ep == start_ep),
            start_it=start_it if ep == start_ep else 0,
            saver=saver,
            args=args,
            tb_lg=tb_lg,
            ld_or_itrt=ld_train,
            iters_train=iters_train,
            trainer=trainer,
            logging_params_milestone=eval_milestone
        )
        
        # print epoch results
        print(f'\nEpoch {ep+1}/{args.ep} Results:')
        print(f'Loss: {metrics["loss"]:.4f}')
        print(f'Reconstruction Loss: {metrics["recon_loss"]:.4f}')
        print(f'VQ Loss: {metrics["vq_loss"]:.4f}')
        print(f'Commit Loss: {metrics["commit_loss"]:.4f}')
        print(f'VAE Learning Rate: {metrics["vae_lr"]:.6f}')
        print(f'Discriminator Learning Rate: {metrics["disc_lr"]:.6f}')
        print(f'VAE Gradient Norm: {metrics["vae_grad_norm"]:.4f}')
        print(f'Discriminator Gradient Norm: {metrics["disc_grad_norm"]:.4f}')
        
        # save checkpoint
        if dist.is_master():
            saver.save(ep, iters_train-1, trainer, args)
        
        # clean GPU memory
        gc.collect()
        torch.cuda.empty_cache()
        
        # synchronize all processes
        dist.barrier()
    
    # training completed
    print(f'\nTraining completed! Best accuracy: {acc_str}')
    return tb_lg, trainer, start_ep, start_it, acc_str, eval_milestone, iters_train, ld_train


if __name__ == '__main__':
    try:
        main_training()
    finally:
        dist.finalize()
        if isinstance(sys.stdout, dist.BackupStreamToFile) and isinstance(sys.stderr, dist.BackupStreamToFile):
            sys.stdout.close(), sys.stderr.close()



            

