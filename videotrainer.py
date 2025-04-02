import sys
from copy import deepcopy
from pprint import pformat
from typing import Callable, Optional, Tuple

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torch.nn.parallel import DistributedDataParallel as DDP

from models import VectorQuantizer3, VQVideoVAE, DinoDisc
from utils import arg_util, misc, nan
from utils.amp_opt import AmpOptimizer
from utils.diffaug import DiffAug
from utils.loss import hinge_loss, linear_loss, softplus_loss
from utils.lpips import LPIPS



# Video VQVAE Trainer 

class VideoVAETrainer(object):
    def __init__(
        self, 
        is_visualizer: bool,
        vae: DDP, 
        vae_wo_ddp: VQVideoVAE, 
        disc: DDP, 
        disc_wo_ddp: DinoDisc, 
        ema_ratio: float,  # decoder, en_de_lin=True, seg_embed=False,
        dcrit: str, 
        vae_opt: AmpOptimizer, 
        disc_opt: AmpOptimizer,
        daug=1.0, 
        lpips_loss: LPIPS = None, 
        lp_reso=64, 
        wei_l1=1.0, 
        wei_l2=0.0,
        wei_entropy=0.0,
        wei_lpips=0.5, 
        wei_disc=0.6, 
        adapt_type=1, 
        bcr=5.0, 
        bcr_cut=0.5, 
        reg=0.0, 
        reg_every=16,
        disc_grad_ckpt=False,
        dbg_unused=False, 
        dbg_nan=False,
    ):
        super(VideoVAETrainer, self).__init__()
        self.dbg_unused, self.dbg_nan = dbg_unused, dbg_nan
        if self.dbg_nan:
            print('[dbg_nan mode on]')
            nan.debug_nan_hook(vae)
            nan.debug_nan_hook(disc)
        
        self.vae, self.disc = vae, disc
        self.vae_opt, self.disc_opt = vae_opt, disc_opt
        self.vae_wo_ddp: VQVideoVAE = vae_wo_ddp  # after torch.compile
        self.disc_wo_ddp: DinoDisc = disc_wo_ddp  # after torch.compile
        self.vae_params: Tuple[nn.Parameter] = tuple(self.vae_wo_ddp.parameters())
