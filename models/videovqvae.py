"""
References:
- VectorQuantizer: VectorQuantizer2 from https://github.com/CompVis/taming-transformers/blob/3ba01b241669f5ade541ce990f7650a3b8f65318/taming/modules/vqvae/quantize.py#L110
- VQVAE: VQModel from https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/models/autoencoder.py#L14
"""
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basic_videovae import CNNVideoDecoder, CNNVideoEncoder
from models.videoquant import VectorQuantizer3


def identity(x, inplace=False): return x


class VQVideoVAE(nn.Module):
    def __init__(
        self,
        # for all:
        grad_ckpt=False,            # whether to use gradient checkpointing
        
        # vitamin encoder:
        vitamin='',                 # 's', 'b', 'l' for using vitamin; 'cnn' or '' for using CNN
        drop_path_rate=0.1,
        
        # CNN encoder & CNN decoder:
        ch=128,                     # basic width of CNN encoder and CNN decoder
        ch_mult=(1, 1, 2, 2, 4),    # downsample_ratio would be 2 ** (len(ch_mult) - 1)
        dropout=0.0,                # dropout in CNN encoder and CNN decoder
        
        # quantizer:
        vocab_size=4096,
        vocab_width=32,
        vocab_norm=False,           # whether to limit the codebook vectors to have unit norm
        beta=0.25,                  # commitment loss weight
        quant_conv_k=3,             # quant conv kernel size
        quant_resi=-0.5,            #
    ):
        super().__init__()
        self.downsample_ratio = 2 ** (len(ch_mult) - 1)
        
        # 1. build encoder
        print(f'[VQVideoVAE] create CNN Video Encoder with {ch=}, {ch_mult=} {dropout=:g} ...', flush=True)
        self.encoder: CNNVideoEncoder = CNNVideoEncoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=2, dropout=dropout,
            img_channels=3, output_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )
        # 2. build conv before quant
        self.quant_conv = nn.Conv3d(vocab_width, vocab_width, quant_conv_k, stride=1, padding=quant_conv_k // 2)
        
        # 3. build quant
        print(f'[VQVideoVAE] create VectorQuantizer with {vocab_size=}, {vocab_width=} {vocab_norm=}, {beta=:g} ...', flush=True)
        self.quantize: VectorQuantizer3 = VectorQuantizer3(vocab_size=vocab_size, vocab_width=vocab_width, vocab_norm=vocab_norm, beta=beta, quant_resi=quant_resi)
        
        # 4. build conv after quant
        self.post_quant_conv = nn.Conv3d(vocab_width, vocab_width, quant_conv_k, stride=1, padding=quant_conv_k // 2)
        print(f'[VQVideoVAE] create CNN Video Decoder with {ch=}, {ch_mult=} {dropout=:g} ...', flush=True)
        
        # 5. build decoder
        self.decoder = CNNVideoDecoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=3, dropout=dropout,
            input_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )
        self.maybe_record_function = nullcontext
    
    def forward(self, video_B3THW, ret_usages=False):
        f_BCTHW = self.encoder(video_B3THW).float()
        with torch.cuda.amp.autocast(enabled=False):
            f_BCTHW, vq_loss, entropy_loss, usages = self.quantize(self.quant_conv(f_BCTHW), ret_usages=ret_usages)
            f_BCTHW = self.post_quant_conv(f_BCTHW)
        return self.decoder(f_BCTHW).float(), vq_loss, entropy_loss, usages
    
    def video_to_idx(self, video_B3THW: torch.Tensor) -> torch.LongTensor:
        f_BCTHW = self.encoder(video_B3THW)
        f_BCTHW = self.quant_conv(f_BCTHW)
        return self.quantize.f_to_idx(f_BCTHW)
    
    def idx_to_video(self, idx_BTHW: torch.Tensor) -> torch.Tensor:
        f_hat_BCTHW = self.quantize.quant_resi(self.quantize.embedding(idx_BTHW).permute(0, 4, 1, 2, 3))
        f_hat_BCTHW = self.post_quant_conv(f_hat_BCTHW)
        return self.decoder(f_hat_BCTHW).clamp_(-1, 1)
    
    def video_to_reconstructed_video(self, video_B3THW) -> torch.Tensor:
        return self.idx_to_video(self.video_to_idx(video_B3THW))
    
    def state_dict(self, *args, **kwargs):
        d = super().state_dict(*args, **kwargs)
        d['vocab_usage_record_times'] = self.quantize.vocab_usage_record_times
        return d
    
    def load_state_dict(self, state_dict: Dict[str, Any], strict=True, assign=False):
        if 'quantize.vocab_usage' not in state_dict or state_dict['quantize.vocab_usage'].shape[0] != self.quantize.vocab_usage.shape[0]:
            state_dict['quantize.vocab_usage'] = self.quantize.vocab_usage
        if 'vocab_usage_record_times' in state_dict:
            self.quantize.vocab_usage_record_times = state_dict.pop('vocab_usage_record_times')
        return super().load_state_dict(state_dict=state_dict, strict=strict, assign=assign)


if __name__ == '__main__':
    for clz in (nn.Linear, nn.LayerNorm, nn.BatchNorm2d, nn.SyncBatchNorm, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d):
        setattr(clz, 'reset_parameters', lambda self: None)
    
    torch.manual_seed(0)
    cnn = VQVideoVAE(ch=32, vocab_width=16, vocab_norm=False)
    print(str(cnn).replace('BnActConvBnActConv', 'ResnetBlock').replace('2x(', '('))
    from models import init_weights
    init_weights(cnn, -0.5)
    torch.save(cnn.state_dict(), 'cnn.pth')
