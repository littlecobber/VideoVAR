import torch
import torch.nn as nn
from models.basic_videovae import CNNVideoEncoder, CNNVideoDecoder
from utils.datasets import DataLoader, DistributedSampler, DatasetFromCSV, get_transforms_video
import torch.optim as optim
from tqdm import tqdm
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from typing import Optional, Tuple
import logging


class VQVideoVAE(nn.Module):
    def __init__(
            self,
            grad_ckpt=True,  # 启用 Gradient Checkpointing
            ch=128,
            ch_mult=(1, 1, 2, 2, 4),
            dropout=0.0,
            vocab_size=4096,
            vocab_width=32
    ):
        super(VQVideoVAE, self).__init__()
        self.encoder = CNNVideoEncoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=2, dropout=dropout,
            img_channels=3, output_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )
        self.decoder = CNNVideoDecoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=3, dropout=dropout,
            input_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )

    def forward(self, x):
        z = checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        reconstructed = checkpoint.checkpoint(self.decoder, z, use_reentrant=False)
        return reconstructed


def train(vae, dataloader, optimizer, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)
    scaler = torch.amp.GradScaler('cuda')

    # for name, param in vae.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Parameter {name} does not require grad!")

    for epoch in range(num_epochs):
        vae.train()
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)

        for batch in progress_bar:
            video = batch['video'].to(device, dtype=torch.float16)  # FP16 训练
            optimizer.zero_grad()

            with torch.amp.autocast('cuda'):
                reconstructed = vae(video)
                # print(reconstructed.requires_grad)
                loss = vae_loss(reconstructed, video)
                # print(loss, loss.requires_grad)

            scaler.scale(loss).backward()  # 缩放 loss 进行反向传播
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 调整 scaler

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}')
        if (epoch + 1) % 100 == 0:  # 每 100 轮保存模型
            torch.save(vae.state_dict(), f'vae_epoch_{epoch+1}.pth')


def vae_loss(reconstructed, original):
    return nn.MSELoss()(reconstructed, original)  # 计算重建损失


def train_one_ep(args, model, train_loader, optimizer, scheduler, ep, device, metric_lg, tb_lg, warmup_disc_schedule, fade_blur_schedule, maybe_record_function):
    """训练一个epoch的视频VQVAE
    
    Args:
        args: 训练参数
        model: VQVAE模型
        train_loader: 训练数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        ep: 当前epoch
        device: 训练设备
        metric_lg: 指标记录器
        tb_lg: tensorboard记录器
        warmup_disc_schedule: 判别器预热调度
        fade_blur_schedule: 模糊效果调度
        maybe_record_function: 记录函数
    
    Returns:
        Tuple[float, float, float, float]: 平均总损失、重建损失、VQ损失和commit损失
    """
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_commit_loss = 0
    
    for it, (videos, _) in enumerate(train_loader):
        # 将视频数据移动到指定设备
        videos = videos.to(device, non_blocking=True)
        
        # 前向传播
        recon_videos, vq_loss, commit_loss = model(videos)
        
        # 计算重建损失
        recon_loss = F.mse_loss(recon_videos, videos)
        
        # 总损失
        loss = recon_loss + vq_loss + commit_loss
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
        
        # 记录损失
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_commit_loss += commit_loss.item()
        
        # 记录到metric logger
        if metric_lg is not None:
            metric_lg.log('train/loss', loss.item())
            metric_lg.log('train/recon_loss', recon_loss.item())
            metric_lg.log('train/vq_loss', vq_loss.item())
            metric_lg.log('train/commit_loss', commit_loss.item())
        
        # 记录到tensorboard
        if tb_lg is not None and it % args.log_interval == 0:
            tb_lg.log('train/loss', loss.item(), ep * len(train_loader) + it)
            tb_lg.log('train/recon_loss', recon_loss.item(), ep * len(train_loader) + it)
            tb_lg.log('train/vq_loss', vq_loss.item(), ep * len(train_loader) + it)
            tb_lg.log('train/commit_loss', commit_loss.item(), ep * len(train_loader) + it)
            
            # 记录视频样本
            if it % (args.log_interval * 10) == 0:
                with torch.no_grad():
                    # 选择第一个batch中的第一个视频
                    sample_video = videos[0].cpu()
                    sample_recon = recon_videos[0].cpu()
                    
                    # 将视频转换为适合可视化的格式
                    # 假设视频格式为 [T, C, H, W]
                    sample_video = sample_video.permute(1, 2, 3, 0)  # [C, H, W, T]
                    sample_recon = sample_recon.permute(1, 2, 3, 0)  # [C, H, W, T]
                    
                    # 记录原始视频和重建视频
                    tb_lg.log_video('train/original_video', sample_video, ep * len(train_loader) + it)
                    tb_lg.log_video('train/reconstructed_video', sample_recon, ep * len(train_loader) + it)
        
        # 打印训练进度
        if it % args.log_interval == 0:
            print(f'Epoch [{ep}/{args.epochs}], Iteration [{it}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, '
                  f'VQ Loss: {vq_loss.item():.4f}, Commit Loss: {commit_loss.item():.4f}')
    
    # 计算平均损失
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_vq_loss = total_vq_loss / len(train_loader)
    avg_commit_loss = total_commit_loss / len(train_loader)
    
    # 记录epoch级别的指标
    if metric_lg is not None:
        metric_lg.log('train/epoch_loss', avg_loss)
        metric_lg.log('train/epoch_recon_loss', avg_recon_loss)
        metric_lg.log('train/epoch_vq_loss', avg_vq_loss)
        metric_lg.log('train/epoch_commit_loss', avg_commit_loss)
    
    if tb_lg is not None:
        tb_lg.log('train/epoch_loss', avg_loss, ep)
        tb_lg.log('train/epoch_recon_loss', avg_recon_loss, ep)
        tb_lg.log('train/epoch_vq_loss', avg_vq_loss, ep)
        tb_lg.log('train/epoch_commit_loss', avg_commit_loss, ep)
    
    return avg_loss, avg_recon_loss, avg_vq_loss, avg_commit_loss


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")  # 启用高精度矩阵计算，加速 FP16 训练

    # data_path = r'D:\VideoVAR\OpenVid_part108\data\train\OpenVid-1M-108part.csv'
    # root = r'D:\VideoVAR\OpenVid_part108\video'

    data_path = r'/projectnb/ec720prj/DenseCap/vaex/OpenVid/OpenVid-1M-108part.csv'
    root = r'/projectnb/ec720prj/DenseCap/vaex/OpenVid/video'

    dataset = DatasetFromCSV(
        data_path,
        transform=get_transforms_video(),
        num_frames=16,
        frame_interval=3,
        root=root,
    )
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=1)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, sampler=sampler, num_workers=4, pin_memory=True, drop_last=True)

    vae = VQVideoVAE(grad_ckpt=True)  # 启用 Gradient Checkpointing
    optimizer = optim.AdamW(vae.parameters(), lr=1e-4)  # AdamW 更稳定

    train(vae=vae, dataloader=loader, optimizer=optimizer, num_epochs=1000)