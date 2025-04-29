import torch
import torch.nn as nn
from models.basic_videovae import CNNVideoEncoder, CNNVideoDecoder
from utils.datasets import DataLoader, DistributedSampler, DatasetFromCSV, get_transforms_video
import torch.optim as optim
from tqdm import tqdm
import torch.utils.checkpoint as checkpoint
from models.videoquant import VectorQuantizer3
import argparse
import os
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter  # 添加TensorBoard支持
import time  # 用于记录时间戳
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import torch.nn.functional as F


class VQVideoVAE(nn.Module):
    def __init__(
            self,
            grad_ckpt=True,  # Enable Gradient Checkpointing
            ch=128,
            ch_mult=(1, 1, 2, 2, 4),
            dropout=0.0,
            vocab_size=32768,
            vocab_width=512,
            vocab_norm=True,
            beta=0.25,
            quant_conv_k=3,
            quant_resi=-0.5
    ):
        super(VQVideoVAE, self).__init__()
        # video encoder
        self.encoder = CNNVideoEncoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=2, dropout=dropout,
            img_channels=3, output_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )
        
        # quantization component
        self.quant_conv = nn.Conv3d(vocab_width, vocab_width, kernel_size=quant_conv_k, padding=quant_conv_k//2)
        self.quantize = VectorQuantizer3(
            vocab_size=vocab_size,
            vocab_width=vocab_width,
            vocab_norm=vocab_norm,
            beta=beta,
            quant_resi=quant_resi
        )
        self.post_quant_conv = nn.Conv3d(vocab_width, vocab_width, kernel_size=quant_conv_k, padding=quant_conv_k//2)
        
        # video decoder
        self.decoder = CNNVideoDecoder(
            ch=ch, ch_mult=ch_mult, num_res_blocks=3, dropout=dropout,
            input_channels=vocab_width, using_sa=True, using_mid_sa=True,
            grad_ckpt=grad_ckpt,
        )
        
        # 保存词汇表大小
        self.vocab_size = vocab_size

    def forward(self, x):
        # encoder
        z = checkpoint.checkpoint(self.encoder, x, use_reentrant=False)
        
        # quantization
        z = self.quant_conv(z)
        z_q, loss_vq, vocab_usage = self.quantize(z)
        z_q = self.post_quant_conv(z_q)
        
        # decoder
        reconstructed = checkpoint.checkpoint(self.decoder, z_q, use_reentrant=False)
        
        return reconstructed, loss_vq, vocab_usage
    
    def get_codebook_usage(self):
        """获取codebook的使用情况"""
        # 从quantize模块获取使用情况
        if hasattr(self.quantize, 'get_codebook_usage'):
            return self.quantize.get_codebook_usage()
        else:
            # 如果没有直接的方法，尝试从vocab_usage属性获取
            if hasattr(self.quantize, 'vocab_usage'):
                return self.quantize.vocab_usage
            else:
                # 如果都没有，返回一个全零的向量
                return torch.zeros(self.vocab_size, device=next(self.parameters()).device)


def plot_codebook_usage(vocab_usage, epoch, writer):
    """绘制codebook使用率图并添加到TensorBoard"""
    # 如果vocab_usage是标量，直接记录到tensorboard
    if isinstance(vocab_usage, (float, np.float32)) or (isinstance(vocab_usage, torch.Tensor) and vocab_usage.dim() == 0):
        writer.add_scalar('Codebook/usage_percentage', float(vocab_usage), epoch)
        return {
            'used_codes': 1,  # 使用1代替0，因为标量表示至少使用了一个码字
            'total_codes': 1,  # 使用1代替0，因为标量表示至少使用了一个码字
            'usage_percentage': float(vocab_usage),
            'max_usage': float(vocab_usage),
            'min_usage': float(vocab_usage)
        }
    
    # 将tensor转换为numpy数组
    if isinstance(vocab_usage, torch.Tensor):
        vocab_usage = vocab_usage.cpu().numpy()
    
    # 计算使用率
    usage_percentage = vocab_usage / vocab_usage.sum() * 100
    
    # 创建图表
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # 绘制使用率条形图
    ax.bar(range(len(usage_percentage)), usage_percentage)
    ax.set_xlabel('Codebook Index')
    ax.set_ylabel('Usage Percentage (%)')
    ax.set_title(f'Codebook Usage at Epoch {epoch}')
    
    # 添加一些统计信息
    used_codes = np.sum(usage_percentage > 0)
    total_codes = len(usage_percentage)
    ax.text(0.02, 0.95, f'Used: {used_codes}/{total_codes} ({used_codes/total_codes*100:.2f}%)', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    # 将图表添加到TensorBoard
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = plt.imread(buf)
    writer.add_image('Codebook_Usage', img.transpose(2, 0, 1), epoch)
    
    # 关闭图表和缓冲区
    plt.close(fig)
    buf.close()
    
    # 返回使用率统计信息
    return {
        'used_codes': used_codes,
        'total_codes': total_codes,
        'usage_percentage': usage_percentage.mean(),
        'max_usage': usage_percentage.max(),
        'min_usage': usage_percentage.min()
    }


def vae_loss(reconstructed, original, loss_vq):
    """计算VQ-VAE的损失函数
    Args:
        reconstructed: 重建的视频
        original: 原始视频
        loss_vq: VQ损失（包含量化损失和commitment损失）

    Returns:
        total_loss: 总损失
        recon_loss: 重建损失
        vq_loss: VQ损失
    """
    # 重建损失
    recon_loss = mse_loss(reconstructed, original)
    
    # VQ损失（包含量化损失和commitment损失）
    vq_loss = loss_vq
    
    # 总损失
    total_loss = recon_loss + vq_loss 
    
    return total_loss, recon_loss, vq_loss


def train(vae, dataloader, optimizer, num_epochs=100, log_dir='runs', resume_path=None, start_epoch=0):
    """训练VQ-VAE模型
    Args:
        vae: VQ-VAE模型
        dataloader: 数据加载器
        optimizer: 优化器
        num_epochs: 训练轮数
        log_dir: TensorBoard日志目录
        resume_path: 恢复训练的检查点路径
        start_epoch: 开始训练的轮数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae.to(device)
    
    # 初始化混合精度训练
    scaler = torch.cuda.amp.GradScaler()
    
    # 创建TensorBoard写入器
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(f'{log_dir}/vae_{timestamp}')
    
    # 如果是从检查点恢复，加载模型状态
    if resume_path is not None:
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
    
    # 记录初始学习率
    writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], start_epoch)

    for epoch in range(start_epoch, num_epochs):
        vae.train()
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_vq_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False)
        
        # 用于累积codebook使用情况
        all_vocab_usage = None

        for batch_idx, batch in enumerate(progress_bar):
            video = batch['video'].to(device, dtype=torch.float16)  # FP16 训练
            optimizer.zero_grad()

            # 使用自动混合精度
            with torch.cuda.amp.autocast():
                # 获取编码器输出和量化后的潜在向量
                z = vae.encoder(video)
                z = vae.quant_conv(z)
                z_q, loss_vq, vocab_usage = vae.quantize(z)
                z_q = vae.post_quant_conv(z_q)
                
                # 解码器重建
                reconstructed = vae.decoder(z_q)
                
                # 计算损失
                total_loss, recon_loss, vq_loss = vae_loss(reconstructed, video, loss_vq)
                
                # 累积codebook使用情况
                if all_vocab_usage is None:
                    all_vocab_usage = vocab_usage
                else:
                    all_vocab_usage += vocab_usage
            
            # 使用scaler进行反向传播
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 记录损失（使用item()获取标量值）
            with torch.no_grad():
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_vq_loss += vq_loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'recon_loss': f'{recon_loss.item():.4f}',
                'vq_loss': f'{vq_loss.item():.4f}'
            })

        # 计算平均损失
        avg_loss = epoch_total_loss / len(dataloader)
        avg_recon_loss = epoch_recon_loss / len(dataloader)
        avg_vq_loss = epoch_vq_loss / len(dataloader)
        
        # 计算codebook使用率
        if all_vocab_usage is not None:
            codebook_usage = plot_codebook_usage(all_vocab_usage, epoch, writer)
            print(f'Codebook Usage: {codebook_usage["used_codes"]}/{codebook_usage["total_codes"]} ({codebook_usage["usage_percentage"]:.2f}%)')
        
        # 每个epoch记录一次损失
        writer.add_scalar('Loss/total', avg_loss, epoch)
        writer.add_scalar('Loss/reconstruction', avg_recon_loss, epoch)
        writer.add_scalar('Loss/vq', avg_vq_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 立即刷新writer
        writer.flush()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, VQ Loss: {avg_vq_loss:.4f}')
        
        # 每100轮保存一次检查点
        if (epoch + 1) % 100 == 0:
            checkpoint_path = os.path.join(log_dir, f'vae_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
    
    # 训练结束后关闭writer
    writer.flush()
    writer.close()
    print(f"TensorBoard logs saved to {log_dir}/vae_{timestamp}")


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/projectnb/ec720prj/DenseCap/vaex/OpenVid/OpenVid-1M-108part.csv')
    parser.add_argument('--video_root', type=str, default='/projectnb/ec720prj/DenseCap/vaex/OpenVid/video')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_interval', type=int, default=100)
    parser.add_argument('--grad_ckpt', action='store_true', help='Enable gradient checkpointing')
    parser.add_argument('--vocab_size', type=int, default=32768)
    parser.add_argument('--vocab_width', type=int, default=512)
    parser.add_argument('--vocab_norm', action='store_true', help='Normalize vocabulary')
    parser.add_argument('--vq_beta', type=float, default=0.25)
    parser.add_argument('--quant_conv_k', type=int, default=3)
    parser.add_argument('--quant_resi', type=float, default=-0.5)
    parser.add_argument('--log_dir', type=str, default='runs', help='Directory for TensorBoard logs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch to start training from')
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 设置随机种子
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 创建数据集和数据加载器
    dataset = DatasetFromCSV(
        args.data_path,
        transform=get_transforms_video(),
        num_frames=16,
        frame_interval=3,
        root=args.video_root,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 创建模型
    vae = VQVideoVAE(
        grad_ckpt=args.grad_ckpt,
        vocab_size=args.vocab_size,
        vocab_width=args.vocab_width,
        vocab_norm=args.vocab_norm,
        beta=args.vq_beta,
        quant_conv_k=args.quant_conv_k,
        quant_resi=args.quant_resi
    ).to(device)
    
    # 创建优化器和学习率调度器
    optimizer = AdamW(
        vae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr/10
    )
    
    mse_loss = nn.MSELoss()

    # 训练循环
    print('Starting training...')
    train(vae, dataloader, optimizer, args.num_epochs, args.log_dir, args.resume, args.start_epoch)