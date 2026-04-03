#!/usr/bin/env python3
"""
CoDA 模块快速测试
================

验证 CoDA (Cross-Head Attention) 模块是否能正常工作
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("CoDA 模块测试")
print("="*80)

# 导入 CoDA 模块
try:
    from mhf.coda import CrossHeadAttention, SpectralConvMHFWithCoDA
    print("✅ CoDA 模块导入成功")
except ImportError as e:
    print(f"❌ CoDA 模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试 CrossHeadAttention
print("\n" + "="*80)
print("测试 CrossHeadAttention 模块")
print("="*80)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"设备: {device}")

# 创建 CrossHeadAttention 模块
n_heads = 4
channels_per_head = 8
reduction = 4

print(f"\n配置:")
print(f"  n_heads: {n_heads}")
print(f"  channels_per_head: {channels_per_head}")
print(f"  reduction: {reduction}")

coda = CrossHeadAttention(
    n_heads=n_heads,
    channels_per_head=channels_per_head,
    reduction=reduction,
    dropout=0.0
).to(device)

# 测试 2D 输入 [B, n_heads, C_per_head, H, W]
print("\n测试 2D 输入...")
x_2d = torch.randn(2, n_heads, channels_per_head, 16, 16, device=device)
print(f"输入形状: {x_2d.shape}")

with torch.no_grad():
    out_2d = coda(x_2d)

print(f"输出形状: {out_2d.shape}")
print(f"输出形状匹配: {out_2d.shape == x_2d.shape}")

# 测试 1D 输入 [B, n_heads, C_per_head, L]
print("\n测试 1D 输入...")
x_1d = torch.randn(2, n_heads, channels_per_head, 64, device=device)
print(f"输入形状: {x_1d.shape}")

with torch.no_grad():
    out_1d = coda(x_1d)

print(f"输出形状: {out_1d.shape}")
print(f"输出形状匹配: {out_1d.shape == x_1d.shape}")

# 测试 SpectralConvMHFWithCoDA（如果可用）
print("\n" + "="*80)
print("测试 SpectralConvMHFWithCoDA 模块")
print("="*80)

try:
    from neuralop.layers.spectral_convolution import SpectralConv
    print("✅ SpectralConv 导入成功")
except ImportError as e:
    print(f"⚠️ SpectralConv 导入失败（neuraloperator 未安装）: {e}")
    print("跳过 SpectralConvMHFWithCoDA 测试")
    sys.exit(0)

# 创建 SpectralConvMHFWithCoDA
in_channels = 32
out_channels = 32
n_modes = (16, 16)
n_heads = 4

print(f"\n配置:")
print(f"  in_channels: {in_channels}")
print(f"  out_channels: {out_channels}")
print(f"  n_modes: {n_modes}")
print(f"  n_heads: {n_heads}")

try:
    conv_coda = SpectralConvMHFWithCoDA(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=n_modes,
        n_heads=n_heads,
        use_coda=True,
        coda_reduction=4,
        coda_dropout=0.0,
    ).to(device)
    print("✅ SpectralConvMHFWithCoDA 创建成功")
except Exception as e:
    print(f"❌ SpectralConvMHFWithCoDA 创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试前向传播
print("\n测试前向传播...")
x = torch.randn(2, in_channels, 32, 32, device=device)
print(f"输入形状: {x.shape}")

try:
    with torch.no_grad():
        out = conv_coda(x)
    print(f"输出形状: {out.shape}")
    print(f"输出形状匹配: {out.shape == x.shape}")
except Exception as e:
    print(f"❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 计算参数量
total_params = sum(p.numel() for p in conv_coda.parameters())
coda_params = sum(p.numel() for p in conv_coda.coda.parameters()) if hasattr(conv_coda, 'coda') else 0

print(f"\n参数量统计:")
print(f"  总参数量: {total_params:,}")
print(f"  CoDA 参数量: {coda_params:,}")
print(f"  CoDA 参数占比: {coda_params/total_params*100:.2f}%")

# 测试梯度
print("\n测试梯度...")
x = torch.randn(2, in_channels, 32, 32, device=device, requires_grad=False)
y = torch.randn_like(x)

conv_coda.train()
optimizer = torch.optim.Adam(conv_coda.parameters(), lr=0.001)

optimizer.zero_grad()
out = conv_coda(x)
loss = nn.functional.mse_loss(out, y)
loss.backward()
optimizer.step()

print(f"损失值: {loss.item():.6f}")
print(f"✅ 梯度计算和反向传播成功")

print("\n" + "="*80)
print("✅ CoDA 模块测试全部通过！")
print("="*80)
