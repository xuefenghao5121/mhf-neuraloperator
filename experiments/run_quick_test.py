#!/usr/bin/env python3
"""
快速 CoDA 实验测试（无复杂导入）
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*80)
print("快速 CoDA 实验测试")
print("="*80)

# ============================================================================
# CoDA 模块定义（直接定义避免导入问题）
# ============================================================================

class CrossHeadAttention(nn.Module):
    """轻轻量级跨头注意力模块"""
    
    def __init__(
        self,
        n_heads: int,
        channels_per_head: int,
        reduction: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.n_heads = n_heads
        self.channels_per_head = channels_per_head
        self.embed_dim = channels_per_head
        hidden_dim = max(channels_per_head // reduction, 4)
        
        # Q, K, V 投影
        self.query = nn.Linear(channels_per_head, channels_per_head, bias=False)
        self.key = nn.Linear(channels_per_head, channels_per_head, bias=False)
        self.value = nn.Linear(channels_per_head, channels_per_head, bias=False)
        
        # 缩放因子
        self.scale = channels_per_head ** -0.5
        
        # 输出投影 + FFN
        self.out_proj = nn.Linear(channels_per_head, channels_per_head)
        self.ffn = nn.Sequential(
            nn.Linear(channels_per_head, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, channels_per_head),
            nn.Dropout(dropout)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(channels_per_head)
        self.norm2 = nn.LayerNorm(channels_per_head)
        
        # 门控参数
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        B = x.shape[0]
        
        # 全局平均池化
        if x.dim() == 4:  # 1D: [B, n_heads, C, L]
            x_pooled = x.mean(dim=-1)
        else:  # 2D: [B, n_heads, C, H, W]
            x_pooled = x.mean(dim=(-2, -1))
        
        # 头间注意力
        residual = x_pooled
        x_norm = self.norm1(x_pooled)
        
        Q = self.query(x_norm)
        K = self.key(x_norm)
        V = self.value(x_norm)
        
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        attn_out = torch.matmul(attn_weights, V)
        attn_out = self.out_proj(attn_out)
        
        x_attn = residual + attn_out
        
        # FFN
        residual = x_attn
        x_norm = self.norm2(x_attn)
        ffn_out = self.ffn(x_norm)
        x_out = residual + ffn_out
        
        # 门控融合
        gate = torch.sigmoid(self.gate)
        attention_weights = gate * (x_out - x_pooled) + x_pooled
        
        # 广播回空间维度
        if x.dim() == 4:  # 1D
            attention_weights = attention_weights.unsqueeze(-1)
        else:  # 2D
            attention_weights = attention_weights.unsqueeze(-1).unsqueeze(-1)
        
        out = x * (1 + attention_weights)
        
        return out


# ============================================================================
# 简单 FNO 模型（带/不带 CoDA）
# ============================================================================

class SimpleConvBlock(nn.Module):
    """简化的卷积块"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, int],
        use_coda: bool = False,
        n_heads: int = 4,
    ):
        super().__init__()
        
        self.use_coda = use_coda
        self.n_heads = n_heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 简化的频谱卷积（用标准卷积模拟）
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # CoDA 模块
        if use_coda:
            if out_channels % n_heads != 0:
                raise ValueError(f"out_channels ({out_channels}) must be divisible by n_heads ({n_heads})")
            
            self.coda = CrossHeadAttention(
                n_heads=n_heads,
                channels_per_head=out_channels // n_heads,
                reduction=4,
                dropout=0.0
            )
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 卷积
        x_out = self.conv(x)
        
        # CoDA
        if self.use_coda:
            B, C, H, W = x_out.shape
            x_heads = x_out.view(B, self.n_heads, C // self.n_heads, H, W)
            x_heads = self.coda(x_heads)
            x_out = x_heads.view(B, C, H, W)
        
        # 激活
        x_out = self.activation(x_out)
        
        return x_out


class SimpleFNO(nn.Module):
    """简化的 FNO 模型"""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_layers: int = 3,
        use_coda: bool = False,
        n_heads: int = 4,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.use_coda = use_coda
        
        # Lifting
        self.lifting = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        
        # FNO blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layers):
            self.blocks.append(
                SimpleConvBlock(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    n_modes=(16, 16),
                    use_coda=use_coda,
                    n_heads=n_heads,
                )
            )
        
        # Projection
        self.projection = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lifting
        x = self.lifting(x)
        
        # FNO blocks
        for block in self.blocks:
            x = block(x)
        
        # Projection
        x = self.projection(x)
        
        return x


# ============================================================================
# 数据生成
# ============================================================================

def create_synthetic_darcy_data(
    n_samples: int = 100,
    resolution: int = 16,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """创建合成 Darcy Flow 数据"""
    X = torch.randn(n_samples, 1, resolution, resolution, device=device)
    Y = torch.zeros_like(X)
    
    # 扩散算子
    for i in range(1, resolution-1):
        for j in range(1, resolution-1):
            Y[:, 0, i, j] = (
                0.5 * X[:, 0, i, j] +
                0.125 * (X[:, 0, i-1, j] + X[:, 0, i+1, j] + 
                         X[:, 0, i, j-1] + X[:, 0, i, j+1])
            )
    
    return X, Y


# ============================================================================
# 训练和评估
# ============================================================================

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 0.001,
    device: str = 'cpu',
    model_name: str = 'Model'
) -> Dict[str, Any]:
    """训练模型"""
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    train_losses = []
    test_losses = []
    
    n_train = X_train.shape[0]
    
    print(f"  训练 {model_name} ({epochs} epochs)...")
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        
        # 随机打乱
        perm = torch.randperm(n_train, device=device)
        epoch_train_loss = 0.0
        batch_count = 0
        
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i+batch_size]
            bx = X_train[batch_idx]
            by = Y_train[batch_idx]
            
            optimizer.zero_grad()
            y_pred = model(bx)
            loss = criterion(y_pred, by)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_train_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # 测试评估
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test), Y_test).item()
        test_losses.append(test_loss)
        
        # 学习率调度
        scheduler.step(test_loss)
        
        # 定期输出
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1:3d}/{epochs}], "
                  f"Train: {avg_train_loss:.6f}, "
                  f"Test: {test_loss:.6f}")
    
    train_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_time': train_time,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'best_test_loss': min(test_losses)
    }


def measure_inference(
    model: nn.Module,
    x: torch.Tensor,
    n_runs: int = 50
) -> Dict[str, float]:
    """测量推理延迟"""
    model.eval()
    
    # 预热
    with torch.no_grad():
        _ = model(x[:1])
    
    # 计时
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(x[:1])
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # ms
    
    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'avg_latency_ms': np.mean(latencies),
        'latency_std_ms': np.std(latencies),
        'total_params': total_params,
        'trainable_params': trainable_params
    }


# ============================================================================
# 主实验
# ============================================================================

def run_quick_test():
    """运行快速测试"""
    
    # 实验配置
    config = {
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.001,
        'seed': 42,
        'resolution': 16,
        'n_train': 100,
        'n_test': 20,
        'hidden_channels': 32,
        'n_heads': 4,
    }
    
    print(f"\n实验配置:")
    print(f"  - 数据集: Darcy Flow (合成)")
    print(f"  - 分辨率: {config['resolution']}x{config['resolution']}")
    print(f"  - 训练样本: {config['n_train']}")
    print(f"  - 测试样本: {config['n_test']}")
    print(f"  - 训练轮数: {config['epochs']}")
    print(f"  - 隐藏通道: {config['hidden_channels']}")
    print(f"  - MHF 头数: {config['n_heads']}")
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 创建数据集
    print(f"\n{'='*80}")
    print("创建数据集...")
    print(f"{'='*80}")
    
    X_train, Y_train = create_synthetic_darcy_data(
        n_samples=config['n_train'],
        resolution=config['resolution'],
        device=device
    )
    
    X_test, Y_test = create_synthetic_darcy_data(
        n_samples=config['n_test'],
        resolution=config['resolution'],
        device=device
    )
    
    print(f"训练数据: {X_train.shape}")
    print(f"测试数据: {X_test.shape}")
    
    # 测试配置
    test_configs = [
        {
            'name': 'Baseline (无 CoDA)',
            'create_fn': lambda: SimpleFNO(
                in_channels=1,
                out_channels=1,
                hidden_channels=config['hidden_channels'],
                n_layers=3,
                use_coda=False,
            ),
            'variant': 'baseline',
        },
        {
            'name': 'With CoDA (4 heads)',
            'create_fn': lambda: SimpleFNO(
                in_channels=1,
                out_channels=1,
                hidden_channels=config['hidden_channels'],
                n_layers=3,
                use_coda=True,
                n_heads=4,
            ),
            'variant': 'coda_4heads',
        },
    ]
    
    # 存储结果
    all_results = {}
    
    for test_config in test_configs:
        print(f"\n{'='*80}")
        print(f"测试: {test_config['name']}")
        print(f"{'='*80}")
        
        try:
            # 创建模型
            print("  创建模型...")
            model = test_config['create_fn']()
            
            # 打印参数量
            total_params = sum(p.numel() for p in model.parameters())
            coda_params = 0
            if hasattr(model, 'use_coda') and model.use_coda:
                for block in model.blocks:
                    if hasattr(block, 'coda'):
                        coda_params += sum(p.numel() for p in block.coda.parameters())
            
            print(f"  参数量: {total_params:,}")
            if coda_params > 0:
                print(f"  CoDA 参数量: {coda_params:,} ({coda_params/total_params*100:.2f}%)")
            
            # 训练
            training_results = train_model(
                model=model,
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                lr=config['learning_rate'],
                device=device,
                model_name=test_config['name']
            )
            
            # 推理测试
            print("  测量推理性能...")
            inference_results = measure_inference(
                model=model,
                x=X_test,
                n_runs=50
            )
            
            # 合并结果
            results = {
                **training_results,
                **inference_results,
                'name': test_config['name'],
                'variant': test_config['variant'],
                'coda_params': coda_params,
            }
            
            all_results[test_config['name']] = results
            
            print(f"\n  ✅ {test_config['name']} 完成:")
            print(f"     - 最佳测试损失: {training_results['best_test_loss']:.6f}")
            print(f"     - 训练时间: {training_results['train_time']:.1f}s")
            print(f"     - 参数量: {inference_results['total_params']:,}")
            print(f"     - 推理延迟: {inference_results['avg_latency_ms']:.2f}ms")
            
        except Exception as e:
            print(f"\n  ❌ {test_config['name']} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[test_config['name']] = {
                'error': str(e),
                'name': test_config['name'],
                'variant': test_config['variant'],
            }
    
    # 保存结果
    results_dir = Path("results/quick_test")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"quick_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'device': device,
            'test_configs': test_configs,
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("实验完成！")
    print(f"{'='*80}")
    print(f"结果已保存到: {results_file}")
    
    # 生成摘要
    print(f"\n{'='*80}")
    print("实验摘要")
    print(f"{'='*80}")
    
    for name, result in all_results.items():
        if 'error' in result:
            print(f"\n{name}: ❌ ERROR - {result['error']}")
        else:
            print(f"\n{name}:")
            print(f"  最佳测试损失: {result['best_test_loss']:.6f}")
            print(f"  训练时间: {result['train_time']:.1f}s")
            print(f"  参数量: {result['total_params']:,}")
            print(f"  推理延迟: {result['avg_latency_ms']:.2f}ms")
            if 'coda_params' in result:
                print(f"  CoDA 参数量: {result['coda_params']:,}")
    
    print(f"\n{'='*80}")
    print("✅ 实验成功完成！")
    print(f"{'='*80}")
    
    return all_results


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    try:
        results = run_quick_test()
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
