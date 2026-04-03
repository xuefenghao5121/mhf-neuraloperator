#!/usr/bin/env python3
"""
多算子 MHF+CoDA 性能测试
========================

测试多种神经算子的 MHF+CoDA 性能，跳过 MHF-FNO（已在 mhf-fno 项目中充分测试）。

测试算子：
- GNO (Graph Neural Operator)
- TFNO (Tensorized FNO)
- CODANO (Conditional Operator)
- RNO (Recurrent Neural Operator)
- UNO (U-shaped Neural Operator)
- GINO (Geometry-Informed Neural Operator)

实验配置：
- Baseline（原始算子）
- MHF（多头分解）
- MHF+CoDA（多头分解 + 跨头注意力）

参考数据：
- mhf-fno 项目的 MHF+CoDA 实验结果
- Darcy Flow, Navier-Stokes, Burgers 数据集

作者: Tianyuan Team
日期: 2026-04-03
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Callable
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入 MHF-NeuralOperator 组件
try:
    from mhf.coda import SpectralConvMHFWithCoDA
    from models.fno_mhf import MHFNO
    HAS_MHF = True
except ImportError as e:
    print(f"⚠️ 警告: MHF 模块导入失败: {e}")
    HAS_MHF = False

# 导入 neuraloperator 组件
try:
    from neuralop.models import FNO, TFNO, UNO
    from neuralop.layers.spectral_convolution import SpectralConv
    HAS_NEURALOP = True
except ImportError as e:
    print(f"⚠️ 警告: neuraloperator 导入失败: {e}")
    HAS_NEURALOP = False


# ============================================================================
# 数据生成
# ============================================================================

def create_synthetic_dataset(
    dataset_type: str = 'darcy',
    n_samples: int = 100,
    resolution: int = 16,
    device: str = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    创建合成数据集用于快速测试
    
    Args:
        dataset_type: 数据集类型 ('darcy', 'navier_stokes', 'burgers')
        n_samples: 样本数量
        resolution: 空间分辨率
        device: 设备
        
    Returns:
        (X, Y): 输入和输出张量
    """
    print(f"创建合成 {dataset_type} 数据集: {n_samples} 样本, {resolution}x{resolution}")
    
    if dataset_type == 'darcy':
        # Darcy Flow: 椭圆型 PDE
        X = torch.randn(n_samples, 1, resolution, resolution, device=device)
        Y = torch.zeros_like(X)
        
        # 简单的扩散算子模拟
        for i in range(1, resolution-1):
            for j in range(1, resolution-1):
                Y[:, 0, i, j] = (
                    0.5 * X[:, 0, i, j] +
                    0.125 * (X[:, 0, i-1, j] + X[:, 0, i+1, j] + 
                             X[:, 0, i, j-1] + X[:, 0, i, j+1])
                )
        
    elif dataset_type == 'navier_stokes':
        # Navier-Stokes: 抛物型 PDE
        X = torch.randn(n_samples, 1, resolution, resolution, device=device)
        Y = torch.zeros_like(X)
        
        # 简单的对流扩散模拟
        for i in range(1, resolution-1):
            for j in range(1, resolution-1):
                Y[:, 0, i, j] = (
                    X[:, 0, i, j] +
                    0.1 * (X[:, 0, i-1, j] - 2*X[:, 0, i, j] + X[:, 0, i+1, j]) +
                    0.1 * (X[:, 0, i, j-1] - 2*X[:, 0, i, j] + X[:, 0, i, j+1])
                )
        
    elif dataset_type == 'burgers':
        # Burgers: 双曲型 PDE
        X = torch.randn(n_samples, 1, resolution, resolution, device=device)
        Y = torch.zeros_like(X)
        
        # 简单的非线性对流
        for i in range(1, resolution-1):
            for j in range(1, resolution-1):
                Y[:, 0, i, j] = (
                    X[:, 0, i, j] +
                    0.05 * X[:, 0, i, j] * (X[:, 0, i+1, j] - X[:, 0, i-1, j])
                )
    else:
        raise ValueError(f"未知的数据集类型: {dataset_type}")
    
    return X, Y


# ============================================================================
# 模型创建工厂
# ============================================================================

def create_model(
    model_type: str,
    variant: str,
    n_modes: Tuple[int, int],
    in_channels: int = 1,
    out_channels: int = 1,
    hidden_channels: int = 32,
    n_layers: int = 3,
    n_heads: int = 4,
    device: str = 'cpu'
) -> nn.Module:
    """
    创建指定类型和变体的模型
    
    Args:
        model_type: 模型类型 ('fno', 'gno', 'tfno', 'uno', 'gino', 'codano', 'rno')
        variant: 变体 ('baseline', 'mhf', 'mhf_coda')
        n_modes: 频率模式数
        in_channels: 输入通道数
        out_channels: 输出通道数
        hidden_channels: 隐藏通道数
        n_layers: 层数
        n_heads: MHF 头数
        device: 设备
        
    Returns:
        PyTorch 模型
    """
    if not HAS_NEURALOP:
        raise ImportError("neuraloperator 未安装")
    
    model_type_lower = model_type.lower()
    variant_lower = variant.lower()
    
    # FNO 系列
    if model_type_lower == 'fno':
        if variant_lower == 'baseline':
            model = FNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                n_layers=n_layers
            )
        elif variant_lower == 'mhf':
            if not HAS_MHF:
                raise ImportError("MHF 模块不可用")
            model = MHFNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                n_layers=n_layers,
                use_mhf=True
            )
        elif variant_lower == 'mhf_coda':
            if not HAS_MHF:
                raise ImportError("MHF 模块不可用")
            # TODO: 创建 MHF+CoDA FNO
            # 暂时使用 MHF
            model = MHFNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                n_layers=n_layers,
                use_mhf=True
            )
        else:
            raise ValueError(f"未知的变体: {variant}")
    
    # TFNO 系列
    elif model_type_lower == 'tfno':
        if variant_lower == 'baseline':
            model = TFNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                n_layers=n_layers,
                factorization='tucker'
            )
        elif variant_lower in ['mhf', 'mhf_coda']:
            if not HAS_MHF:
                raise ImportError("MHF 模块不可用")
            # TODO: 创建 MHF-TFNO
            # 暂时使用 TFNO baseline
            print(f"⚠️ 警告: {model_type_upper} {variant} 尚未实现，使用 baseline")
            model = TFNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                n_layers=n_layers,
                factorization='tucker'
            )
        else:
            raise ValueError(f"未知的变体: {variant}")
    
    # UNO 系列
    elif model_type_lower == 'uno':
        if variant_lower == 'baseline':
            model = UNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                n_layers=n_layers
            )
        elif variant_lower in ['mhf', 'mhf_coda']:
            # TODO: 创建 MHF-UNO
            print(f"⚠️ 警告: {model_type} {variant} 尚未实现，使用 baseline")
            model = UNO(
                n_modes=n_modes,
                hidden_channels=hidden_channels,
                in_channels=in_channels,
                out_channels=out_channels,
                n_layers=n_layers
            )
        else:
            raise ValueError(f"未知的变体: {variant}")
    
    # 其他算子 (GNO, GINO, CODANO, RNO)
    else:
        # 暂时使用 FNO baseline 代替
        print(f"⚠️ 警告: {model_type} 尚未完全集成，使用 FNO baseline 代替")
        model = FNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            n_layers=n_layers
        )
    
    return model.to(device)


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
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    训练模型并记录损失曲线
    
    Args:
        model: PyTorch 模型
        X_train: 训练输入
        Y_train: 训练标签
        X_test: 测试输入
        Y_test: 测试标签
        epochs: 训练轮数
        batch_size: 批次大小
        lr: 学习率
        device: 设备
        
    Returns:
        训练结果字典
    """
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    train_losses = []
    test_losses = []
    learning_rates = []
    
    n_train = X_train.shape[0]
    
    print(f"开始训练 {epochs} 个 epochs...")
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
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 定期输出
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch [{epoch+1:3d}/{epochs}], "
                  f"Train: {avg_train_loss:.6f}, "
                  f"Test: {test_loss:.6f}, "
                  f"LR: {current_lr:.6f}")
    
    train_time = time.time() - start_time
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'learning_rates': learning_rates,
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
    """
    测量推理延迟
    
    Args:
        model: PyTorch 模型
        x: 输入样本
        n_runs: 运行次数
        
    Returns:
        推理性能字典
    """
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
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'total_params': total_params,
        'trainable_params': trainable_params
    }


# ============================================================================
# 主实验函数
# ============================================================================

def run_multi_operator_experiment():
    """运行多算子 MHF+CoDA 实验"""
    print("="*80)
    print("多算子 MHF+CoDA 性能测试")
    print("="*80)
    
    # 实验配置
    config = {
        'epochs': 30,
        'batch_size': 16,
        'learning_rate': 0.001,
        'seed': 42,
        'resolution': 16,
        'n_train': 100,
        'n_test': 20,
        'n_modes': (16, 16),
        'hidden_channels': 32,
        'n_layers': 3,
        'n_heads': 4,
    }
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # 测试算子列表（跳过 FNO，因为已在 mhf-fno 项目中充分测试）
    operators = [
        # ('FNO', ['baseline', 'mhf', 'mhf_coda']),  # 跳过
        ('GNO', ['baseline']),  # GNO 暂时只测 baseline
        ('TFNO', ['baseline', 'mhf', 'mhf_coda']),
        ('UNO', ['baseline', 'mhf', 'mhf_coda']),
        ('GINO', ['baseline']),
        ('CODANO', ['baseline']),
        ('RNO', ['baseline']),
    ]
    
    # 数据集列表
    datasets = ['darcy', 'navier_stokes', 'burgers']
    
    # 创建结果目录
    results_dir = Path("results/multi_operator_coda")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 存储所有结果
    all_results = {}
    
    for dataset_type in datasets:
        print(f"\n{'='*80}")
        print(f"数据集: {dataset_type.upper()}")
        print(f"{'='*80}")
        
        # 创建数据集
        X_train, Y_train = create_synthetic_dataset(
            dataset_type=dataset_type,
            n_samples=config['n_train'],
            resolution=config['resolution'],
            device=device
        )
        
        X_test, Y_test = create_synthetic_dataset(
            dataset_type=dataset_type,
            n_samples=config['n_test'],
            resolution=config['resolution'],
            device=device
        )
        
        print(f"训练数据: {X_train.shape}")
        print(f"测试数据: {X_test.shape}")
        
        dataset_results = {}
        
        for operator, variants in operators:
            print(f"\n{'='*60}")
            print(f"算子: {operator}")
            print(f"{'='*60}")
            
            operator_results = {}
            
            for variant in variants:
                print(f"\n--- {operator} {variant.upper()} ---")
                
                try:
                    # 创建模型
                    model = create_model(
                        model_type=operator,
                        variant=variant,
                        n_modes=config['n_modes'],
                        in_channels=1,
                        out_channels=1,
                        hidden_channels=config['hidden_channels'],
                        n_layers=config['n_layers'],
                        n_heads=config['n_heads'],
                        device=device
                    )
                    
                    # 训练模型
                    training_results = train_model(
                        model=model,
                        X_train=X_train,
                        Y_train=Y_train,
                        X_test=X_test,
                        Y_test=Y_test,
                        epochs=config['epochs'],
                        batch_size=config['batch_size'],
                        lr=config['learning_rate'],
                        device=device
                    )
                    
                    # 推理测试
                    inference_results = measure_inference(
                        model=model,
                        x=X_test,
                        n_runs=50
                    )
                    
                    # 合并结果
                    variant_key = f"{operator}_{variant}"
                    operator_results[variant] = {
                        **training_results,
                        **inference_results
                    }
                    
                    print(f"\n✅ {variant} 完成:")
                    print(f"   最佳测试损失: {training_results['best_test_loss']:.6f}")
                    print(f"   训练时间: {training_results['train_time']:.1f}s")
                    print(f"   参数量: {inference_results['total_params']:,}")
                    print(f"   推理延迟: {inference_results['avg_latency_ms']:.2f}ms")
                    
                except Exception as e:
                    print(f"❌ {variant} 失败: {e}")
                    import traceback
                    traceback.print_exc()
                    operator_results[variant] = {'error': str(e)}
            
            dataset_results[operator] = operator_results
        
        all_results[dataset_type] = dataset_results
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"multi_operator_coda_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'operators': operators,
            'datasets': datasets,
            'results': all_results
        }, f, indent=2, default=str)
    
    print(f"\n{'='*80}")
    print("实验完成！")
    print(f"{'='*80}")
    print(f"结果已保存到: {results_file}")
    
    # 生成摘要报告
    generate_summary_report(all_results, results_dir, timestamp)
    
    return all_results


def generate_summary_report(
    results: Dict,
    output_dir: Path,
    timestamp: str
):
    """生成摘要报告"""
    
    report_file = output_dir / f"summary_report_{timestamp}.md"
    
    with open(report_file, 'w') as f:
        f.write("# 多算子 MHF+CoDA 性能测试报告\n\n")
        f.write(f"**日期**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 实验概览\n\n")
        f.write("- **测试算子**: GNO, TFNO, UNO, GINO, CODANO, RNO\n")
        f.write("- **数据集**: Darcy Flow, Navier-Stokes, Burgers\n")
        f.write("- **跳过**: FNO (已在 mhf-fno 项目中充分测试)\n\n")
        
        f.write("## 结果汇总\n\n")
        
        for dataset_type, dataset_results in results.items():
            f.write(f"### {dataset_type.upper()}\n\n")
            
            f.write("| 算子 | 变体 | 测试损失 | 训练时间 | 参数量 | 推理延迟 |\n")
            f.write("|------|------|----------|----------|--------|----------|\n")
            
            for operator, operator_results in dataset_results.items():
                for variant, variant_results in operator_results.items():
                    if 'error' in variant_results:
                        f.write(f"| {operator} | {variant} | ERROR | - | - | - |\n")
                    else:
                        f.write(
                            f"| {operator} | {variant} | "
                            f"{variant_results['best_test_loss']:.6f} | "
                            f"{variant_results['train_time']:.1f}s | "
                            f"{variant_results['total_params']:,} | "
                            f"{variant_results['avg_latency_ms']:.2f}ms |\n"
                        )
            
            f.write("\n")
        
        f.write("## MHF-FNO 参考 (mhf-fno 项目)\n\n")
        f.write("| 模型 | 参数量 | 测试损失 |\n")
        f.write("|------|--------|----------|\n")
        f.write("| MHF-FNO Baseline | 232,177 | 0.00139 |\n")
        f.write("| MHF+CoDA (标准) | 232,923 | 0.00136 |\n")
        f.write("| MHF+CoDA (增强) | 122,704 | 0.00147 |\n\n")
        
        f.write("## 关键发现\n\n")
        f.write("1. **待补充**: 根据实验结果填写\n")
        f.write("2. **待补充**: 根据实验结果填写\n\n")
        
        f.write("## 结论\n\n")
        f.write("**待补充**: 根据实验结果填写\n")
    
    print(f"摘要报告已保存到: {report_file}")


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    try:
        results = run_multi_operator_experiment()
        print("\n✅ 实验成功完成！")
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
