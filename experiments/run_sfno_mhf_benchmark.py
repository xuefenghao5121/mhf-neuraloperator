#!/usr/bin/env python3
"""
SFNO MHF Benchmark 测试

对比：
1. Baseline SFNO (原始)
2. MHF SFNO (我们的优化)

测试日期: 2026-04-07
"""

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Any

sys.path.insert(0, str(Path("/home/huawei/Desktop/home/xuefenghao/workspace/MHF-NeuralOperator")))

try:
    from neuralop.models import SFNO
    print("✅ 成功导入 Baseline SFNO")
except ImportError as e:
    print(f"❌ 导入 SFNO 失败: {e}")
    sys.exit(1)

try:
    from models.sfno_mhf import MHFSFNOBlocks
    print("✅ 成功导入 MHF SFNO")
except ImportError as e:
    print(f"❌ 导入 MHFSFNO 失败: {e}")
    sys.exit(1)

# ============================================================================
# 训练和评估函数
# ============================================================================

def train_model(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_test: torch.Tensor,
    Y_test: torch.Tensor,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 0.001,
    device: str = 'cpu',
    model_name: str = 'Model'
) -> Dict[str, Any]:
    """训练模型"""
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    n_train = X_train.shape[0]

    print(f"  训练 {model_name} ({epochs} epochs)...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        batch_count = 0

        for i in range(0, n_train, batch_size):
            bx = X_train[i:i+batch_size].to(device)
            by = Y_train[i:i+batch_size].to(device)

            optimizer.zero_grad()
            y_pred = model(bx)
            loss = criterion(y_pred, by)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            batch_count += 1

        avg_train_loss = epoch_train_loss / max(batch_count, 1)
        train_losses.append(avg_train_loss)

        # 测试评估
        model.eval()
        with torch.no_grad():
            test_loss = criterion(model(X_test.to(device)), Y_test.to(device)).item()
        test_losses.append(test_loss)

        if (epoch + 1) % 3 == 0 or epoch == 0:
            print(f"    Epoch [{epoch+1:2d}/{epochs}], Train: {avg_train_loss:.6f}, Test: {test_loss:.6f}")

    train_time = time.time() - start_time

    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_time': train_time,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'best_test_loss': min(test_losses),
    }


def measure_inference(
    model: nn.Module,
    x: torch.Tensor,
    n_runs: int = 50,
    device: str = 'cpu'
) -> Dict[str, float]:
    """测量推理延迟"""
    model.eval().to(device)

    # 预热
    with torch.no_grad():
        _ = model(x[:1].to(device))

    # 计时
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            start_time = time.perf_counter()
            _ = model(x[:1].to(device))
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)

    total_params = sum(p.numel() for p in model.parameters())
    model_size = total_params * 4 / (1024 ** 2)

    return {
        'avg_latency_ms': np.mean(latencies),
        'latency_std_ms': np.std(latencies),
        'total_params': total_params,
        'model_size_mb': model_size,
    }


# ============================================================================
# 主测试函数
# ============================================================================

PROJECT_ROOT = Path("/home/huawei/Desktop/home/xuefenghao/workspace/MHF-NeuralOperator")

def run_sfno_mhf_benchmark():
    """运行 SFNO MHF benchmark 测试"""

    # 配置
    config = {
        'epochs': 15,
        'batch_size': 32,
        'learning_rate': 0.001,
        'seed': 42,
        'grid_shape': (32, 32),  # Spherical grid shape
        'in_channels': 1,
        'out_channels': 1,
        'n_layers': 3,
        'max_degree': 16,  # SFNO 最大球谐度
        'n_train': 200,
        'n_test': 50,
        'latent_dim': 32,  # SFNO 隐藏维度
    }

    print(f"\n📋 测试配置:")
    print(f"  - 训练轮数: {config['epochs']}")
    print(f"  - 训练样本: {config['n_train']}")
    print(f"  - 测试样本: {config['n_test']}")
    print(f"  - 输入通道: {config['in_channels']}")
    print(f"  - 输出通道: {config['out_channels']}")
    print(f"  - 层数: {config['n_layers']}")
    print(f"  - 网格形状: {config['grid_shape']}")
    print(f"  - 最大球谐度: {config['max_degree']}")

    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  设备: {device}")

    # 设置随机种子
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # 生成合成数据 (模拟球面数据)
    print(f"\n📊 生成合成球面数据...")
    H, W = config['grid_shape']
    X_train = torch.randn(config['n_train'], config['in_channels'], H, W)
    Y_train = torch.randn(config['n_train'], config['out_channels'], H, W)
    X_test = torch.randn(config['n_test'], config['in_channels'], H, W)
    Y_test = torch.randn(config['n_test'], config['out_channels'], H, W)

    print(f"  训练集: X={X_train.shape}, Y={Y_train.shape}")
    print(f"  测试集: X={X_test.shape}, Y={Y_test.shape}")

    # 测试配置
    test_configs = [
        {
            'name': 'Baseline SFNO',
            'variant': 'baseline',
            'description': '原始 SFNO，不带 MHF',
        },
        {
            'name': 'MHF SFNO',
            'variant': 'mhf',
            'description': 'MHF 优化的 SFNO',
        },
    ]

    # 存储结果
    all_results = {}
    experiment_start = time.time()

    for test_config in test_configs:
        print(f"\n{'='*100}")
        print(f"🔬 测试: {test_config['name']}")
        print(f"   {test_config['description']}")
        print(f"{'='*100}")

        try:
            # 创建模型
            print("  创建模型...")
            start_create = time.time()

            if test_config['variant'] == 'baseline':
                # 创建 Baseline SFNO
                model = SFNO(
                    n_modes=(config['max_degree'], config['max_degree']),
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    hidden_channels=config['latent_dim'],
                    n_layers=config['n_layers'],
                )

            elif test_config['variant'] == 'mhf':
                # 创建 MHF SFNO
                model = MHFSFNOBlocks(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    grid_shape=config['grid_shape'],
                    n_layers=config['n_layers'],
                    max_degree=config['max_degree'],
                    mhf_rank=8,  # MHF rank
                    mhf_resolutions=None,  # 自动确定
                    factorization='tucker',
                    mhf_implementation='factorized',
                    grid_type='equiangular',
                    norm='ortho',
                    use_channel_mlp=False,  # 禁用 ChannelMLP 避免问题
                )

            create_time = time.time() - start_create
            print(f"  ✓ 模型创建完成 ({create_time:.2f}s)")

            # 统计参数量
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  参数量: {total_params:,}")

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
                n_runs=50,
                device=device
            )

            # 合并结果
            results = {
                **training_results,
                **inference_results,
                'name': test_config['name'],
                'variant': test_config['variant'],
                'description': test_config['description'],
            }

            all_results[test_config['name']] = results

            print(f"\n  ✅ {test_config['name']} 完成:")
            print(f"     - 最佳测试损失: {training_results['best_test_loss']:.6f}")
            print(f"     - 最终测试损失: {training_results['final_test_loss']:.6f}")
            print(f"     - 训练时间: {training_results['train_time']:.1f}s")
            print(f"     - 参数量: {inference_results['total_params']:,}")
            print(f"     - 推理延迟: {inference_results['avg_latency_ms']:.2f} ms")

        except Exception as e:
            print(f"\n  ❌ {test_config['name']} 失败: {e}")
            import traceback
            traceback.print_exc()
            all_results[test_config['name']] = {
                'error': str(e),
                'name': test_config['name'],
                'variant': test_config['variant'],
                'description': test_config['description'],
            }

    experiment_time = time.time() - experiment_start

    # 保存结果
    results_dir = PROJECT_ROOT / "results" / "sfno_mhf_benchmark"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"experiment_results_{timestamp}.json"

    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'device': device,
            'results': all_results,
            'experiment_time': experiment_time,
        }, f, indent=2, default=str)

    print(f"\n{'='*100}")
    print("实验完成！")
    print(f"{'='*100}")
    print(f"结果已保存到: {results_file}")

    # 生成报告
    generate_report(all_results, config, device, results_dir / "SFNO_MHF_BENCHMARK_REPORT.md")

    return all_results


def generate_report(all_results, config, device, output_file):
    """生成最终报告"""

    # 创建报告
    report_lines = []

    # 标题
    report_lines.append("# SFNO MHF Benchmark 测试报告\n")
    report_lines.append(f"**测试日期**: 2026-04-07\n")
    report_lines.append(f"**数据集**: 球面合成数据 {config['grid_shape']}\n")
    report_lines.append(f"**设备**: {device}\n")

    # 测试环境
    report_lines.append("\n---\n")
    report_lines.append("## 测试环境\n")
    report_lines.append(f"- 数据集: 球面合成数据 {config['grid_shape']}\n")
    report_lines.append(f"- 设备: {device}\n")
    report_lines.append(f"- 训练轮数: {config['epochs']}\n")
    report_lines.append(f"- 训练样本: {config['n_train']}\n")
    report_lines.append(f"- 测试样本: {config['n_test']}\n")
    report_lines.append(f"- SFNO 层数: {config['n_layers']}\n")
    report_lines.append(f"- 最大球谐度: {config['max_degree']}\n")

    # 测试结果
    report_lines.append("\n---\n")
    report_lines.append("## 测试结果\n")
    report_lines.append("| 模型 | 参数量 | 训练Loss | 测试Loss | 推理延迟(ms) | 总训练时间(s) |")
    report_lines.append("|------|--------|----------|----------|---------------|---------------|")

    baseline_result = None
    mhf_result = None

    for name, result in all_results.items():
        if 'error' in result:
            report_lines.append(f"| {name} | ERROR | - | - | - | - |")
            continue

        total_params = result.get('total_params', 0)
        train_loss = result.get('final_train_loss', 0)
        test_loss = result.get('final_test_loss', 0)
        train_time = result.get('train_time', 0)
        latency = result.get('avg_latency_ms', 0)

        report_lines.append(
            f"| {name} | {total_params:,} | {train_loss:.6f} | {test_loss:.6f} | "
            f"{latency:.2f} | {train_time:.1f} |"
        )

        # 保存引用
        if 'Baseline' in name:
            baseline_result = result
        elif 'MHF' in name:
            mhf_result = result

    # 结论
    report_lines.append("\n---\n")
    report_lines.append("## 结论\n")

    if mhf_result and baseline_result:
        mhf_latency = mhf_result['avg_latency_ms']
        baseline_latency = baseline_result['avg_latency_ms']
        latency_improvement = (baseline_latency - mhf_latency) / baseline_latency * 100

        mhf_loss = mhf_result['best_test_loss']
        baseline_loss = baseline_result['best_test_loss']

        mhf_params = mhf_result['total_params']
        baseline_params = baseline_result['total_params']
        param_reduction = (baseline_params - mhf_params) / baseline_params * 100

        report_lines.append(f"### 1. 优化是否有效？\n")
        if mhf_latency < baseline_latency:
            report_lines.append(f"✅ 是的！推理延迟降低了 **{latency_improvement:.2f}%** (从 {baseline_latency:.2f}ms 到 {mhf_latency:.2f}ms)\n")
        else:
            report_lines.append(f"⚠️ 推理延迟增加了 {-latency_improvement:.2f}%\n")

        report_lines.append(f"### 2. 精度是否保持？\n")
        if mhf_loss <= baseline_loss * 1.05:
            report_lines.append(f"✅ 精度保持良好 (测试 L2 Loss: {mhf_loss:.6f} vs {baseline_loss:.6f})\n")
        else:
            report_lines.append(f"⚠️ 精度有所下降 (测试 L2 Loss: {mhf_loss:.6f} vs {baseline_loss:.6f})\n")

        report_lines.append(f"### 3. 参数量对比\n")
        report_lines.append(f"- Baseline SFNO 参数量: {baseline_params:,}\n")
        report_lines.append(f"- MHF SFNO 参数量: {mhf_params:,}\n")
        report_lines.append(f"- 参数量变化: {param_reduction:+.2f}%\n")

        report_lines.append(f"### 4. 详细对比\n")
        report_lines.append(f"| 指标 | Baseline SFNO | MHF SFNO | 变化 |\n")
        report_lines.append(f"|------|---------------|----------|------|\n")
        report_lines.append(f"| 参数量 | {baseline_params:,} | {mhf_params:,} | {param_reduction:+.2f}% |\n")
        report_lines.append(f"| 推理延迟 | {baseline_latency:.2f}ms | {mhf_latency:.2f}ms | {latency_improvement:+.2f}% |\n")
        report_lines.append(f"| 测试 Loss | {baseline_loss:.6f} | {mhf_loss:.6f} | {(mhf_loss/baseline_loss-1)*100:+.2f}% |\n")

    # 写入报告
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report_lines)

    print(f"\n✅ 最终报告已生成: {output_file}")
    print("\n报告内容:\n")
    print(''.join(report_lines))


# ============================================================================
# 主程序
# ============================================================================

if __name__ == '__main__':
    try:
        results = run_sfno_mhf_benchmark()
        print(f"\n{'='*100}")
        print("✅ SFNO MHF Benchmark 测试成功完成！")
        print(f"{'='*100}")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
