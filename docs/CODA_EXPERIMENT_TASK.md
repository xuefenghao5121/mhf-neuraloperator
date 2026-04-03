# 多算子 MHF+CoDA 真实效果测试（简化版）

## 任务背景

天渊团队项目：验证 CoDA (Cross-Head Attention) 在多种神经算子上的有效性。

### 参考项目
- **mhf-fno**: `/home/huawei/.openclaw/workspace/tianyuan-mhf-fno`
- **CoDA 实现核心**: `mhf_fno/mhf_attention.py`
- **已有实验结果**: `results/coda_performance/`

### 项目路径
- **MHF-NeuralOperator**: `/home/huawei/Desktop/home/xuefenghao/workspace/MHF-NeuralOperator/`
- **GitHub**: https://github.com/xuefenghao5121/mhf-neuraloperator

## 任务目标

### 简化版任务（今天完成）
1. **跳过 MHF-FNO 测试**（已在 mhf-fno 项目中充分测试）
2. **重点测试 2-3 种其他神经算子**：
   - ✅ GINO (Geometry-Informed Neural Operator)
   - ✅ TFNO (Tensorized Fourier Neural Operator)
3. **验证 CoDA 在不同算子上的有效性**

### 测试对比
- **Baseline**（原始算子）
- **MHF**（多头分解）
- **MHF+CoDA**（多头分解 + 跨头注意力）

### 评估指标
- 参数量
- L2 Loss（测试损失）
- 训练时间
- 推理延迟

## CoDA 原理

### 核心思想
CoDA (Cross-Head Attention) 解决 MHF 的头独立性假设问题：

- **当前 MHF**: 头之间完全独立，无法跨频率交互
- **问题**: NS 等复杂 PDE 需要全频率耦合
- **解决方案**: 添加跨头注意力机制，允许不同频率头之间的信息交换

### 设计原则
- **参数高效**: 注意力模块参数量应远小于 MHF 卷积
- **计算高效**: 避免 O(N²）的全局注意力
- **保持 MHF 优势**: 频率分离 + 头间交互

### CoDA 架构
```
输入 → MHF 多头分解 → IFFT → CoDA 注意力 → 合并 → 输出
                    ↓               ↓
              [Head 1]      [交互增强]
              [Head 2]      [交互增强]
              [Head 3]      [交互增强]
              [Head 4]      [交互增强]
```

### 参数效率
- MHF 卷积: ~n_heads × (C/n_heads)² × modes
- CoDA 注意力: ~3 × (C/n_heads)² + FFN
- CoDA 参数占 MHF 参数的 <1%

## 实现文件

### 新增文件
1. **`models/gino_mhf_coda.py`**
   - `SpectralConvMHFWithCoDA`: 带 CoDA 的频谱卷积层
   - `MHF_GINO_CoDA`: MHF+CoDA 优化的 GINO
   - `MHFFNOGNO_CoDA`: MHF+CoDA 优化的 FNOGNO

2. **`models/tfno_mhf_coda.py`**
   - `MHF_TFNO_CoDA`: MHF+CoDA 优化的 TFNO
   - `MHF_TFNO_Baseline`: TFNO Baseline

3. **`experiments/test_coda_module.py`**
   - CoDA 模块单元测试
   - 验证前向传播和梯度计算

4. **`experiments/run_coda_simplified.py`**
   - 简化版实验脚本
   - 测试 GINO 和 TFNO 的 Baseline/MHF/MHF+CoDA

### 修改文件
- **`models/__init__.py`**: 导出新的 CoDA 模型类

## 实验步骤

### 第一步：运行 CoDA 模块测试
```bash
cd /home/huawei/Desktop/home/xuefenghao/workspace/MHF-NeuralOperator
python experiments/test_coda_module.py
```

### 第二步：运行简化版实验
```bash
python experiments/run_coda_simplified.py
```

### 第三步：查看结果
结果保存在 `results/coda_simplified/` 目录：
- `coda_simplified_results_*.json`: 实验结果（JSON 格式）
- `summary_report_*.md`: 摘要报告（Markdown 格式）

## 预期结果

### mhf-fno 项目参考数据（MHF-FNO）
| 模型 | 参数量 | 测试损失 |
|------|--------|----------|
| MHF-FNO Baseline | 232,177 | 0.00139 |
| MHF+CoDA (标准) | 232,923 | 0.00136 |
| MHF+CoDA (增强) | 122,704 | 0.00147 |

### 本次实验预期
1. **参数量**: MHF+CoDA 略高于 MHF（<1% 增加）
2. **测试损失**: MHF+CoDA 应优于或等于 MHF
3. **训练时间**: MHF+CoDA 略高于 MHF（CoDA 计算开销）
4. **推理延迟**: MHF+CoDA 略高于 MHF（可接受）

## 进度跟踪

- [x] 研究 mhf-fno 的 CoDA 实现
- [x] 提取通用 CoDA 模块
- [x] 实现 GINO MHF+CoDA
- [x] 实现 TFNO MHF+CoDA
- [ ] 测试 CoDA 模块
- [ ] 运行简化版实验
- [ ] 分析实验结果
- [ ] 提交代码到 GitHub

## 注意事项

1. **通道数要求**: 使用 CoDA 时，通道数必须能被 n_heads 整除
2. **计算开销**: CoDA 会增加少量计算开销（<5%）
3. **内存开销**: CoDA 需要存储注意力权重（内存增加 <1%）
4. **回退机制**: 如果 CoDA 创建失败，自动回退到 Baseline

## 下一步计划

1. 运行实验并分析结果
2. 如果效果显著，扩展到更多算子（UNO, CODANO, RNO）
3. 撰写论文或技术报告
4. 提交代码到 GitHub 并发布

## 参考资料

- [TransFourier: Multi-Head Attention in Spectral Domain](https://arxiv.org/abs/2401.06014)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [Neural Operator Library](https://github.com/neuraloperator/neuraloperator)

---

**作者**: Tianyuan Team
**日期**: 2026-04-03
**版本**: 1.0.0
