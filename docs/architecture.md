# 架构设计文档

## 1. 设计目标

- **通用性**：支持neuraloperator所有算子类型的MHF优化
- **可扩展性**：易于添加新的算子类型
- **兼容性**：保持与原始neuraloperator API的兼容性
- **模块化**：清晰的模块分离，便于维护

## 2. 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Application                          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MHF Optimized Models                         │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐            │
│  │MHF-FNO│ │MHF-UNO│ │MHF-GINO│ │MHF-CODANO│ ...            │
│  └───┬───┘ └───┬───┘ └───┬───┘ └───┬─────┘            │
└───────────────┼───────────┼───────────┼─────────────────┘
                │           │           │
                ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     MHF Core Module                              │
│  ┌─────────────────────────┐  ┌─────────────────────────────┐   │
│  │  BaseMHFOpimizer        │  │  MHFMetadata                │   │
│  │  (抽象基类)             │  │  (存储分解信息)             │   │
│  └───────────┬─────────────┘  └───────────┬─────────────────┘   │
│              │                              │                   │
│  ┌───────────▼─────────────┐  ┌───────────▼─────────────────┐  │
│  │  MultiResolutionDecomp  │  │  HierarchicalFactorization  │  │
│  │  (多分辨率分解)         │  │  (层次化张量分解)           │  │
│  └─────────────────────────┘  └─────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Modified NeuralOperator Layers                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │SpectralConvMHF  │  │  FNOBlockMHF     │  │  GNOBlockMHF    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Original neuraloperator Base                   │
│               (继承并扩展原始类，保持兼容性)                     │
└─────────────────────────────────────────────────────────────────┘
```

## 3. 核心接口定义

### 3.1 BaseMHF 基类

所有MHF优化模块的抽象基类：

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tensorly.decomposition import CP, Tucker, TT

class BaseMHF(ABC, nn.Module):
    """MHF优化的抽象基类
    
    所有具体的MHF优化器都需要继承此类并实现抽象方法。
    """
    
    def __init__(
        self,
        ranks: Union[int, List[int]],
        resolutions: List[int],
        factorization: str = "tucker",
    ):
        super().__init__()
        self.ranks = ranks
        self.resolutions = resolutions
        self.factorization = factorization
        self._decomposed = False
        
    @abstractmethod
    def decompose(self) -> None:
        """执行MHF分解
        
        将原始权重进行多分辨率层次化分解
        """
        pass
    
    @abstractmethod
    def recompose(self) -> torch.Tensor:
        """重构完整权重
        
        从分解因子重构完整权重张量
        """
        pass
    
    @abstractmethod
    def forward_mhf(self, x: torch.Tensor) -> torch.Tensor:
        """使用MHF分解进行前向传播
        
        直接使用分解因子进行计算，不重构完整权重
        """
        pass
    
    @property
    def compression_ratio(self) -> float:
        """计算压缩比"""
        original_params = self._original_num_params
        decomposed_params = self._decomposed_num_params
        return decomposed_params / original_params
    
    def is_decomposed(self) -> bool:
        """返回是否已经执行分解"""
        return self._decomposed
```

### 3.2 MultiResolutionHierarchicalFactorization 核心分解类

```python
class MultiResolutionHierarchicalFactorization:
    """多分辨率层次化分解
    
    将权重张量在多个分辨率下进行层次分解：
    1. 对原始权重进行低分辨率近似
    2. 计算残差并在更高分辨率分解
    3. 递归直到达到原始分辨率
    """
    
    def __init__(
        self,
        resolutions: List[int],
        ranks: Union[int, Dict[str, int]],
        factorization_type: str = "tucker",
    ):
        self.resolutions = sorted(resolutions)
        self.ranks = ranks
        self.factorization_type = factorization_type
        self.factors: Dict[int, Any] = {}
        
    def decompose(
        self,
        weight: torch.Tensor,
        dims: Optional[List[int]] = None
    ) -> Dict[int, Any]:
        """执行多分辨率层次分解
        
        Parameters
        ----------
        weight : torch.Tensor
            原始权重张量
        dims : List[int], optional
            需要分解的维度，默认为空间维度
            
        Returns
        -------
        Dict[int, Any]
            每个分辨率对应的分解因子
        """
        pass
    
    def reconstruct(self) -> torch.Tensor:
        """从层次分解重构完整张量"""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """分层计算，直接使用分解因子进行张量收缩"""
        pass
```

### 3.3 SpectralConvMHF - MHF优化的谱卷积

继承neuraloperator的BaseSpectralConv，添加MHF优化：

```python
class SpectralConvMHF(BaseSpectralConv):
    """MHF优化的谱卷积层
    
    使用多分辨率层次分解压缩谱卷积权重
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: Tuple[int, ...],
        mhf_rank: Union[int, List[int]] = 8,
        mhf_resolutions: Optional[List[int]] = None,
        factorization: str = "tucker",
        implementation: str = "factorized",
        separable: bool = False,
        complex_data: bool = False,
    ):
        super().__init__(in_channels, out_channels, n_modes, ...)
        
        # MHF参数
        self.mhf_rank = mhf_rank
        self.mhf_resolutions = mhf_resolutions or self._auto_resolutions(n_modes)
        self.mhf_decomposition = MultiResolutionHierarchicalFactorization(
            resolutions=self.mhf_resolutions,
            ranks=mhf_rank,
            factorization_type=factorization
        )
        
        # 是否已经分解
        self._mhf_decomposed = False
        
    def decompose_weights(self) -> None:
        """对权重进行MHF分解"""
        if not self._mhf_decomposed:
            self.mhf_decomposition.decompose(self.weight)
            self._mhf_decomposed = True
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，如果MHF已分解则使用分解计算"""
        if self._mhf_decomposed:
            return self.mhf_decomposition.forward(x)
        else:
            return super().forward(x)
```

## 4. 分解策略

### 4.1 多分辨率层次分解算法

对于一个形状为 `(in_channels, out_channels, k_1, k_2, ..., k_d)` 的谱权重：

1. **初始分辨率** `r_0 < max(k_i)`:
   - 将权重下采样到分辨率 `r_0`
   - 进行张量分解得到 `W_0 = U_0 × V_0 × ...`

2. **计算残差**: `R_1 = W_{original} - up_sample(reconstruct(W_0))`

3. **下一分辨率** `r_1 > r_0`:
   - 对残差 `R_1` 在分辨率 `r_1` 进行分解
   - 得到 `W_1 = U_1 × V_1 × ...`

4. **递归**: 重复直到达到原始分辨率

### 4.2 分解顺序

```
分辨率: r_1 → r_2 → ... → r_k = original_resolution
分解:   W_0 → residual_1 → W_1 → residual_2 → ... → W_k
重构:   W = Σ (up_sample(W_i))
```

### 4.3 张量分解选择

| 分解方法 | 参数压缩率 | 计算速度 | 适用场景 |
|---------|-----------|---------|---------|
| CP | 高 | 快 | 低秩明显场景 |
| Tucker | 中 | 中 | 一般场景 |
| TT (Tensor Train) | 很高 | 较慢 | 高维场景 |

## 5. 各类算子适配方案

### 5.1 FNO系列 (FNO, TFNO, SFNO, UNO)

**核心**: 对SpectralConv中的权重进行MHF分解

```
原始: SpectralConv (dense weight)
MHF优化: SpectralConvMHF (MHF分解权重)
```

**修改位置**:
- `neuralop.layers.spectral_convolution.SpectralConv` → `SpectralConvMHF`
- `neuralop.layers.fno_block.FNOBlocks` → `FNOBlocksMHF`

### 5.2 GINO/GNO

**核心**: 对核积分部分的权重进行MHF分解

GINO使用GNO（Graph Neural Operator）处理不规则网格，核函数需要分解：

```
原始: Kernel integral with dense weight
MHF优化: MHF-GNOBlock 对核权重进行多分辨率分解
```

### 5.3 CODANO

**核心**: 条件编码部分的注意力机制权重分解

### 5.4 LocalNO

**核心**: 局部积分核的分解

### 5.5 RNO

**核心**: 循环块的投影矩阵分解

## 6. 兼容性设计

### 6.1 API兼容性

MHF优化的模型保持与原始neuraloperator模型完全相同的API：

```python
# 原始用法
from neuralop.models import FNO
model = FNO(n_modes=(16, 16), in_channels=1, out_channels=1, hidden_channels=64)

# MHF用法
from mhf.models import MHFNO
model = MHFNO(n_modes=(16, 16), in_channels=1, out_channels=1, hidden_channels=64,
              mhf_rank=8, mhf_resolutions=[4, 8, 16])
# 接口完全相同，使用方式一样
output = model(input)
```

### 6.2 权重兼容性

支持从原始权重加载后进行MHF分解：

```python
# 加载原始检查点
checkpoint = torch.load("original_fno.pth")
model = MHFNO(...)
model.load_original_weights(checkpoint)
model.decompose()  # 执行MHF分解
torch.save(model.state_dict(), "mhf_fno.pth")
```

## 7. 压缩策略默认值

根据不同分辨率自动选择MHF秩：

| 分辨率 | 默认MHF秩 | 预期压缩比 |
|-------|----------|-----------|
| 16×16 | 4-8 | ~5-10× |
| 32×32 | 8-16 | ~4-8× |
| 64×64 | 16-32 | ~3-6× |

## 8. 部署流程

1. **训练阶段**: 使用原始完整模型训练
2. **压缩阶段**: 加载训练好的权重，执行MHF分解
3. **部署阶段**: 直接使用分解后的模型进行推理，参数更少内存更小

支持训练后压缩，不需要重新训练。

## 9. 扩展新算子

添加新算子只需要三步：

1. 在 `mhf/models/` 创建新文件 `xxx_mhf.py`
2. 继承原始算子类，将其中的卷积层替换为MHF版本
3. 在 `mhf/models/__init__.py` 导出

详细示例见 [集成指南](./integration.md)
