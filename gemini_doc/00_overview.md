# Sparsebit 项目总览

## 1. 项目简介

Sparsebit 是一个深度学习模型压缩工具包，提供**量化 (Quantization)** 和**剪枝 (Pruning/Sparse)** 两大功能。它基于 PyTorch FX 框架实现，支持后训练量化 (PTQ) 和量化感知训练 (QAT)。

---

## 2. 核心概念（名词、动词、引擎、点火钥匙）

### 2.1 名词 (Nouns) - 核心数据结构

| 名词 | 含义 | 对应代码 |
|------|------|----------|
| **QuantModel** | 量化模型包装器 | `quantization/quant_model.py` |
| **SparseModel** | 剪枝模型包装器 | `sparse/sparse_model.py` |
| **QuantOpr** | 量化算子基类 | `quantization/modules/base.py` |
| **SparseOpr** | 剪枝算子基类 | `sparse/modules/base.py` |
| **Quantizer** | 量化器，执行实际量化操作 | `quantization/quantizers/base.py` |
| **Observer** | 观察器，收集统计信息 | `quantization/observers/base.py` |
| **Sparser** | 剪枝器，计算剪枝掩码 | `sparse/sparsers/base.py` |
| **QModule** | 量化模块（QuantOpr 的实例） | 各种 `modules/*.py` |
| **GraphModule** | FX 追踪后的图模块 | PyTorch FX |
| **Config** | 配置对象 (CfgNode) | `quantization/quant_config.py` |

### 2.2 动词 (Verbs) - 核心操作

| 动词 | 功能 | 对应方法 |
|------|------|----------|
| **trace** | 使用 FX 追踪模型结构 | `_trace()` |
| **convert** | 将普通模块转换为量化/剪枝模块 | `_convert2quantmodule()` |
| **build** | 构建 quantizer/sparser | `build_quantizer()`, `build_sparser()` |
| **calibrate** | 校准量化参数 | `calc_qparams()` |
| **quantize** | 执行量化操作 | `Quantizer.forward()` |
| **observe** | 收集张量统计信息 | `Observer.update()` |
| **simplify** | 简化计算图 | `simplify()` |
| **fuse** | 融合算子（如 conv+bn） | `fuse_operations()` |
| **export** | 导出 ONNX 模型 | `export_onnx()` |
| **prune** | 执行剪枝操作 | `calc_mask()` |

### 2.3 引擎 (Engines) - 核心处理流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      QuantModel 引擎流程                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  _trace  │───→│ simplify │───→│ _convert │───→│ _build_  │  │
│  │          │    │          │    │ 2quant   │    │ quantizer│  │
│  └──────────┘    └──────────┘    └──────────┘    └────┬─────┘  │
│                                                       │         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐        │         │
│  │ export_  │←───│  fuse_   │←───│calibrate │←───────┘         │
│  │  onnx    │    │operations│    │          │                  │
│  └──────────┘    └──────────┘    └──────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.4 点火钥匙 (Ignition Keys) - 用户入口点

| 钥匙 | 说明 | 代码示例 |
|------|------|----------|
| **parse_qconfig** | 解析量化配置 | `from sparsebit.quantization import parse_qconfig` |
| **QuantModel** | 创建量化模型 | `qmodel = QuantModel(model, config)` |
| **prepare_calibration** | 准备校准 | `qmodel.prepare_calibration()` |
| **calc_qparams** | 计算量化参数 | `qmodel.calc_qparams()` |
| **set_quant** | 启用/禁用量化 | `qmodel.set_quant(w_quant=True, a_quant=True)` |
| **init_QAT** | 初始化 QAT | `qmodel.init_QAT()` |

---

## 3. 主要模块架构

### 3.1 模块层次图

```
sparsebit/
│
├── quantization/                    # 量化模块
│   ├── quant_model.py              # QuantModel - 主入口
│   ├── quant_config.py             # 配置解析
│   ├── quant_tracer.py             # FX 追踪器
│   │
│   ├── modules/                    # 量化算子实现
│   │   ├── base.py                 # QuantOpr, MultipleInputsQuantOpr
│   │   ├── conv.py                 # QConv2d, QConvTranspose2d
│   │   ├── linear.py               # QLinear
│   │   ├── activation.py           # QReLU, QSigmoid, etc.
│   │   └── ...
│   │
│   ├── quantizers/                 # 量化器实现
│   │   ├── base.py                 # Quantizer 基类
│   │   ├── uniform.py              # 均匀量化
│   │   ├── adaround.py             # AdaRound
│   │   ├── lsq.py                  # LSQ
│   │   └── ...
│   │
│   ├── observers/                  # 观察器实现
│   │   ├── base.py                 # Observer, DataCache
│   │   ├── minmax.py               # MinMax 观察器
│   │   ├── mse.py                  # MSE 观察器
│   │   └── ...
│   │
│   ├── converters/                 # 图转换
│   │   ├── simplifiers/            # 图简化
│   │   └── fuse_operations/        # 算子融合
│   │
│   └── tools/                      # 工具函数
│       ├── calibration.py          # 校准流程
│       ├── graph_wrapper.py        # 图包装器
│       └── tensor_wrapper.py       # 张量包装器
│
└── sparse/                          # 剪枝模块
    ├── sparse_model.py             # SparseModel - 主入口
    ├── sparse_config.py            # 剪枝配置
    ├── modules/                    # 剪枝算子
    └── sparsers/                   # 剪枝器实现
```

### 3.2 核心类继承关系

```
                            nn.Module
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   QuantModel             QuantOpr              SparseModel
   (量化模型)              (量化算子基类)          (剪枝模型)
        │                      │                      │
   SparseModel           QConv2d                    SparseOpr
   (剪枝模型)              QLinear                   (剪枝算子基类)
   (同为nn.Module)         QReLU...                      │
                               │                      SConv2d
                         MultipleInputs                  SLinear
                         QuantOpr                        ...
                         (多输入量化算子)


                            nn.Module
                               │
                         Quantizer (量化器基类)
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   UniformQuantizer      AdaRoundQuantizer       LSQQuantizer
        │                      │                      │
   (标准均匀量化)           (自适应舍入)            (可学习步长)


                            nn.Module
                               │
                         Observer (观察器基类)
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
   MinMaxObserver         MSEObserver          PercentileObserver
        │                      │                      │
   (最值统计)             (最小化MSE)            (百分位数)
```

---

## 4. 调用图 (Call Graph)

### 4.1 量化模型初始化流程

```
用户代码
    │
    ▼
┌──────────────────┐
│   QuantModel()   │ ◄────────────────── 构造函数
└────────┬─────────┘
         │
    ┌────┴────┬────────────┬─────────────┐
    ▼         ▼            ▼             ▼
┌────────┐ ┌────────┐ ┌────────────┐ ┌──────────────┐
│_trace()│ │simplify│ │_convert2   │ │_build_       │
│        │ │        │ │ quantmodule│ │ quantizer()  │
└────┬───┘ └───┬────┘ └──────┬─────┘ └──────┬───────┘
     │         │             │              │
     ▼         ▼             ▼              ▼
┌────────┐ ┌────────┐ ┌────────────┐ ┌──────────────┐
│QTracer │ │各种    │ │遍历graph   │ │遍历modules,  │
│.trace()│ │简化器  │ │替换为      │ │为每个        │
│        │ │        │ │QModule     │ │QuantOpr      │
└────────┘ └────────┘ └────────────┘ │构建quantizer │
                                     └──────────────┘
```

### 4.2 校准流程 (Calibration)

```
┌─────────────────────────────┐
│    prepare_calibration()    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│    CalibrationRunner        │
│    - 注册 forward hook      │
│    - 捕获输入输出           │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│      calc_qparams()         │
└─────────────┬───────────────┘
              │
    ┌─────────┴─────────┐
    ▼                   ▼
┌─────────────┐   ┌─────────────────┐
│run_feature_ │   │ run_weight_     │
│calibration()│   │ calibration()   │
└──────┬──────┘   └────────┬────────┘
       │                   │
       ▼                   ▼
┌─────────────┐   ┌─────────────────┐
│update_      │   │ update_observer │
│observer()   │   │ (weight)        │
└──────┬──────┘   └────────┬────────┘
       │                   │
       └─────────┬─────────┘
                 ▼
        ┌─────────────────┐
        │  calc_qparams() │ ◄── Observer 基类方法
        │  (计算scale/zp) │
        └─────────────────┘
```

### 4.3 前向传播流程

```
输入数据 x
    │
    ▼
┌─────────────────┐
│  input_quantizer│ ◄── 如果启用，对输入进行量化
│      (x)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ weight_quantizer│ ◄── 如果启用，对权重进行量化
│    (weight)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   原始算子运算   │ ◄── 如 F.conv2d, F.linear
│                 │
└────────┬────────┘
         │
         ▼
    输出结果
```

---

## 5. 重要文件 Top 10

| 排名 | 文件路径 | 核心类/函数 | 重要性说明 |
|------|----------|-------------|-----------|
| 1 | `quantization/quant_model.py` | `QuantModel` | 量化主入口，控制整个流程 |
| 2 | `quantization/modules/base.py` | `QuantOpr` | 所有量化算子的基类 |
| 3 | `quantization/quantizers/base.py` | `Quantizer` | 量化器基类，定义量化接口 |
| 4 | `quantization/observers/base.py` | `Observer` | 观察器基类，统计信息收集 |
| 5 | `quantization/tools/calibration.py` | `CalibrationRunner` | 校准流程控制 |
| 6 | `quantization/quant_config.py` | `parse_qconfig` | 配置解析和验证 |
| 7 | `quantization/quant_tracer.py` | `QTracer` | FX 图追踪 |
| 8 | `sparse/sparse_model.py` | `SparseModel` | 剪枝主入口 |
| 9 | `quantization/converters/simplifiers/__init__.py` | `simplify` | 图简化 |
| 10 | `quantization/converters/fuse_operations/__init__.py` | `fuse_operations` | 算子融合 |

---

## 6. 关键设计模式

### 6.1 注册器模式 (Registry Pattern)

```python
# QMODULE_MAP 注册器示例
QMODULE_MAP = {}

def register_qmodule(sources):
    def real_register(qmodule):
        for src in sources:
            QMODULE_MAP[src] = qmodule
        return qmodule
    return real_register

# 使用装饰器注册
@register_qmodule(sources=[nn.Conv2d])
class QConv2d(QuantOpr):
    ...
```

### 6.2 策略模式 (Strategy Pattern)

```python
# Quantizer 和 Observer 使用策略模式
# 通过配置选择不同的实现
QUANTIZERS_MAP = {
    "uniform": UniformQuantizer,
    "adaround": AdaRoundQuantizer,
    "lsq": LSQQuantizer,
    ...
}

OBSERVERS_MAP = {
    "minmax": MinMaxObserver,
    "mse": MSEObserver,
    ...
}
```

### 6.3 模板方法模式 (Template Method)

```python
# Quantizer 基类定义模板
class Quantizer(nn.Module):
    def forward(self, x):
        if self.is_enable:
            scale, zp = self._qparams_preprocess(x)
            return self._forward(x, scale, zp)  # 子类实现
        return x
    
    def _forward(self, x, scale, zp):  # 抽象方法
        raise NotImplementedError
```

---

## 7. 数据流图

```
原始 PyTorch 模型 (nn.Module)
           │
           ▼
    ┌──────────────┐
    │  FX Trace    │ ◄── 使用 QTracer 追踪
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ GraphModule  │ ◄── FX 图表示
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │   Simplify   │ ◄── 图简化优化
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Convert2Quant│ ◄── 替换为 QuantOpr
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │BuildQuantizer│ ◄── 附加 Quantizer
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │   Calibrate  │ ◄── 收集统计信息
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │ Export ONNX  │ ◄── 导出 QDQ-ONNX
    └──────────────┘
```

---

## 8. 配置系统

配置使用 YACS (Yet Another Configuration System) 的 CfgNode：

```yaml
# 配置层次结构
BACKEND: "virtual"                    # 后端: virtual/onnxruntime/tensorrt
DEVICE: "cuda"
SKIP_TRACE_MODULES: []                # 跳过的模块名列表

SCHEDULE:
  FUSE_BN: false                      # 是否融合 BN
  BN_TUNING: false                    # BN 微调
  DISABLE_UNNECESSARY_QUANT: true     # 禁用不必要的量化

W:                                    # 权重 (Weight) 配置
  QSCHEME: "per-channel-symmetric"    # 量化方案
  QUANTIZER:
    TYPE: "uniform"                   # 量化器类型
    BIT: 8                            # 比特数
  OBSERVER:
    TYPE: "MINMAX"                    # 观察器类型
  SPECIFIC: []                        # 特定模块配置

A:                                    # 激活 (Activation) 配置
  QSCHEME: "per-tensor-symmetric"
  QUANTIZER:
    TYPE: "uniform"
    BIT: 8
  OBSERVER:
    TYPE: "MINMAX"
```
