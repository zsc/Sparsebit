# 量化配置系统

## 文件信息

- **路径**: `sparsebit/quantization/quant_config.py`
- **核心函数**: `parse_qconfig`, `verify_bits`, `verify_backend`, `verify_schedule`

---

## OCaml Type Signature

```ocaml
type config = CfgNode

type weight_config = {
  qscheme : string option;              (* "per-channel-symmetric" etc. *)
  quantizer_type : string;              (* "uniform", "adaround", etc. *)
  quantizer_disable : bool;
  quantizer_bit : int;
  observer_type : string;               (* "MINMAX", "MSE", etc. *)
  specific : (string * config) list;    (* 特定模块覆盖配置 *)
}

type activation_config = {
  qscheme : string option;
  quantizer_type : string;
  quantizer_disable : bool;
  quantizer_bit : int;
  observer_type : string;
  moving_average_ema_ratio : float;
  aciq_distribution : string;           (* "GAUS", "LAPLACE" *)
  layout : string;                      (* "NCHW", "NLC" *)
  specific : (string * config) list;
}

type schedule_config = {
  fuse_bn : bool;
  bn_tuning : bool;
  disable_unnecessary_quant : bool;
}

type quantization_config = {
  backend : string;                     (* "virtual", "onnxruntime", "tensorrt" *)
  device : string;
  skip_trace_modules : string list;
  schedule : schedule_config;
  weight : weight_config;
  activation : activation_config;
}

val parse_qconfig : string -> config
val verify_bits : config -> unit
val verify_backend : config -> unit
val verify_schedule : config -> unit
```

---

## 详细功能说明

### 配置结构

配置使用 **YACS (Yet Another Configuration System)** 的 `CfgNode`，支持层次化配置。

```
ROOT
├── BACKEND: str                    # 后端类型
├── DEVICE: str                     # 运行设备
├── SKIP_TRACE_MODULES: list        # 跳过的模块名（支持通配符）
│
├── SCHEDULE                        # 调度配置
│   ├── FUSE_BN: bool              # 是否融合 BatchNorm
│   ├── BN_TUNING: bool            # 是否进行 BN 微调
│   └── DISABLE_UNNECESSARY_QUANT: bool
│
├── W                               # 权重配置
│   ├── QSCHEME: str               # 量化方案
│   ├── QUANTIZER
│   │   ├── TYPE: str              # 量化器类型
│   │   ├── DISABLE: bool
│   │   └── BIT: int               # 量化位宽
│   ├── OBSERVER
│   │   ├── TYPE: str              # 观察器类型
│   │   ├── PERCENTILE.ALPHA: float
│   │   └── ACIQ.DISTRIBUTION: str
│   └── SPECIFIC: list             # 特定模块覆盖
│
└── A                               # 激活配置
    ├── QSCHEME: str
    ├── QUANTIZER
    │   ├── TYPE: str
    │   ├── DISABLE: bool
    │   ├── BIT: int
    │   └── PACT.ALPHA_VALUE: float
    ├── OBSERVER
    │   ├── TYPE: str
    │   ├── MOVING_AVERAGE.EMA_RATIO: float
    │   └── LAYOUT: str
    ├── QADD.ENABLE_QUANT: bool    # 是否量化 Add 操作
    └── SPECIFIC: list
```

---

## 核心函数

### `parse_qconfig(cfg_file) -> CfgNode`

解析配置文件并返回配置对象。

**流程**:
```python
def parse_qconfig(cfg_file):
    qconfig = _parse_config(cfg_file, default_cfg=_C)
    verify_bits(qconfig)
    verify_backend(qconfig)
    verify_schedule(qconfig)
    return qconfig
```

---

### `verify_bits(qconfig)`

验证量化位宽设置。

```python
def verify_bits(qconfig):
    assert qconfig.W.QUANTIZER.BIT >= 0, "权重位宽应为非负数"
    assert qconfig.A.QUANTIZER.BIT >= 0, "激活位宽应为非负数"
```

**特殊含义**:
- `BIT = 0`: 禁用该量化器（已废弃，建议使用 `QUANTIZER.DISABLE`）
- `BIT > 0`: 正常量化

---

### `verify_backend(qconfig)`

验证后端配置是否一致。

**约束**:

| 后端 | 约束 |
|------|------|
| `onnxruntime` | 仅支持 8bit 量化 |
| `tensorrt` | 仅支持 8bit，权重 per-channel-symmetric，激活 per-tensor-symmetric |
| `virtual` | 无约束，用于研究 <8bit 场景 |

```python
def verify_backend(qconfig):
    backend = get_backend(qconfig.BACKEND)
    w_qscheme = get_qscheme(qconfig.W.QSCHEME)
    a_qscheme = get_qscheme(qconfig.A.QSCHEME)
    
    if backend in [Backend.ONNXRUNTIME, Backend.TENSORRT]:
        assert wbit == 8 and abit == 8
    
    if backend == Backend.TENSORRT:
        assert w_qscheme == torch.per_channel_symmetric
        assert a_qscheme == torch.per_tensor_symmetric
```

---

### `verify_schedule(qconfig)`

验证调度配置。

```python
def verify_schedule(qconfig):
    if qconfig.SCHEDULE.BN_TUNING:
        w_qscheme = get_qscheme(qconfig.W.QSCHEME)
        assert w_qscheme in [torch.per_channel_symmetric, 
                             torch.per_channel_affine]
        # BN 微调需要 per-channel 权重量化
```

---

## QScheme 格式

支持的量化方案格式：

| 配置字符串 | PyTorch 对应值 | 说明 |
|-----------|---------------|------|
| `per-tensor-symmetric` | `torch.per_tensor_symmetric` | 张量级别对称 |
| `per-tensor-affine` | `torch.per_tensor_affine` | 张量级别非对称 |
| `per-channel-symmetric` | `torch.per_channel_symmetric` | 通道级别对称 |
| `per-channel-affine` | `torch.per_channel_affine` | 通道级别非对称 |

---

## 配置示例

### 基础 PTQ 配置

```yaml
BACKEND: "virtual"
DEVICE: "cuda"
SKIP_TRACE_MODULES: ["head*", "neck*"]  # 跳过 head 和 neck 模块

SCHEDULE:
  FUSE_BN: true
  BN_TUNING: false
  DISABLE_UNNECESSARY_QUANT: true

W:
  QSCHEME: "per-channel-symmetric"
  QUANTIZER:
    TYPE: "uniform"
    BIT: 8
  OBSERVER:
    TYPE: "MINMAX"
  SPECIFIC: []

A:
  QSCHEME: "per-tensor-symmetric"
  QUANTIZER:
    TYPE: "uniform"
    BIT: 8
  OBSERVER:
    TYPE: "MINMAX"
  SPECIFIC:
    - ["layer1*", ["QUANTIZER.BIT", 6]]  # layer1 使用 6bit
```

### AdaRound 配置

```yaml
W:
  QUANTIZER:
    TYPE: "adaround"
    BIT: 4
  OBSERVER:
    TYPE: "MSE"  # AdaRound 需要 MSE observer
```

### LSQ QAT 配置

```yaml
W:
  QUANTIZER:
    TYPE: "lsq"
    BIT: 4

A:
  QUANTIZER:
    TYPE: "lsq"
    BIT: 4
  OBSERVER:
    TYPE: "MOVING_AVERAGE"
    MOVING_AVERAGE:
      EMA_RATIO: 0.9
```

---

## 依赖关系

```
quant_config
    ├── _parse_config (utils/yaml_utils.py)
    ├── update_config (utils/yaml_utils.py)
    ├── get_backend (common.py)
    ├── get_qscheme (common.py)
    └── CfgNode (yacs)
```
