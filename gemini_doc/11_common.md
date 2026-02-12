# Common - 共享定义与工具

## 文件信息

- **路径**: `sparsebit/quantization/common.py`
- **内容**: 枚举定义、工具函数

---

## OCaml Type Signature

```ocaml
type granularity = Layerwise | Channelwise

type quant_target = Weight | Feature

type backend = Virtual | Onnxruntime | Tensorrt

type qscheme =
  | PerTensorSymmetric
  | PerTensorAffine
  | PerChannelSymmetric
  | PerChannelAffine

val get_backend : string -> backend
val get_qscheme : string -> torch.qscheme
```

---

## 枚举定义

### Granularity - 校准粒度

```python
class Granularity(Enum):
    LAYERWISE = 0    # 层级别：整个张量作为一个整体
    CHANNELWISE = 1  # 通道级别：每个通道单独处理
```

**用途**: `DataCache.get_data_for_calibration()` 根据粒度组织数据。

---

### QuantTarget - 量化目标

```python
class QuantTarget(Enum):
    WEIGHT = 0   # 量化权重
    FEATURE = 1  # 量化特征（激活）
```

**用途**: 区分是构建 `weight_quantizer` 还是 `input_quantizer`。

---

### Backend - 后端类型

```python
class Backend(Enum):
    VIRTUAL = 0      # 虚拟平台，用于研究
    ONNXRUNTIME = 1  # ONNX Runtime 后端
    TENSORRT = 2     # TensorRT 后端
```

**约束**:
| 后端 | 位宽支持 | 量化方案约束 |
|------|---------|-------------|
| VIRTUAL | 任意 | 无 |
| ONNXRUNTIME | 仅 8bit | 无 |
| TENSORRT | 仅 8bit | 权重 per-channel-symmetric，激活 per-tensor-symmetric |

---

## 工具函数

### `get_backend(backend: str) -> Backend`

将字符串转换为 Backend 枚举。

```python
def get_backend(backend):
    if backend == "virtual":
        return Backend.VIRTUAL
    if backend == "onnxruntime":
        return Backend.ONNXRUNTIME
    if backend == "tensorrt":
        return Backend.TENSORRT
    raise TypeError("only support backend in {}".format(target_backend))
```

---

### `get_qscheme(qscheme: str) -> torch.qscheme`

将字符串转换为 PyTorch qscheme。

**支持的格式**:

| 输入字符串 | PyTorch 值 | 说明 |
|-----------|-----------|------|
| `per-tensor-symmetric` | `torch.per_tensor_symmetric` | 张量级别对称 |
| `per-tensor-affine` | `torch.per_tensor_affine` | 张量级别非对称 |
| `per-channel-symmetric` | `torch.per_channel_symmetric` | 通道级别对称 |
| `per-channel-affine` | `torch.per_channel_affine` | 通道级别非对称 |

```python
def get_qscheme(qscheme):
    if qscheme == "per-tensor-symmetric":
        return torch.per_tensor_symmetric
    if qscheme == "per-tensor-affine":
        return torch.per_tensor_affine
    if qscheme == "per-channel-symmetric":
        return torch.per_channel_symmetric
    if qscheme == "per-channel-affine":
        return torch.per_channel_affine
    raise TypeError(...)
```

---

## 使用场景

### 在 `QuantOpr.build_quantizer()` 中

```python
def build_quantizer(self, config):
    _backend = get_backend(config.BACKEND)
    
    if self.weight is not None:
        update_config(config.W, "TARGET", (QuantTarget.WEIGHT,))
        self.weight_quantizer = build_quantizer(cfg=config.W)
        self.weight_quantizer.set_backend(_backend)
    
    update_config(config.A, "TARGET", (QuantTarget.FEATURE,))
    self.input_quantizer = build_quantizer(cfg=config.A)
    self.input_quantizer.set_backend(_backend)
```

### 在 `Observer` 中

```python
def calc_minmax(self):
    if self.is_perchannel:
        data = self.data_cache.get_data_for_calibration(Granularity.CHANNELWISE)
        ...
    else:
        data = self.data_cache.get_data_for_calibration(Granularity.LAYERWISE)
        ...
```

---

## 量化方案对比

```
Per-Tensor vs Per-Channel:

Per-Tensor:                     Per-Channel:
┌─────────────────┐            ┌─────────────────┐
│  整个张量共享   │            │  每个通道独立   │
│  一个 scale/zp  │            │  scale/zp      │
│                 │            │                 │
│  [C,H,W]        │            │  [C,H,W]        │
│     ↓           │            │     ↓           │
│  scale[1]       │            │  scale[C]       │
│  zp[1]          │            │  zp[C]          │
└─────────────────┘            └─────────────────┘

适用场景:
- Per-Tensor: 激活（通常通道间分布相似）
- Per-Channel: 权重（不同通道分布差异大）

Symmetric vs Affine:

Symmetric:                      Affine:
┌─────────────────┐            ┌─────────────────┐
│  零点固定为 0   │            │  零点可调整     │
│                 │            │                 │
│  -max ─┬─ +max  │            │  min ─┬─ max    │
│        0        │            │       zp        │
│                 │            │                 │
│  范围对称       │            │  范围可不对称   │
└─────────────────┘            └─────────────────┘

适用场景:
- Symmetric: 权重（通常以 0 为中心）
- Affine: 激活（ReLU 后非负，分布不对称）
```
