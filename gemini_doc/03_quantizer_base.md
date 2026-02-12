# Quantizer - 量化器基类

## 文件信息

- **路径**: `sparsebit/quantization/quantizers/base.py`
- **核心类**: `Quantizer`
- **继承**: `torch.nn.Module`

---

## OCaml Type Signature

```ocaml
type qdesc = {
  bit : int;
  scheme : torch.qscheme;
  ch_axis : int;
  qrange : int * int;
  is_perchannel : bool;
  is_symmetric : bool;
  target : quant_target;
}

type quantizer = {
  cfg : config;
  qdesc : qdesc;
  device : torch.device;
  scale : torch.tensor;
  zero_point : torch.tensor;
  observer : observer;
  use_quant : bool;
  export_onnx : bool;
  fake_fused : bool;
  dims : int;  (* 输入张量的维度数 *)
}

class quantizer : config ->
  method forward : torch.tensor -> torch.tensor
  method _forward : torch.tensor -> torch.tensor -> torch.tensor -> torch.tensor
  method calc_qparams : unit -> torch.tensor * torch.tensor
  method update_observer : torch.tensor -> unit
  method set_backend : backend -> unit
  method set_fake_fused : unit -> unit
  method enable_quant : unit -> unit
  method disable_quant : unit -> unit
  method enable_export_onnx : unit -> unit
  method disable_export_onnx : unit -> unit
end
```

---

## 详细功能说明

### Quantizer 类

`Quantizer` 是所有量化器的**抽象基类**。它负责：
1. 管理量化参数（scale, zero_point）
2. 协调 Observer 收集统计信息
3. 执行实际的量化/反量化操作
4. 支持多种后端（virtual, onnxruntime, tensorrt）

### 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `scale` | `torch.Tensor` | 量化缩放因子 |
| `zero_point` | `torch.Tensor` | 量化零点 |
| `qdesc` | `QuantDescriptor` | 量化描述符（位宽、方案等） |
| `observer` | `Observer` | 观察器，用于收集统计信息 |
| `use_quant` | `bool` | 是否启用量化 |
| `export_onnx` | `bool` | 是否为 ONNX 导出模式 |
| `fake_fused` | `bool` | 是否假融合（禁用状态） |

### 量化公式

**对称量化**:
```
x_q = round(x_f / scale)
x_dq = x_q * scale
scale = max(|x_min|, |x_max|) * 2 / (2^bit - 1)
zero_point = 0
```

**非对称量化**:
```
x_q = round(x_f / scale) + zero_point
x_dq = (x_q - zero_point) * scale
scale = (x_max - x_min) / (2^bit - 1)
zero_point = round(-x_min / scale)
```

---

## 核心方法

### `__init__(self, config)`

初始化量化器。

**关键操作**:
1. 创建 `QuantDescriptor`
2. 注册 `scale` 和 `zero_point` 为 buffer
3. 构建 Observer
4. 如果 `QUANTIZER.DISABLE=True`，设置 `fake_fused`

---

### `forward(self, x: torch.Tensor) -> torch.Tensor`

前向传播，执行量化操作。

**实现逻辑**:
```python
def forward(self, x):
    if self.is_enable:  # use_quant and not fake_fused
        scale, zp = self._qparams_preprocess(x)
        if self.export_onnx:
            # 使用 PyTorch 原生 fake_quant 以便导出 ONNX
            x_dq = torch_fake_quant(x, scale, zp, self.qdesc)
        else:
            # 使用自定义量化实现
            x_dq = self._forward(x, scale, zp)
    else:
        x_dq = x  # 直通，不量化
    return x_dq
```

---

### `_forward(self, x_f, scale, zero_point) -> torch.Tensor`

**抽象方法**，子类必须实现具体的量化逻辑。

---

### `calc_qparams(self) -> (scale, zero_point)`

计算量化参数。

**实现逻辑**:
```python
def calc_qparams(self):
    if self.fake_fused:
        return self.scale, self.zero_point
    
    # 从 observer 获取 min/max，计算 scale/zp
    scale, zp = self.observer.calc_qparams()
    
    # 广播到正确的形状（支持 per-channel）
    self.scale = self._broadcast_qparams(scale)
    self.zero_point = self._broadcast_qparams(zp)
    
    return self.scale, self.zero_point
```

---

### `update_observer(self, x: torch.Tensor)`

更新观察器的统计数据。

```python
def update_observer(self, x):
    self.dims = len(x.shape)  # 记录维度，用于广播
    self.observer.data_cache.update(x.detach())
```

---

### `_broadcast_qparams(self, params)`

将量化参数广播到与输入张量兼容的形状。

```python
def _broadcast_qparams(self, params):
    # 例如：对于 per-channel 卷积权重 [C_out, C_in, K, K]
    # ch_axis=0，需要将 scale 变为 [C_out, 1, 1, 1]
    dst_shape = [1] * self.dims
    dst_shape[self.qdesc.ch_axis] = -1
    return params.reshape(dst_shape)
```

---

### `set_fake_fused(self)`

设置假融合状态。用于完全禁用某个量化器。

```python
def set_fake_fused(self):
    self.fake_fused = True
    if isinstance(self.scale, nn.Parameter):
        self.scale.requires_grad_(False)
        self.zero_point.requires_grad_(False)
    else:
        self.scale = torch.tensor([1.0])
        self.zero_point = torch.tensor([0.0])
```

---

### `enable_export_onnx() / disable_export_onnx()`

控制 ONNX 导出模式。

**关键操作**:
```python
def enable_export_onnx(self):
    self.export_onnx = True
    self.zero_point = self.zero_point.round()  # ONNX 要求整数零点
```

---

## 属性方法

| 属性 | 说明 |
|------|------|
| `is_enable` | `use_quant and not fake_fused` |
| `bit` | 量化位宽 |
| `ch_axis` | 通道维度索引 |
| `is_perchannel` | 是否为 per-channel 量化 |
| `is_symmetric` | 是否为对称量化 |

---

## 继承关系

```
                    nn.Module
                       │
                   Quantizer
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
   UniformQuantizer  AdaRound      LSQQuantizer  LSQPlusQuantizer
   (标准均匀量化)     (自适应舍入)   (可学习步长)   (改进版LSQ)
        │              │              │              │
   STE.apply      基于优化问题      学习 scale      学习 scale+bias
```

---

## 典型子类实现

### UniformQuantizer

```python
@register_quantizer
class Quantizer(BaseQuantizer):
    TYPE = "uniform"
    
    def _forward(self, x_f, scale, zero_point):
        # 使用直通估计器 (Straight-Through Estimator)
        x_dq = STE.apply(x_f, scale, zero_point, self.qdesc, self.backend)
        return x_dq
```

### STE (Straight-Through Estimator)

```python
class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_f, scale, zero_point, qdesc, backend):
        # 前向：执行量化
        x_int = torch.clamp(
            round_pass(x_f / scale) + zero_point,
            qdesc.qmin, qdesc.qmax
        )
        x_dq = (x_int - zero_point) * scale
        return x_dq
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向：直通梯度
        return grad_output, None, None, None, None
```

---

## 依赖关系

```
Quantizer
    ├── QuantDescriptor (quantizers/quant_descriptor.py)
    ├── build_observer (observers/__init__.py)
    ├── torch_fake_quant (quantizers/quant_tensor.py)
    └── STE (quantizers/quant_tensor.py)
```
