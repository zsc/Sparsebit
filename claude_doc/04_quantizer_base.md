# 04：`sparsebit/quantization/quantizers/base.py` —— Quantizer 基类与生命周期

## 1. 角色定位

`Quantizer` 是“量化行为”的封装：

- **校准阶段**：接收 `Observer` 的统计结果，得到 `scale/zero_point`
- **训练/推理阶段**：根据开关决定是否对张量做 fake-quant（量化→反量化）
- **导出阶段**：切换到 ONNX 友好的 fake-quant 实现

它是连接 `Observer`（统计）与 `QuantOpr`（算子）之间的核心部件。

## 2. 关键字段（理解状态机）

- `cfg`：量化配置（W 或 A 的子配置）
- `qdesc : QuantDescriptor`：由 `QSCHEME/BIT/TARGET/LAYOUT` 推导出的“量化描述符”
- `scale` / `zero_point`：buffer（或某些量化器里替换为 `nn.Parameter`）
- `observer`：由 `build_observer(cfg, qdesc)` 构建
- `use_quant : bool`：运行时量化开关（由 `enable_quant/disable_quant` 控制）
- `export_onnx : bool`：导出开关（导出时走 `torch_fake_quant`）
- `fake_fused : bool`：逻辑禁用（被图 pass 用来“关闭量化”）

初始化时若 `cfg.QUANTIZER.DISABLE=True` 会直接 `set_fake_fused()`。

## 3. qparams 计算

### 3.1 `update_observer(x)`

把 `x.detach()` 放进 `observer.data_cache`，并记录 `dims = len(x.shape)` 以便后续广播 qparams。

### 3.2 `calc_qparams()`

- 若 `fake_fused=True`：直接返回现有 `scale/zp`（通常是 1/0）
- 否则：
  - 调用 `observer.calc_qparams()`（内部会 `calc_minmax`）
  - 把返回的 `scale/zp` 通过 `_broadcast_qparams` reshape 成可与输入张量 broadcast 的形状

### 3.3 `_broadcast_qparams(params)`

构造 `[1; 1; ...]` 的 shape，并在 `qdesc.ch_axis` 维设为 `-1`，实现 per-channel 的 broadcast。

## 4. forward 路径（运行时行为）

```text
Quantizer.forward(x):
  if is_enable:
     scale,zp = _qparams_preprocess(x)
     if export_onnx:
        x_dq = torch_fake_quant(x, scale, zp, qdesc)
     else:
        x_dq = _forward(x, scale, zp)   # 子类实现（通常调用 quant_tensor.STE）
  else:
     x_dq = x
```

其中 `is_enable = use_quant && (not fake_fused)`。

## 5. 导出与 fake_fused

- `enable_export_onnx()`：
  - `export_onnx=True`
  - 把 `zero_point` round（ONNX 里 zp 是整数语义）
- `set_fake_fused()`：
  - `fake_fused=True`
  - 强制把 `scale=1`、`zp=0` 且禁止梯度

这两者共同支持 fuse/disable pass 的需求：

- 有的 pass 真的 fuse 掉结构（例如 fuse_bn）
- 有的 pass 只是“关闭某些层量化”（例如 disable_unnecessary_quant），就靠 `fake_fused` 来完成

## 6. OCaml type signature（接口投影）

```ocaml
type cfg
type tensor
type backend
type qdesc
type observer

type quantizer

val create : cfg -> quantizer
val set_backend : quantizer -> backend -> unit

val update_observer : quantizer -> tensor -> unit
val calc_qparams : quantizer -> tensor * tensor
val calc_qparams_with_minmax : quantizer -> min_val:tensor -> max_val:tensor -> tensor * tensor

val enable_quant : quantizer -> unit
val disable_quant : quantizer -> unit
val enable_export_onnx : quantizer -> unit
val disable_export_onnx : quantizer -> unit
val set_fake_fused : quantizer -> unit

val forward : quantizer -> tensor -> tensor
```

## 7. 重写时的注意事项

- `scale/zp` 在基类里是 buffer，但 LSQ 等量化器会把它们变成 `nn.Parameter`；新实现要明确支持“qparams 可训练”这一类算法。
- 当前实现依赖 `update_observer` 时记录的 `dims` 来广播 qparams，这个隐式状态容易踩坑；建议让广播逻辑显式绑定到输入 shape。

