# 06：`sparsebit/quantization/observers/base.py` —— Observer 与 DataCache

## 1. 角色定位

Observer 的职责是：**收集统计量并产出量化参数**。在 Sparsebit 的实现里，它被拆成两层：

- `DataCache`：缓存观测数据，并提供“按粒度整理”的数据视图
- `Observer`：基类实现 `calc_qparams_with_minmax`（把 min/max → scale/zp），并把 `calc_minmax` 留给子类实现

## 2. DataCache：缓存与整理数据

### 2.1 `update(data)`

简单 append 到 `_data_cache`，通常由 `Quantizer.update_observer(x)` 调用。

### 2.2 `get_data_for_calibration(granularity)`

支持两种粒度：

- `Granularity.CHANNELWISE`
  - `data = cat(cache, dim=ch_axis)`
  - 若 `ch_axis != 0`：把 channel 维换到 0
  - `data = flatten(1)`，得到形状约为 `[C, N]`
- `Granularity.LAYERWISE`
  - 把所有 batch flatten 后再 cat，得到 1D 张量

这个过程隐含了一个关键假设：per-channel 量化时，**channel 维是可拼接且可置换的**。

### 2.3 `get_batch_size()`

- 如果 target 是 `WEIGHT`：返回 `None`
- 如果是 `FEATURE`：返回所有 cached tensor 在 batch 维（`bs_axis`）的总和

## 3. Observer：从 min/max 到 scale/zp

### 3.1 `calc_qparams()`

调用链：

```text
calc_qparams()
  -> calc_minmax()                 # 子类实现，必须 reset data_cache
  -> calc_qparams_with_minmax()
```

### 3.2 `calc_qparams_with_minmax(min_val, max_val)`

核心逻辑：

- `min_val_neg = min(min_val, 0)`
- `max_val_pos = max(max_val, 0)`
- `qmin/qmax = qdesc.qrange`
- 若 symmetric：
  - `max_val_pos = max(-min_val_neg, max_val_pos)`
  - `scale = max_val_pos * 2 / (qmax - qmin)`
  - `zero_point = 0`
- 若 affine：
  - `scale = (max_val_pos - min_val_neg) / (qmax - qmin)`
  - `zero_point = round(-min_val_neg / scale)`
- `scale` 强制下界 `1e-6`
- 最后 assert `len(data_cache) == 0`（要求子类已 reset）

## 4. 子类 Observer 的实现范式

以 `minmax` 为例（`sparsebit/quantization/observers/minmax.py`）：

- per-channel：`data.max(axis=1)` / `data.min(axis=1)`
- per-tensor：`data.max()` / `data.min()`
- 计算完成后 `data_cache.reset()`

其他 observer（mse、percentile、kl_histogram、moving_average、aciq）会覆盖 `calc_minmax` 或直接覆盖 `calc_qparams`。

## 5. OCaml type signature（接口投影）

```ocaml
type tensor
type granularity = Layerwise | Channelwise

type data_cache
val update : data_cache -> tensor -> unit
val reset : data_cache -> unit
val get_data_for_calibration : data_cache -> granularity -> tensor

type observer
val calc_qparams : observer -> tensor * tensor
val calc_qparams_with_minmax : observer -> min_val:tensor -> max_val:tensor -> tensor * tensor
```

## 6. 重写时的注意事项

- DataCache 的“拼接+置换+flatten”策略很依赖 layout 假设（NCHW/NLC）；重写时建议把 layout 变成显式类型，并在输入检查阶段就报错。
- 当前实现用 `assert len(data_cache) == 0` 强制约束子类 reset；更稳妥的方式是由基类统一 reset（或者让 cache 只提供一次性 iterator）。

