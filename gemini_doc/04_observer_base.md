# Observer - 观察器基类

## 文件信息

- **路径**: `sparsebit/quantization/observers/base.py`
- **核心类**: `Observer`, `DataCache`

---

## OCaml Type Signature

```ocaml
type granularity = Layerwise | Channelwise

type quant_target = Weight | Feature

type data_cache = {
  qdesc : qdesc;
  _data_cache : torch.tensor list;
}

type observer = {
  cfg : config;
  qdesc : qdesc;
  device : torch.device;
  min_val : torch.tensor;
  max_val : torch.tensor;
  data_cache : data_cache;
}

(* DataCache 方法 *)
val update : data_cache -> torch.tensor -> unit
val reset : data_cache -> unit
val get_data_for_calibration : data_cache -> granularity -> torch.tensor
val get_batch_size : data_cache -> int option

(* Observer 方法 *)
val calc_qparams : observer -> torch.tensor * torch.tensor
val calc_qparams_with_minmax : observer -> torch.tensor -> torch.tensor -> torch.tensor * torch.tensor
val calc_minmax : observer -> torch.tensor * torch.tensor
```

---

## 详细功能说明

### DataCache 类

`DataCache` 用于**缓存校准数据**，支持两种粒度：
- **Layerwise**: 层级别，将所有数据展平后拼接
- **Channelwise**: 通道级别，沿通道维度拼接

#### 核心方法

##### `update(self, data: torch.Tensor)`

添加新的数据到缓存。

```python
def update(self, data):
    self._data_cache.append(data)
```

##### `get_data_for_calibration(self, granularity: Granularity) -> torch.Tensor`

获取用于校准的格式化数据。

**Layerwise 处理**:
```python
data = torch.cat([d.reshape(-1) for d in self._data_cache], axis=0)
# 结果: [总元素数]
```

**Channelwise 处理**:
```python
data = torch.cat(self._data_cache, dim=self.qdesc.ch_axis)
if self.qdesc.ch_axis != 0:
    data = data.transpose(0, self.qdesc.ch_axis)
data = data.flatten(1)
# 结果: [通道数, 每通道元素数]
```

##### `reset(self)`

清空缓存。

```python
def reset(self):
    self._data_cache = []
```

---

### Observer 类

`Observer` 是所有观察器的**抽象基类**。它负责：
1. 收集张量的统计信息（通过 `DataCache`）
2. 计算 min/max
3. 计算量化参数（scale, zero_point）

#### 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `min_val` | `torch.Tensor` | 观察到的最小值 |
| `max_val` | `torch.Tensor` | 观察到的最大值 |
| `data_cache` | `DataCache` | 数据缓存 |
| `qdesc` | `QuantDescriptor` | 量化描述符 |

#### 核心方法

##### `calc_qparams(self) -> (scale, zero_point)`

计算量化参数的主入口。

```python
def calc_qparams(self):
    min_val, max_val = self.calc_minmax()  # 子类实现
    scale, zero_point = self.calc_qparams_with_minmax(min_val, max_val)
    return scale, zero_point
```

##### `calc_qparams_with_minmax(self, min_val, max_val) -> (scale, zero_point)`

**核心算法**，根据 min/max 计算 scale 和 zero_point。

**对称量化**:
```python
max_val_pos = torch.maximum(-min_val_neg, max_val_pos)
scale = max_val_pos * 2 / float(qmax - qmin)
scale = torch.maximum(scale, torch.tensor(1e-6))
zero_point = torch.zeros_like(scale)
```

**非对称量化**:
```python
scale = (max_val_pos - min_val_neg) / float(qmax - qmin)
scale = torch.maximum(scale, torch.tensor(1e-6))
zero_point = torch.round(-min_val_neg / scale)
```

其中：
- `qmin, qmax = self.qdesc.qrange`（如 8bit: -128, 127）
- `min_val_neg = minimum(min_val, 0)`
- `max_val_pos = maximum(max_val, 0)`

---

## 继承关系

```
                    nn.Module
                       │
                   Observer
                       │
        ┌──────────────┼──────────────┬──────────────┬──────────────┐
        │              │              │              │              │
   MinMaxObserver  MSEObserver  PercentileObserver  KLHistogram  ACIQObserver
   (最值统计)      (最小化MSE)   (百分位数截断)    (KL散度)     (分布假设)
        │              │              │              │              │
   简单取min/max   搜索最优min/max  按百分位截断    直方图匹配    假设为高斯/
                                                   最优阈值      拉普拉斯分布
```

---

## 典型子类实现

### MinMaxObserver

最简单直接的观察器，直接取数据的最小最大值。

```python
@register_observer
class Observer(BaseObserver):
    TYPE = "minmax"
    
    def calc_minmax(self):
        if self.is_perchannel:
            # 按通道处理
            data = self.data_cache.get_data_for_calibration(Granularity.CHANNELWISE)
            max_val = data.max(axis=1).values
            min_val = data.min(axis=1).values
        else:
            # 层级别处理
            data = self.data_cache.get_data_for_calibration(Granularity.LAYERWISE)
            min_val, max_val = data.min(), data.max()
        
        self.data_cache.reset()  # 清空缓存
        self.min_val = min_val.to(self.device)
        self.max_val = max_val.to(self.device)
        return self.min_val, self.max_val
```

---

## 观察器对比

| 观察器 | 核心思想 | 优点 | 缺点 |
|--------|---------|------|------|
| **MinMax** | 直接取 min/max | 简单快速 | 对异常值敏感 |
| **MSE** | 搜索使 MSE 最小的 min/max | 量化误差小 | 计算开销大 |
| **Percentile** | 按百分位数截断 | 鲁棒性好 | 需要调参 |
| **KLHistogram** | 最小化 KL 散度 | 适合激活分布 | 需要直方图计算 |
| **ACIQ** | 假设为高斯/拉普拉斯分布 | 有理论依据 | 假设可能不成立 |
| **MovingAverage** | EMA 更新 min/max | 适合 QAT | 需要调 EMA 比例 |

---

## 依赖关系

```
Observer
    ├── DataCache
    ├── QuantDescriptor (common.py)
    ├── Granularity (common.py)
    └── QuantTarget (common.py)
```
