# 02：`sparsebit/quantization/quant_config.py` —— 量化配置与校验

## 1. 角色定位

该文件定义了量化的默认配置树（基于 `yacs.config.CfgNode`），并提供：

- `parse_qconfig(cfg_file)`：从 YAML 解析并合并到默认配置
- `verify_bits/verify_backend/verify_schedule`：对一些关键约束做 assert

它是 “点火钥匙” 的第一步：**决定后续所有量化器、观测器、导出后端的行为**。

## 2. 配置树结构（按重要度）

> 以下字段名来自默认配置 `_C`，YAML 中可覆盖。

### 2.1 全局

- `BACKEND`：`virtual | onnxruntime | tensorrt`
- `SKIP_TRACE_MODULES`：`[]`，模块名 pattern 列表（`fnmatch`），用于 FX trace/替换阶段的“跳过/视为 leaf”
- `SCHEDULE.*`：控制 fuse_operations 的开关（见 2.4）

### 2.2 权重量化 `W.*`

- `W.QSCHEME`：`per-channel-symmetric` 等（最终映射为 `torch.per_*`）
- `W.QUANTIZER.TYPE`：`uniform/LSQ/LSQ_plus/DoReFa/...`
- `W.QUANTIZER.DISABLE`：显式禁用量化（会触发 `Quantizer.set_fake_fused()`）
- `W.QUANTIZER.BIT`：bitwidth（要求 >=0）
- `W.OBSERVER.TYPE`：`MINMAX/MSE/PERCENTILE/KL_HISTOGRAM/ACIQ/...`
- `W.SPECIFIC`：针对特定 module 名的覆盖（见 2.5）

### 2.3 激活量化 `A.*`

- `A.QSCHEME` / `A.QUANTIZER.*` / `A.OBSERVER.*`：与 W 类似
- `A.OBSERVER.LAYOUT`：`NCHW | NLC`（影响 `QuantDescriptor` 的 `ch_axis`）
- `A.QADD.ENABLE_QUANT`：是否为 `QAdd` 这种多输入算子插 `QIdentity` 并对输入量化
- `A.SPECIFIC`：同 W

### 2.4 SCHEDULE（后处理 pass 的开关）

- `SCHEDULE.FUSE_BN`：是否启用 `fuse_bn` pass
- `SCHEDULE.BN_TUNING`：是否启用 BN tuning 模式（QuantModel 里会先禁止 fuse_bn）
- `SCHEDULE.DISABLE_UNNECESSARY_QUANT`：是否启用 `disable_unnecessary_quant` pass

这些 key 的名字需要与 `sparsebit/quantization/converters/fuse_operations/lists.py` 的任务文件名一致（上层通过 `getattr(config, task.upper(), True)` 读取）。

## 3. parse 流程

`parse_qconfig(cfg_file)` 做的事情很少但很关键：

1) `_parse_config(cfg_file, default_cfg=_C)`：
   - clone 默认配置
   - `DEVICE = cuda if available else cpu`
   - `merge_from_file(cfg_file)` 合并 YAML
2) 依次执行 verify：
   - `verify_bits`
   - `verify_backend`
   - `verify_schedule`

## 4. 关键校验逻辑

### 4.1 bitwidth

- `W.QUANTIZER.BIT >= 0`
- `A.QUANTIZER.BIT >= 0`

（注：注释提到 SPECIFIC 里的 bit 校验 TODO，尚未实现。）

### 4.2 backend 约束

`verify_backend` 的核心约束：

- 对 ORT/TRT：只支持 `bit=8`（否则建议用 `virtual` 做研究）
- 对 TRT：
  - 权重量化 scheme 必须是 `per_channel_symmetric`
  - 激活量化 scheme 必须是 `per_tensor_symmetric`

这些约束来自 `quant_tensor.trt_fake_quant()` 的实现：TRT 只支持对称量化，`zero_point` 必须为 0。

### 4.3 schedule 约束

`BN_TUNING=True` 时要求 W 的 qscheme 必须是 per-channel（symmetric 或 affine）。

## 5. SPECIFIC 的语义与格式（非常重要）

`SPECIFIC` 用于“按 module 名 pattern 覆盖量化配置”。在示例 YAML 中通常长这样：

```yaml
A:
  SPECIFIC: [{
    "*ln*": ["QUANTIZER.DISABLE", True],
    "softmax*": ["QUANTIZER.DISABLE", True],
    "lm_head": ["OBSERVER.TYPE", "ACIQ", "OBSERVER.ACIQ.DISTRIBUTION", "laplace"],
  }]
```

含义：

- `SPECIFIC` 是一个 list，当前实现只取 `SPECIFIC[0]`（一个 dict）
- dict 的 key 是 `fnmatch` pattern（与 FX module 的 `node.target` 名字匹配）
- dict 的 value 是 `yacs` 的 `merge_from_list` 参数（key1, value1, key2, value2,...）

使用位置：`QuantModel._build_quantizer()`，会为每个 QuantOpr 构造一个子配置并 merge SPECIFIC 覆盖。

## 6. OCaml type signature（接口投影）

```ocaml
type cfg

val parse_qconfig : string -> cfg

val verify_bits : cfg -> unit
val verify_backend : cfg -> unit
val verify_schedule : cfg -> cfg
```

## 7. 重写时的注意事项

- 当前配置是 `yacs` 的动态树结构；重写时建议先把配置“实体化”为强类型记录（否则错误很难提前暴露）。
- `SPECIFIC` 的格式（list 包 dict）比较别扭：建议在新实现里直接定义为 `(pattern * kv list) list`。
- `SCHEDULE.*` 的 key 名必须与 pass 文件名保持一致（这是一个隐式约定）。

