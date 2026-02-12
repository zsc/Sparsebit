# 01：`sparsebit/quantization/quant_model.py` —— QuantModel 总控

## 1. 角色定位

`QuantModel` 是量化主线的“点火总控”。它把一个普通 `nn.Module` 变成一个可以：

- 自动把算子替换为 QModule（`QuantOpr` 体系）
- 给每层挂 `Quantizer/Observer` 并做校准
- 做图层面的 simplify / fuse / disable quant pass
- 导出 QDQ-ONNX（并可选写入额外 bit 信息）

的模型包装器。

## 2. 关键公开方法（按常用程度）

- `__init__(model, config)`：完成 trace→simplify→替换→建量化器→后处理（fuse）。
- `prepare_calibration()`：进入校准准备状态（注册 hook 收集输入数据）。
- `calc_qparams(asym=False, w_quant=False, a_quant=False)`：执行 layerwise 校准，填充每层 quantizer 的 `scale/zero_point`。
- `init_QAT()`：PTQ 初始化 + 打开量化开关，作为 QAT 起点。
- `set_quant(w_quant=False, a_quant=False)`：全局打开/关闭权重/激活量化。
- `export_onnx(..., extra_info=False)`：导出 ONNX（QDQ），可选写入额外 bit 信息。
- `get_quantization_error(data, checker, is_async)`：用 ErrorProfiler 评估每层量化误差。
- `dump_mermaid(f)`：输出 mermaid 格式的图（调试/可视化）。

## 3. 初始化流水线（核心）

```text
QuantModel.__init__
  ├─ _trace(model)                 -> fx.GraphModule
  ├─ _run_simplifiers()            -> converters.simplify (pre-pass)
  ├─ _convert2quantmodule()        -> 图中算子替换为 QModule（QuantOpr 等）
  ├─ _build_quantizer()            -> 每个 QuantOpr 挂 quantizer/observer
  └─ _run_fuse_operations()        -> converters.fuse_operations (post-pass)
```

### 3.1 `_trace(model)`：如何 trace

- 使用 `QTracer(SKIP_TRACE_MODULES)`（定义在 `quant_tracer.py`）进行 FX trace。
- `QTracer.is_leaf_module()` 的策略是：
  - `torch.nn.*` 里的模块（且不是 `nn.Sequential`）默认视为 leaf（不展开内部）
  - 额外允许用户用 `SKIP_TRACE_MODULES`（支持 `fnmatch`）指定“强制 leaf”的模块路径

这一步的产物是一个 `fx.GraphModule`，后续所有 pass 都在它上面做。

### 3.2 `_run_simplifiers()`：pre-pass

调用 `converters.simplify(self.model)`，按顺序运行若干 simplifier（如去掉 `nn.Identity`、把部分 get_attr/method 调整为更好匹配的形式）。

> 这些 pass 的共同点：运行在“原生算子/模块”层级，还没替换为 QModule。

### 3.3 `_convert2quantmodule()`：核心替换逻辑

遍历 `fx.GraphModule.graph.nodes`，对不同 node.op 处理：

- `call_module`：用原模块实例 `org_module = named_modules[node.target]`
  - 若模块来自 `sparsebit.quantization.*`：认为已经是 QModule，跳过
  - 若模块名命中 `SKIP_TRACE_MODULES`：深拷贝原模块（保持 fp32 行为）
  - 否则按 `QMODULE_MAP[type(org_module)](org_module)` 替换
- `call_function`：按 `QMODULE_MAP[node.target](node, cfg)` 替换（functional/builtin）
- `call_method`：把 method string 映射到 `getattr(torch.Tensor, name)` 再查 `QMODULE_MAP`

替换方式：在图中插入一个新的 `call_module` 节点，并把旧节点的所有 uses 指向新节点，然后删除旧节点。

**关键依赖**：`QMODULE_MAP` 由 `sparsebit/quantization/modules/__init__.py` 中的 `@register_qmodule` 装饰器在 import 时填充。

### 3.4 `_build_quantizer()`：给每层挂 quantizer

分两类：

1) `QuantOpr`（单输入或内部 self.weight）  
   - 为每个 QuantOpr 构建 `weight_quantizer`（如果有 weight）与 `input_quantizer`
   - 支持 `SPECIFIC`：按 `fnmatch(module_name, pattern)` 匹配后，对 W/A 子配置做 `merge_from_list`

2) `MultipleInputsQuantOpr`（多输入，如 `QAdd`）  
   - 若该 node 的输入数 > 1：会调用 `module.prepare_input_quantizer(node, model)`
   - 该方法会在每个输入前插一个 `QIdentity` 节点，从而让“每个输入”都走一次 `input_quantizer`
   - 随后给这些插入的 `QIdentity` 构建激活量化器

### 3.5 `_run_fuse_operations()`：post-pass

调用 `converters.fuse_operations(self.model, cfg.SCHEDULE)`：

- 典型包括：
  - `fuse_bn`：conv/linear 与 bn 融合
  - `disable_unnecessary_quant`：遇到某些结构关闭后续层量化（`set_fake_fused`）
- 若 `BN_TUNING=True`：会先强制关闭 `FUSE_BN`（避免先 fuse 掉 BN）

> 这些 pass 的共同点：运行在“QModule/QuantOpr”层级，可能修改 quantizer 状态或参数。

## 4. BN Tuning（批归一化微调）

`batchnorm_tuning()` 是一个 contextmanager：

1) `train()` 模式 + 打开量化  
2) 对 `QBatchNorm2d` 重置 `num_batches_tracked`  
3) 用户在 `with` 块里跑若干数据 forward 更新 BN 统计  
4) 退出后 `eval()`，并以 `custom_fuse_list=["fuse_bn"]` 执行 BN 融合  
5) 最后关闭量化

## 5. ONNX 导出与 extra 信息

### 5.1 `export_onnx()`

- 强制 `set_quant(True, True)`，并对每个 `Quantizer` 执行 `enable_export_onnx()`
  - 导出时 quantizer 走 `torch_fake_quant` 路径以产生 ONNX QDQ 节点
- 如果存在 `bit != 8` 的 quantizer：要求 `extra_info=True`（否则直接 assert）

### 5.2 `add_extra_info_to_onnx()`

这是一个基于“位置对齐”的后处理：

- 读取 ONNX graph，建立 tensor→producer/consumer 索引
- 线性扫描 ONNX node，跳过 `QuantizeLinear/DequantizeLinear/Constant` 等节点
- 同时线性扫描 `self.model.named_modules()`（跳过 `Observer/Quantizer/QIdentity/Clone`）
- 当遇到一个 `QuantOpr` 且其 quantizer enable 时：
  - 找到其输入/权重对应的 Q/DQ 节点，并在 attribute 中写入 `bits`

**注意**：该对齐策略依赖导出图的稳定性；如果 ONNX 图结构变化（优化、常量折叠、算子展开），对齐可能失效。

## 6. OCaml type signature（用于重写时的“接口投影”）

> 这里的 OCaml 签名是对 Python 行为的“抽象接口描述”，便于重写时先对齐边界，而不是逐行翻译实现。

```ocaml
(* 抽象类型：在重写时可替换为更明确的记录/模块 *)
type cfg
type device
type fx_graph_module
type tensor
type onnx_path = string

type quant_model

val create : model:'a -> cfg:cfg -> quant_model
val forward : quant_model -> 'input -> 'output

val prepare_calibration : quant_model -> unit
val calc_qparams :
  quant_model ->
  ?asym:bool ->
  ?w_quant:bool ->
  ?a_quant:bool ->
  unit

val init_qat : quant_model -> unit
val set_quant : quant_model -> w_quant:bool -> a_quant:bool -> unit

val export_onnx :
  quant_model ->
  dummy_data:tensor ->
  name:onnx_path ->
  ?input_names:string list ->
  ?output_names:string list ->
  ?dynamic_axes:(string, int list) Hashtbl.t ->
  ?opset_version:int ->
  ?verbose:bool ->
  ?extra_info:bool ->
  unit

val dump_mermaid : quant_model -> out_channel -> unit
```

## 7. 与其它文件的交叉引用

- Quant 配置：`claude_doc/02_quant_config.md`
- QuantOpr/多输入插 QIdentity：`claude_doc/03_quant_modules_base.md`
- Calibration 实现：`claude_doc/07_calibration_runner.md`、`claude_doc/08_graph_wrapper.md`
- 图 pass 引擎：`claude_doc/09_subgraph_matching.md`

