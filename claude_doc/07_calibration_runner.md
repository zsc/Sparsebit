# 07：`sparsebit/quantization/tools/calibration.py` —— Layerwise CalibrationRunner

## 1. 角色定位

`CalibrationRunner` 实现了 Sparsebit 的 PTQ/QAT 初始化核心算法：**按 FX 图的拓扑顺序逐层校准**。

它解决的问题是：

- 对每个 `QuantOpr` 的输入/权重量化器，收集观测数据并计算 `scale/zp`
- 在校准过程中尽量少占显存（通过 `SharedData.finish_node` 释放中间结果）
- 支持 `asym=True` 的“前置层量化”校准模式
- 对 `AdaRound` 提供重建入口（`reconstruct_qlayer`）

## 2. prepare_calibration：如何收集输入数据

`prepare_calibration()` 的目标：**把“模型输入 placeholder 的实际张量”缓存下来**，以便后续对第一层/接输入的层做观测。

实现方式：

- 先找出 FX 图里的 placeholder 名称集合 `input_names_cache`
- 构造一个 `hook_wrapper(node, module, storage)`：
  - 若该 node 的输入包含 placeholder，则给 module 注册 forward_hook
  - hook 会把 `x_in` 中与 placeholder 对应的那一部分 tensor 存到 `storage.outputs[placeholder_name]`
  - 每个 placeholder 只缓存一次（命中后从 `input_names_cache` 移除）
- 用 `GraphVisitor(self.model, hook_wrapper)` 批量注册 hooks

## 3. layerwise_calibration：主流程

关键结构：

```text
for node in fx.graph.nodes (topo order):
  1) run_feature_calibration(node)
  2) float_outputs = module_forward(...)
     storage.set_output(node.target, float_outputs)
  3) run_weight_calibration(node)
  4) (optional asym)
     quant_outputs = module_forward(..., asym=True, w_quant, a_quant)
     qstorage.set_output(node.target, quant_outputs)
  5) storage.finish_node(node.target)  # 释放 out-degree=0 的缓存
```

### 3.1 `run_feature_calibration(node)`

对 `QuantOpr.input_quantizer`：

- 遍历所有输入节点 `inp_node`：
  - 从 `storage.get_output(inp_node.target)` 取出缓存的输入张量列表
  - 对每个张量调用 `input_quantizer.update_observer(t)`
- 调用 `input_quantizer.calc_qparams()` 得到 scale/zp
- `observer.data_cache.reset()`

### 3.2 `run_weight_calibration(node)`

对 `QuantOpr.weight_quantizer`：

- `update_observer(module.weight)`
- `calc_qparams()`
- 若 quantizer 类型是 `adaround`：
  - 要求该算子只有一个输入
  - 取输入/输出 tensor（拼成 batch），调用 `reconstruct_qlayer(...)` 做重建

### 3.3 `module_forward(...)`

- 对 `call_module`：`module.eval()`
- 若 `asym=True` 且 module 是 `QuantOpr`：临时 `module.set_quant(w_quant, a_quant)`
- 对每个 batch：
  - 从 storage 里把 `node.args/kwargs` 解引用成真实 tensor（支持嵌套结构）
  - `to_device` 搬到目标 device
  - `with torch.no_grad(): outputs.append(to_cpu(module(*args, **kwargs)))`
- 最后如果是 `QuantOpr`：恢复 `set_quant(False, False)`

对 `get_attr`（常量/参数）节点：直接读 `module.data`。

## 4. OCaml type signature（接口投影）

```ocaml
type device
type tensor
type fx_graph_module

type calibration_runner
val create : fx_graph_module -> calibration_runner

val prepare_calibration : calibration_runner -> unit

val layerwise_calibration :
  calibration_runner ->
  device:device ->
  asym:bool ->
  w_quant:bool ->
  a_quant:bool ->
  unit
```

## 5. 重写时的注意事项

- 当前实现把输出缓存成 “每个 node 一个 list[Tensor]（按 batch）”，便于逐层运行但对长序列模型会较重；重写时可考虑 streaming/分块策略。
- `asym=True` 模式会复制一份 storage（`qstorage = deepcopy(storage)`），成本较高；需要明确其收益与适用场景。
