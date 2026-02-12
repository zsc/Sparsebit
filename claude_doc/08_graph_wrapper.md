# 08：`sparsebit/quantization/tools/graph_wrapper.py` —— GraphVisitor 与 SharedData

## 1. 角色定位

这是 quantization 工具链的“执行/缓存引擎”，主要给：

- `CalibrationRunner`
- `QuantizationErrorProfiler`

提供统一的 FX 图遍历、hook 注册与中间结果管理能力。

## 2. fx_symbolic_trace

`fx_symbolic_trace(model)`：如果 `model` 没有 `graph` 属性，就 `fx.symbolic_trace(model)`；否则认为已经是 GraphModule。

这允许上层既能传 `nn.Module`，也能传 `fx.GraphModule`。

## 3. SharedData：中间结果与依赖管理

核心字段：

- `outputs : {name -> value}`：缓存某个 node 的输出（可为 list[Tensor]）
- `edges : {name -> [input_name...]}`：记录依赖边（输入来自哪些节点）
- `output_degrees : {name -> out_degree}`：用于判断何时可以释放某个输出
- `values : {value_name -> {name -> value}}`：缓存额外指标（例如 diff）

### 3.1 `add_node(name, inputs)`

记录依赖关系，并对每个输入节点累计 out-degree。

### 3.2 `finish_node(name)`：释放策略

对 `edges[name]` 中的每个输入 `inp`：

- `output_degrees[inp] -= 1`
- 若归零：删除 `inp` 的 degree，并 `outputs.pop(inp, None)`

对 `name` 本身：

- 若 `output_degrees[name]==0`：也会释放自身输出

这使得上层可以像“拓扑顺序执行解释器”一样，边跑边释放不再使用的中间结果。

### 3.3 `extract_node_args/kwargs`

把 FX 的 `node.args` / `node.kwargs` 中出现的 `fx.Node` 替换为缓存的真实输出：

- 如果缓存里没有（或不是 Tensor 且为空），则回退到 hook 里提供的 `real_input`
- 支持 tuple/list/dict 的递归解包
- 支持 `batch` 选择（用于 calibration 逐 batch 前向）

## 4. GraphVisitor：批量注册 hook

构造时传入：

- `model : fx.GraphModule`
- `hook_wrapper(node, module, storage) -> handles`

`build()` 会按 FX 图顺序遍历：

- 跳过 `placeholder/output`
- 对 `get_attr`：
  - `module = getattr(model, node.target)`（常量/参数）
- 对普通 `call_module`：
  - `module = named_modules[node.target]`
- 调用 `storage.add_node(node.target, input_node_targets)`
- 调用 `hook_wrapper`，收集 handle（并在析构时 remove）

## 5. OCaml type signature（接口投影）

```ocaml
type tensor
type fx_node
type fx_graph_module

type shared_data
val add_node : shared_data -> name:string -> inputs:string list -> unit
val set_output : shared_data -> name:string -> 'a -> unit
val get_output : shared_data -> name:string -> 'a option
val finish_node : shared_data -> name:string -> unit
val set_value : shared_data -> name:string -> value_name:string -> 'a -> unit
val extract_value : shared_data -> value_name:string -> (string, 'a) Hashtbl.t

type graph_visitor
val create :
  fx_graph_module ->
  hook_wrapper:(fx_node -> 'module -> shared_data -> 'handle list) ->
  graph_visitor
```

## 6. 重写时的注意事项

- `SharedData.__del__` 里有强 assert（要求 outputs/values 清空），这在 Python 的 GC 语义下不总是可靠；重写时建议显式 close/release。
- `extract_node_args/kwargs` 的“回退 real_input”是为了解决 hook 时拿不到缓存输入的情况；新实现可考虑更明确的输入绑定方式。

