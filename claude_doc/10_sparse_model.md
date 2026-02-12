# 10：`sparsebit/sparse/sparse_model.py` —— SparseModel（剪枝/稀疏化）总控

## 1. 角色定位

`SparseModel` 是 sparse/pruning 主线的“总控”，功能与 `QuantModel` 类似但更原型化：

- 把原模型 `fx.symbolic_trace` 成 FX 图
- 跑一遍 `simplify`（直接复用了 quantization 的 simplifier，源码里标了 `FIXME`）
- 把部分 `call_module` 替换成 SparseOpr（如 `SConv2d`）
- 为每个 SparseOpr 构建 `Sparser`，并能计算 mask

当前 sparse 子系统整体较轻量，功能明显少于 quantization。

## 2. 初始化流程

```text
SparseModel.__init__
  ├─ fx.symbolic_trace(model)
  ├─ _run_simplifiers()         -> converters.simplify (FIXME: 复用 quant 的)
  ├─ _convert2sparsemodule()    -> 用 SMODULE_MAP 替换部分 module
  └─ _build_sparser()           -> 每个 SparseOpr 挂 sparser
```

## 3. `_convert2sparsemodule()`：替换规则

遍历 FX node：

- 只处理 `call_module`：
  - 若 `type(named_modules[n.target]) in SMODULE_MAP`：
    - `new_module = SMODULE_MAP[type(org_module)](org_module)`
  - 否则直接复用原 module（不做替换）
- 其他 node.op（call_function/method/placeholder/get_attr/output）跳过

替换方式与 QuantModel 类似：插入新 module 和新 `call_module` 节点，重连 uses，删除旧节点，然后 recompile。

## 4. `_build_sparser()`：给每层挂 sparser

遍历 `self.model.named_modules()`：

- 若 `isinstance(m, SparseOpr)`：
  - `_config = self.config.clone()`
  - `m.build_sparser(_config)`

SparseOpr 的 `build_sparser` 会调用 `sparsebit/sparse/sparsers/__init__.py` 里的 `build_sparser`，按 `config.SPARSER.STRATEGY` 选择实现（目前主要是 `l1norm`）。

## 5. `disable_sparse_before_add()`：残差加法前禁用稀疏

该函数扫描 FX 图里所有 add（`operator.add` / `torch.add`）节点，并沿输入向前回溯：

- 若遇到带 `sparser` 的 module：把 `ratio` 设置为 `0.0`（禁用稀疏）
- 回溯时如果遇到 `SConv2d` 会停止继续向前扩展（只对其本身做处理）

直觉：避免 residual 分支在 add 前出现结构不一致的稀疏化导致数值/形状问题（但该策略非常 heuristic）。

## 6. `calc_params()`：计算 mask（当前实现较薄）

按 FX 节点顺序：

- 若 node 是 `call_module` 且 module 是 `SparseOpr` 且有 sparser：
  - `pre_mask = module.calc_mask(pre_mask)`

目前 `pre_mask` 的传播语义并不完善（大多数 SparseOpr 的 `calc_mask` 并未使用 pre_mask）。

## 7. OCaml type signature（接口投影）

```ocaml
type cfg
type device
type tensor
type fx_graph_module

type sparse_model

val create : model:'a -> cfg:cfg -> sparse_model
val forward : sparse_model -> 'input -> 'output
val calc_params : sparse_model -> unit
val disable_sparse_before_add : sparse_model -> unit

val export_onnx :
  sparse_model ->
  dummy_data:tensor ->
  name:string ->
  ?input_names:string list ->
  ?output_names:string list ->
  ?dynamic_axes:(string, int list) Hashtbl.t ->
  ?opset_version:int ->
  ?verbose:bool ->
  unit
```

## 8. 重写时的注意事项

- sparse 子系统目前复用了 quantization 的 simplifier（并标了 FIXME），重写时应把两条主线的 IR/pass 关系理清：哪些 pass 可共享，哪些必须分离。
- 当前只替换 `call_module`，不处理 `call_function/call_method`，因此能力有限；如果想做更完整的结构化剪枝，IR 层需要更统一的算子抽象。

