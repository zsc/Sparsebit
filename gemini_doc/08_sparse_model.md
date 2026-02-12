# SparseModel - 剪枝模型主类

## 文件信息

- **路径**: `sparsebit/sparse/sparse_model.py`
- **核心类**: `SparseModel`
- **继承**: `torch.nn.Module`

---

## OCaml Type Signature

```ocaml
type sparse_model = {
  model : fx.GraphModule;
  config : config;
  device : torch.device;
}

val create : torch.nn.Module -> config -> sparse_model
val convert : sparse_model -> unit
val build_sparsers : sparse_model -> unit
val calc_params : sparse_model -> torch.mask option -> unit
val disable_sparse_before_add : sparse_model -> unit
val export_onnx : sparse_model -> tensor -> string -> unit
```

---

## 详细功能说明

### SparseModel 类

`SparseModel` 是剪枝功能的**核心入口**。与 `QuantModel` 类似，它：
1. 使用 FX 追踪模型
2. 将特定算子替换为 `SparseOpr`
3. 附加 `Sparser` 计算剪枝掩码
4. 支持结构化/非结构化剪枝

### 与 QuantModel 的对比

| 特性 | QuantModel | SparseModel |
|------|-----------|-------------|
| 核心操作 | 量化 | 剪枝 |
| 基类 | `QuantOpr` | `SparseOpr` |
| 附加组件 | `Quantizer` + `Observer` | `Sparser` |
| 输出 | QDQ-ONNX | 剪枝后的 ONNX |
| 校准 | 需要收集统计信息 | 计算掩码即可 |

---

## 核心方法

### `__init__(self, model: nn.Module, config)`

初始化剪枝模型。

**流程**:
```python
def __init__(self, model, config):
    super().__init__()
    self.model = model
    self.config = config
    self.device = torch.device(config.DEVICE)
    
    # 1. FX 追踪
    self.model = fx.symbolic_trace(model)
    
    # 2. 图简化
    self._run_simplifiers()
    
    # 3. 转换为 SparseOpr
    self._convert2sparsemodule()
    
    # 4. 构建 Sparser
    self._build_sparser()
```

---

### `_convert2sparsemodule(self)`

将普通算子转换为剪枝算子。

**实现逻辑**:
```python
def _convert2sparsemodule(self):
    named_modules = dict(self.model.named_modules(remove_duplicate=False))
    traced = self.model
    snodes = []  # 避免重复遍历
    
    for n in traced.graph.nodes:
        if not isinstance(n, fx.Node) or n in snodes:
            continue
        elif n.op == "call_module":
            # 只替换在 SMODULE_MAP 中的模块
            if type(named_modules[n.target]) in SMODULE_MAP:
                org_module = named_modules[n.target]
                new_module = SMODULE_MAP[type(org_module)](org_module)
            else:
                continue  # 跳过不在映射中的模块
        elif n.op in ["call_function", "call_method", ...]:
            continue  # 剪枝目前只处理 call_module
        
        # 替换节点
        with traced.graph.inserting_after(n):
            traced.add_module(n.name, new_module)
            new_node = traced.graph.call_module(n.name, n.args, n.kwargs)
            snodes.append(new_node)
            n.replace_all_uses_with(new_node)
            traced.graph.erase_node(n)
    
    traced.recompile()
    self.model = fx.GraphModule(traced, traced.graph)
```

**注意**: 与 `QuantModel` 不同，`SparseModel` 只替换在 `SMODULE_MAP` 中显式注册的模块类型。

---

### `_build_sparser(self)`

为每个 `SparseOpr` 构建 `Sparser`。

```python
def _build_sparser(self):
    for n, m in self.model.named_modules():
        if isinstance(m, SparseOpr):
            _config = self.config.clone()
            m.build_sparser(_config)
```

---

### `calc_params(self)`

计算剪枝参数（掩码）。

```python
def calc_params(self):
    pre_mask = None
    for node in self.model.graph.nodes:
        if node.op == "call_module":
            module = getattr(self.model, node.target, None)
            if isinstance(module, SparseOpr) and getattr(module, "sparser", None):
                # 传播掩码：前一层的输出掩码作为当前层的输入掩码
                pre_mask = module.calc_mask(pre_mask)
```

**掩码传播**:
```
Conv1 (out_channels=64)  ──→  Conv2 (in_channels=64)
       │                           │
       ▼                           ▼
   mask1 [64]    ───────────→  影响 Conv2 的输入掩码
                               同时 Conv2 输出 mask2
```

---

### `disable_sparse_before_add(self)`

在 Add 操作之前禁用剪枝。

**原因**: 如果 Add 的两个输入分支都经过剪枝，且剪枝比例/模式不同，会导致维度不匹配。

```python
def disable_sparse_before_add(self):
    named_modules = dict(self.model.named_modules())
    
    # 找到所有 Add 节点
    add_nodes = [
        n for n in self.model.graph.nodes
        if n.op == "call_function" and n.target in [operator.add, torch.add]
    ]
    
    for add_node in add_nodes:
        # 获取 Add 的输入节点
        add_inputs = [a for a in add_node.args if isinstance(a, torch.fx.Node)]
        
        while len(add_inputs) > 0:
            n = add_inputs.pop()
            if n.op == "call_module" and n.target in named_modules:
                m = named_modules[n.target]
            else:
                m = None
            
            # 如果有 sparser，设置比例为 0（即不剪枝）
            if hasattr(m, "sparser") and m.sparser:
                m.sparser.set_ratio(0.0)
            
            # 继续向上游传播（除非是 Conv，通常只影响一层）
            if not isinstance(m, SConv2d):
                n_list = [a for a in n.args if isinstance(a, torch.fx.Node)]
                add_inputs.extend(n_list)
```

---

## 依赖关系

```
SparseModel
    ├── simplify (quantization/converters/simplifiers)  # 复用简化器
    ├── SparseOpr (sparse/modules/base.py)
    ├── SMODULE_MAP (sparse/modules/__init__.py)
    └── build_sparser (sparse/sparsers/__init__.py)
```

---

## 使用示例

```python
from sparsebit.sparse import SparseModel
from sparsebit.sparse.sparse_config import parse_sconfig

# 1. 解析配置
config = parse_sconfig("sparse_config.yaml")

# 2. 创建剪枝模型
model = ...  # 原始模型
smodel = SparseModel(model, config)

# 3. 可选：在 Add 前禁用剪枝
smodel.disable_sparse_before_add()

# 4. 计算剪枝掩码
smodel.calc_params()

# 5. 导出 ONNX
smodel.export_onnx(dummy_data, "pruned_model.onnx")
```

---

## 剪枝配置示例

```yaml
# sparse_config.yaml
DEVICE: "cuda"
SPARSER:
  TYPE: "l1norm"        # 剪枝算法: l1norm, l0norm, fisher, etc.
  STRATEGY: "structured" # 结构化/非结构化
  RATIO: 0.5            # 剪枝比例
```
