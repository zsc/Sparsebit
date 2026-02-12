# QTracer - FX 量化追踪器

## 文件信息

- **路径**: `sparsebit/quantization/quant_tracer.py`
- **核心类**: `QTracer`
- **继承**: `torch.fx.Tracer`

---

## OCaml Type Signature

```ocaml
type qtracer = {
  skipped_module_names : string list;  (* 需要跳过的模块名列表 *)
}

class qtracer : string list ->
  inherit torch.fx.Tracer
  method is_leaf_module : torch.nn.Module -> string -> bool
  method _probe : string -> string list -> bool
end
```

---

## 详细功能说明

### QTracer 类

`QTracer` 是 PyTorch FX `Tracer` 的**自定义子类**，用于：
1. 控制哪些模块应该被视为**叶子节点**（不再展开内部结构）
2. 支持通过配置跳过特定模块的追踪

### FX 追踪基础

```
PyTorch FX 追踪原理:

原始模型                    FX Graph
┌──────────┐               ┌─────────────────────────┐
│ Conv2d   │               │ placeholder (input)     │
│   ↓      │    trace      │     ↓                   │
│ BatchNorm│   ───────→    │ call_module (conv1)     │
│   ↓      │               │     ↓                   │
│  ReLU    │               │ call_module (bn1)       │
│   ↓      │               │     ↓                   │
│ Linear   │               │ call_module (relu1)     │
└──────────┘               │     ↓                   │
                           │ call_module (fc1)       │
                           │     ↓                   │
                           │ output                  │
                           └─────────────────────────┘
```

### 叶子节点 (Leaf Module)

叶子节点是**追踪的边界**：
- 叶子节点的内部结构不会被展开
- 在图中表示为单个 `call_module` 节点
- 普通 `nn.Module` 会被递归展开，直到遇到叶子节点

**默认叶子节点**（PyTorch FX）:
- `torch.nn` 模块（除了 `Sequential`）

---

## 核心方法

### `__init__(self, skipped_module_names: List[str])`

初始化追踪器。

**参数**:
- `skipped_module_names`: 需要视为叶子的模块名列表（支持通配符 `*`）

```python
def __init__(self, skipped_module_names):
    super().__init__()
    self.skipped_module_names = skipped_module_names
```

---

### `_probe(self, module_name: str, patterns: List[str]) -> bool`

检查模块名是否匹配任何跳过模式。

```python
def _probe(self, module_name, patterns):
    for p in patterns:
        if fnmatch(module_name, p):  # 支持通配符匹配
            return True
    return False
```

**示例**:
```python
_probe("backbone.layer1.0.conv1", ["backbone*"])  # True
_probe("head.fc", ["head*", "neck*"])             # True
_probe("conv1", ["backbone*"])                    # False
```

---

### `is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool`

**核心方法**，决定模块是否为叶子节点。

```python
def is_leaf_module(self, m, module_qualified_name):
    return (
        # 条件 1: PyTorch nn 模块（除了 Sequential）
        m.__module__.startswith("torch.nn") 
        and not isinstance(m, torch.nn.Sequential)
    ) or (
        # 条件 2: 用户配置的跳过模块
        self._probe(module_qualified_name, self.skipped_module_names)
    )
```

**逻辑**:
1. 如果模块是 `torch.nn` 下的标准模块（且不是 `Sequential`）→ 叶子
2. 如果模块名匹配 `skipped_module_names` → 叶子
3. 否则 → 继续展开内部

---

## 使用示例

### 基本使用

```python
from sparsebit.quantization.quant_tracer import QTracer

# 创建追踪器，跳过 head 和 neck 模块
tracer = QTracer(["head*", "neck*"])

# 追踪模型
graph = tracer.trace(model)

# 创建 GraphModule
traced_model = torch.fx.GraphModule(tracer.root, graph, "traced_model")
```

### 在 QuantModel 中的使用

```python
# quant_model.py
def _trace(self, model):
    skipped_modules = self.cfg.SKIP_TRACE_MODULES
    tracer = QTracer(skipped_modules)
    graph = tracer.trace(model)
    name = model.__class__.__name__
    traced = fx.GraphModule(tracer.root, graph, name)
    return traced
```

---

## 追踪结果示例

### 配置

```yaml
SKIP_TRACE_MODULES: ["backbone*"]
```

### 原始模型

```python
class Model(nn.Module):
    def __init__(self):
        self.backbone = ResNet50()  # 会被视为叶子
        self.head = nn.Linear(2048, 1000)  # 会展开
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x
```

### 追踪后的 Graph

```
# 不使用 QTracer (SKIP_TRACE_MODULES: [])
graph:
    %x : [#users=1] = placeholder[target=input]
    %conv1 : [#users=1] = call_module[target=backbone.conv1](args=(%x,), kwargs={})
    %bn1 : [#users=1] = call_module[target=backbone.bn1](args=(%conv1,), kwargs={})
    %relu : [#users=1] = call_module[target=backbone.relu](args=(%bn1,), kwargs={})
    ... (ResNet50 所有层都被展开)
    %fc : [#users=1] = call_module[target=head](args=(%flatten,), kwargs={})
    return fc

# 使用 QTracer (SKIP_TRACE_MODULES: ["backbone*"])
graph:
    %x : [#users=1] = placeholder[target=input]
    %backbone : [#users=1] = call_module[target=backbone](args=(%x,), kwargs={})  # 整体
    %head : [#users=1] = call_module[target=head](args=(%backbone,), kwargs={})
    return head
```

---

## 依赖关系

```
QTracer
    └── torch.fx.Tracer (PyTorch FX)
```

---

## 相关概念

| 概念 | 说明 |
|------|------|
| `placeholder` | 输入节点 |
| `call_module` | 调用模块 |
| `call_function` | 调用函数（如 `torch.add`, `F.relu`） |
| `call_method` | 调用方法（如 `tensor.view`） |
| `get_attr` | 获取属性（如 `self.weight`） |
| `output` | 输出节点 |
