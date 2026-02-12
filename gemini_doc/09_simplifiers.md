# Simplifiers - 图简化器

## 文件信息

- **路径**: `sparsebit/quantization/converters/simplifiers/__init__.py`
- **核心函数**: `simplify`

---

## OCaml Type Signature

```ocaml
type graph_module = torch.fx.GraphModule

type simplifier = {
  apply : graph_module -> unit;
}

val simplify : graph_module -> graph_module
val fx_symbolic_trace : torch.nn.Module -> graph_module
```

---

## 详细功能说明

### simplify 函数

`simplify` 函数是**图简化**的入口，它：
1. 对模型进行符号化追踪
2. 执行剪枝（移除死节点）
3. 依次应用各种简化模式

```python
def simplify(model: torch.fx.GraphModule):
    model = fx_symbolic_trace(model)
    model = PruneGraph().apply(model)
    
    for task in lists:
        module = importlib.import_module(".{}".format(task), package=__package__)
        module.ReplacePattern().apply(model)
        model = PruneGraph().apply(model)
    
    return model
```

---

## 简化器列表 (lists.py)

```python
lists = [
    "remove_identity",              # 移除无操作
    "unbind_getitem_to_subtensor",  # 优化 unbind+getitem
    "getattr_to_shape",             # 转换 getattr 到 shape 操作
]
```

---

## 各简化器详解

### 1. RemoveIdentity - 移除 Identity 操作

**文件**: `remove_identity.py`

**作用**: 移除无操作的 `identity` 节点。

```
转换前:              转换后:
    A                    A
    │                    │
    ▼                    ▼
┌─────────┐          ┌─────┐
│Identity │    →     │ Op  │
└────┬────┘          └─────┘
     │
     ▼
   ┌─────┐
   │ Op  │
   └─────┘
```

**实现逻辑**:
```python
class ReplacePattern:
    def apply(self, model: torch.fx.GraphModule):
        for node in list(model.graph.nodes):
            if node.op == "call_function" and node.target == torch.clone:
                # 将所有使用 node 的地方替换为 node 的输入
                node.replace_all_uses_with(node.args[0])
                model.graph.erase_node(node)
        model.recompile()
```

---

### 2. UnbindGetitemToSubtensor - 优化 Unbind+GetItem

**文件**: `unbind_getitem_to_subtensor.py`

**作用**: 将 `unbind` + `getitem` 模式优化为直接的索引操作。

```python
# 原始代码
x = torch.unbind(tensor, dim=0)  # 返回 tuple
y = x[0]  # getitem

# 优化后
y = tensor[0]  # 直接索引
```

**好处**: 避免创建中间 tuple，更高效。

---

### 3. GetAttrToShape - 转换 GetAttr 到 Shape

**文件**: `getattr_to_shape.py`

**作用**: 将 `getattr` 操作（获取张量形状）转换为专门的 shape 操作。

---

## PruneGraph - 死节点剪枝

**文件**: `sparsebit/quantization/converters/prune.py`

**作用**: 移除图中没有输出的节点（死代码消除）。

```python
class PruneGraph:
    def apply(self, model: torch.fx.GraphModule):
        # 找到所有没有用户的节点（除了 output 节点）
        for node in list(model.graph.nodes):
            if len(node.users) == 0 and node.op != "output":
                model.graph.erase_node(node)
        model.recompile()
        return model
```

---

## 简化流程

```
输入模型
    │
    ▼
┌─────────────────┐
│ fx_symbolic_trace│  # 重新追踪，获得干净图
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PruneGraph    │  # 移除死节点
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        依次应用简化器               │
│  ┌─────────────────────────────┐    │
│  │ RemoveIdentity              │    │
│  │ UnbindGetitemToSubtensor    │    │
│  │ GetAttrToShape              │    │
│  └─────────────────────────────┘    │
│           │                         │
│           ▼                         │
│      PruneGraph（每次后）           │
└─────────────────────────────────────┘
         │
         ▼
    简化后的模型
```

---

## 依赖关系

```
simplify
    ├── fx_symbolic_trace (tools/graph_wrapper.py)
    ├── PruneGraph (converters/prune.py)
    └── 各简化器模块
```
