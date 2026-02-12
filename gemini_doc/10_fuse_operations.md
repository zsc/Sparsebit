# Fuse Operations - 算子融合

## 文件信息

- **路径**: `sparsebit/quantization/converters/fuse_operations/__init__.py`
- **核心函数**: `fuse_operations`

---

## OCaml Type Signature

```ocaml
type graph_module = torch.fx.GraphModule
type config = CfgNode

type fuse_operation = {
  apply : graph_module -> unit;
}

val fuse_operations : graph_module -> config -> ?custom_lists:string list -> graph_module
```

---

## 详细功能说明

### fuse_operations 函数

`fuse_operations` 是**算子融合**的入口，它将多个连续的小算子合并为一个大算子，以：
1. 减少计算图节点数
2. 提高推理效率
3. 减少量化误差累积

```python
def fuse_operations(model, config, custom_lists=None):
    model = fx_symbolic_trace(model)
    model = PruneGraph().apply(model)
    
    cur_list = custom_lists if custom_lists else default_lists
    
    for task in cur_list:
        if getattr(config, task.upper(), True):  # 检查配置是否启用
            module = importlib.import_module(".{}".format(task), package=__package__)
            
            if getattr(module, "ReplacePatterns", None):
                # 多个替换模式
                classes = module.ReplacePatterns
                for cls in classes:
                    cls.apply(model)
            else:
                # 单个替换模式
                func = module.ReplacePattern
                func().apply(model)
            
            model = PruneGraph().apply(model)
    
    return model
```

---

## 融合操作列表

```python
default_lists = [
    "fuse_bn",                      # 融合 Conv/Linear + BN
    "disable_unnecessary_quant",    # 禁用不必要的量化
]
```

---

## 各融合操作详解

### 1. Fuse BN - 融合 BatchNorm

**文件**: `fuse_bn.py`

**作用**: 将 `Conv2d + BatchNorm2d` 或 `Linear + BatchNorm1d` 融合为单个层。

**融合公式**:

```
BN(x) = γ * (x - μ) / √(σ² + ε) + β

Conv(x) = W * x + b

融合后:
Conv_fused(x) = W' * x + b'

其中:
W' = W * γ / √(σ² + ε)
b' = (b - μ) * γ / √(σ² + ε) + β   (如果 b 存在)
b' = -μ * γ / √(σ² + ε) + β       (如果 b 为 None)
```

**代码实现**:
```python
class ReplacePattern:
    def apply(self, model):
        # 匹配 Conv-BN 或 Linear-BN 模式
        for node in model.graph.nodes:
            if node.op == "call_module":
                module = get_module(model, node.target)
                
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    # 检查下一个节点是否为 BN
                    if len(node.users) == 1:
                        next_node = list(node.users)[0]
                        next_module = get_module(model, next_node.target)
                        
                        if isinstance(next_module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                            # 执行融合
                            self.fuse_conv_bn(model, node, next_node, module, next_module)
    
    def fuse_conv_bn(self, model, conv_node, bn_node, conv, bn):
        # 计算融合后的权重和偏置
        var_sqrt = torch.sqrt(bn.running_var + bn.eps)
        weight_new = conv.weight * (bn.weight / var_sqrt).reshape([-1] + [1] * (conv.weight.ndim - 1))
        
        if conv.bias is not None:
            bias_new = (conv.bias - bn.running_mean) * bn.weight / var_sqrt + bn.bias
        else:
            bias_new = -bn.running_mean * bn.weight / var_sqrt + bn.bias
        
        # 更新 Conv 参数
        conv.weight = nn.Parameter(weight_new)
        conv.bias = nn.Parameter(bias_new)
        
        # 替换 BN 节点为 Conv 节点
        bn_node.replace_all_uses_with(conv_node)
        model.graph.erase_node(bn_node)
        model.recompile()
```

**融合效果**:
```
转换前:                    转换后:
    x                          x
    │                          │
    ▼                          ▼
┌─────────┐              ┌─────────┐
│ Conv2d  │              │ Conv2d  │  (融合 BN 参数)
└────┬────┘              │ (fused) │
     │                   └────┬────┘
     ▼                        │
┌───────────┐                 │
│BatchNorm2d│                 │
└─────┬─────┘                 │
      │                       │
      ▼                       ▼
     out                    out
```

---

### 2. Disable Unnecessary Quant - 禁用不必要的量化

**文件**: `disable_unnecessary_quant.py`

**作用**: 在某些特定模式下禁用量化，避免冗余的 QDQ 操作。

**典型场景**:
- 量化节点后紧跟另一个量化节点（精度相同）
- 某些操作不需要量化（如 shape 操作）

---

## 融合流程

```
输入模型
    │
    ▼
┌─────────────────┐
│ fx_symbolic_trace│  # 重新追踪
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   PruneGraph    │  # 移除死节点
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│        依次应用融合操作             │
│  ┌─────────────────────────────┐    │
│  │ Fuse BN                     │    │
│  │ - Conv+BN → Conv            │    │
│  │ - Linear+BN → Linear        │    │
│  └─────────────────────────────┘    │
│           │                         │
│           ▼                         │
│      PruneGraph                     │
│           │                         │
│           ▼                         │
│  ┌─────────────────────────────┐    │
│  │ Disable Unnecessary Quant   │    │
│  └─────────────────────────────┘    │
│           │                         │
│           ▼                         │
│      PruneGraph                     │
└─────────────────────────────────────┘
         │
         ▼
    融合后的模型
```

---

## BN Tuning 流程

`QuantModel` 支持 **BatchNorm Tuning**，这是一种调整 BN 统计量的技术，用于补偿量化引入的噪声。

**原理**: 在量化模型上运行少量训练数据，更新 BN 的 running_mean 和 running_var。

```python
@contextmanager
def batchnorm_tuning(self):
    # 准备阶段
    self.model.train()
    self.set_quant(w_quant=True, a_quant=True)
    
    for n, m in self.model.named_modules():
        if isinstance(m, QBatchNorm2d):
            # 重置统计计数
            m.module.num_batches_tracked = m.module.num_batches_tracked.zero_()
    
    yield  # 用户在这里运行训练数据
    
    # 清理阶段
    self.model.eval()
    update_config(self.cfg.SCHEDULE, "FUSE_BN", True)
    self.model = fuse_operations(self.model, self.cfg.SCHEDULE, custom_fuse_list=["fuse_bn"])
    self.set_quant(w_quant=False, a_quant=False)
```

**使用**:
```python
with qmodel.batchnorm_tuning():
    for data in dataloader:
        qmodel(data)
```

---

## 依赖关系

```
fuse_operations
    ├── fx_symbolic_trace (tools/graph_wrapper.py)
    ├── PruneGraph (converters/prune.py)
    └── 各融合模块 (fuse_bn, disable_unnecessary_quant)
```

---

## 配置控制

```yaml
SCHEDULE:
  FUSE_BN: true               # 启用 BN 融合
  BN_TUNING: false            # 禁用 BN 微调
  DISABLE_UNNECESSARY_QUANT: true  # 禁用不必要量化
```

可以通过配置启用/禁用特定的融合操作：

```python
# 只融合 BN，跳过其他
fuse_operations(model, config, custom_fuse_list=["fuse_bn"])
```
