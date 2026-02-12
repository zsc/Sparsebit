# QuantOpr - 量化算子基类

## 文件信息

- **路径**: `sparsebit/quantization/modules/base.py`
- **核心类**: `QuantOpr`, `MultipleInputsQuantOpr`
- **继承**: `torch.nn.Module`

---

## OCaml Type Signature

```ocaml
type quant_opr = {
  weight : torch.tensor option;
  input_quantizer : quantizer option;
  weight_quantizer : quantizer option;
}

type multiple_inputs_quant_opr = {
  input_quantizer_generated : bool;
  apply_input_quant : bool;
}

class quant_opr : torch.nn.Module ->
  method forward : torch.tensor -> torch.tensor
  method build_quantizer : config -> unit
  method set_quant : bool -> bool -> unit
end

class multiple_inputs_quant_opr : torch.nn.Module ->
  method prepare_input_quantizer : fx.node -> fx.GraphModule -> unit
end
```

---

## 详细功能说明

### QuantOpr 类

`QuantOpr` 是所有**单输入**量化算子的**抽象基类**。它提供了量化算子的基本框架，包括：
- 输入量化器 (`input_quantizer`)
- 权重量化器 (`weight_quantizer`，可选）
- 量化开关控制

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `weight` | `torch.Tensor` | 权重张量引用（如果有） |
| `input_quantizer` | `Quantizer` | 输入量化器 |
| `weight_quantizer` | `Quantizer` | 权重量化器 |

#### 方法

##### `forward(x_in: torch.Tensor) -> torch.Tensor`

**抽象方法**，子类必须实现。

标准实现模板：
```python
def forward(self, x_in: torch.Tensor):
    # 1. 量化输入
    x_in = self.input_quantizer(x_in)
    # 2. 量化权重（如果有）
    weight = self.weight_quantizer(self.weight)
    # 3. 执行原始运算
    out = F.some_operation(x_in, weight, ...)
    return out
```

##### `build_quantizer(self, config)`

根据配置构建量化器。

**实现逻辑**:
```python
def build_quantizer(self, config):
    _backend = get_backend(config.BACKEND)
    
    # 构建权重量化器（如果有权重）
    if self.weight is not None:
        update_config(config.W, "TARGET", (QuantTarget.WEIGHT,))
        self.weight_quantizer = build_quantizer(cfg=config.W)
        self.weight_quantizer.set_backend(_backend)
    
    # 构建输入量化器
    update_config(config.A, "TARGET", (QuantTarget.FEATURE,))
    self.input_quantizer = build_quantizer(cfg=config.A)
    self.input_quantizer.set_backend(_backend)
```

##### `set_quant(self, w_quant=False, a_quant=False)`

控制量化器的开关状态。

**参数**:
- `w_quant`: 是否启用权重量化
- `a_quant`: 是否启用输入量化

**实现逻辑**:
```python
def set_quant(self, w_quant=False, a_quant=False):
    if self.weight_quantizer:
        if w_quant and not self.weight_quantizer.fake_fused:
            self.weight_quantizer.enable_quant()
        else:
            self.weight_quantizer.disable_quant()
    
    if self.input_quantizer:
        if a_quant and not self.input_quantizer.fake_fused:
            self.input_quantizer.enable_quant()
        else:
            self.input_quantizer.disable_quant()
```

---

### MultipleInputsQuantOpr 类

`MultipleInputsQuantOpr` 是**多输入**量化算子（如 `Add`, `Concat`）的基类。

**设计原因**: 多输入算子需要对每个输入独立量化，但量化参数可能不同。因此采用**延迟插入 QIdentity** 的策略。

#### 属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `input_quantizer_generated` | `bool` | 是否已生成输入量化器 |
| `apply_input_quant` | `bool` | 是否应用输入量化 |

#### 方法

##### `prepare_input_quantizer(self, node, model)`

在图的输入路径上插入 `QIdentity` 节点。

```
原始图结构:          转换后:
    A                    A
    │                    │
    ▼                    ▼
  ┌───┐               ┌─────────┐
  │Add│               │QIdentity│
  └───┘               └────┬────┘
    ▲                      │
    │                      ▼
    B                   ┌───┐
                        │Add│
                        └───┘
                           ▲
                           │
                        ┌──┴────┐
                        │QIdentity│
                        └──┬────┘
                           │
                           B
```

**实现逻辑**:
```python
def prepare_input_quantizer(self, node, model):
    from .unary import QIdentity
    
    if self.input_quantizer_generated:
        return
    
    input_nodes_cache = list(node.all_input_nodes)
    for idx, input_node in enumerate(input_nodes_cache):
        # 创建 QIdentity 模块
        new_module_name = node.name + "_identity{}".format(idx)
        new_module = QIdentity()
        model.add_module(new_module_name, new_module)
        
        # 在图中插入新节点
        with model.graph.inserting_before(node):
            identity_node = model.graph.create_node(
                op="call_module",
                target=new_module_name,
                args=(input_node,),
                kwargs={},
                name=new_module_name,
            )
        # 替换输入连接
        node.replace_input_with(input_node, identity_node)
    
    self.input_quantizer_generated = True
```

---

## 继承关系

```
                    nn.Module
                       │
              ┌────────┴────────┐
              │                 │
         QuantOpr       MultipleInputsQuantOpr
              │                 │
    ┌─────────┼─────────┐       │
    │         │         │       │
 QConv2d   QLinear   QReLU    QAdd
 QConvTranspose2d   QSigmoid  QConcat
    ...         ...      ...    ...
```

---

## 典型子类实现示例

### QConv2d (单输入)

```python
@register_qmodule(sources=[nn.Conv2d])
class QConv2d(QuantOpr):
    def __init__(self, org_module, config=None):
        super().__init__()
        self.fwd_kwargs = dict(
            stride=org_module.stride,
            padding=org_module.padding,
            ...
        )
        self.weight = org_module.weight  # 引用原始权重
        self.bias = org_module.bias
    
    def forward(self, x_in: torch.Tensor):
        x_in = self.input_quantizer(x_in)        # 量化输入
        weight = self.weight_quantizer(self.weight)  # 量化权重
        out = F.conv2d(x_in, weight, self.bias, **self.fwd_kwargs)
        return out
```

### QAdd (多输入)

```python
@register_qmodule(sources=[torch.add, operator.add])
class QAdd(MultipleInputsQuantOpr):
    def __init__(self, node, config):
        super().__init__()
        # 注意：QAdd 本身不做量化，依赖 QIdentity
```

---

## 依赖关系

```
QuantOpr
    ├── build_quantizer (quantizers/__init__.py)
    ├── QuantTarget (common.py)
    ├── get_backend (common.py)
    └── update_config (utils/common.py)

MultipleInputsQuantOpr
    └── QIdentity (modules/unary.py)
```
