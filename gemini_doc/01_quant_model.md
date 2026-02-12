# QuantModel - 量化模型主类

## 文件信息

- **路径**: `sparsebit/quantization/quant_model.py`
- **核心类**: `QuantModel`
- **继承**: `torch.nn.Module`

---

## OCaml Type Signature

```ocaml
type quant_model = {
  cfg : config;
  device : torch.device;
  model : fx.GraphModule;
  enable_qat : bool;
  calibration_runner : calibration_runner option;
}

val create : torch.nn.Module -> config -> quant_model
val trace : torch.nn.Module -> config -> fx.GraphModule
val convert : fx.GraphModule -> fx.GraphModule
val build_quantizers : quant_model -> unit
val prepare_calibration : quant_model -> unit
val calc_qparams : quant_model -> bool -> bool -> bool -> unit
val set_quant : quant_model -> bool -> bool -> unit
val export_onnx : quant_model -> tensor -> string -> ?opset_version:int -> unit
```

---

## 详细功能说明

### 类职责

`QuantModel` 是整个量化系统的**核心入口**。它包装一个普通的 PyTorch 模型，通过 FX 追踪和转换，将模型中的普通算子替换为对应的量化算子 (`QuantOpr`)，并附加量化器 (`Quantizer`) 和观察器 (`Observer`)。

### 初始化流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      QuantModel.__init__                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. _trace(model)                                               │
│     └── 使用 QTracer 进行 FX 追踪，得到 GraphModule             │
│                                                                 │
│  2. _run_simplifiers()                                          │
│     └── 简化计算图（移除冗余节点、优化结构）                    │
│                                                                 │
│  3. _convert2quantmodule()                                      │
│     └── 将普通算子替换为对应的 QuantOpr                         │
│                                                                 │
│  4. _build_quantizer()                                          │
│     └── 为每个 QuantOpr 构建 input_quantizer 和 weight_quantizer│
│                                                                 │
│  5. _run_fuse_operations()                                      │
│     └── 融合算子（如 conv+bn，根据配置）                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 核心方法

#### `__init__(self, model: nn.Module, config)`

构造函数，执行完整的量化模型初始化。

**参数**:
- `model`: 原始 PyTorch 模型
- `config`: 量化配置 (CfgNode)

---

#### `_trace(self, model) -> fx.GraphModule`

使用自定义的 `QTracer` 对模型进行符号化追踪。

**关键代码**:
```python
def _trace(self, model):
    skipped_modules = self.cfg.SKIP_TRACE_MODULES
    tracer = QTracer(skipped_modules)
    graph = tracer.trace(model)
    traced = fx.GraphModule(tracer.root, graph, name)
    return traced
```

---

#### `_convert2quantmodule(self)`

将 GraphModule 中的普通算子替换为对应的 `QuantOpr`。

**处理逻辑**:
```
遍历 graph.nodes:
    if node.op == "call_module":
        # 查找 QMODULE_MAP，替换为对应的 QuantOpr
        new_module = QMODULE_MAP[type(org_module)](org_module)
    elif node.op == "call_function":
        # 处理函数调用（如 F.relu, torch.add）
        new_module = QMODULE_MAP[n.target](n, self.cfg)
    elif node.op == "call_method":
        # 处理方法调用（如 tensor.view）
        new_module = QMODULE_MAP[target_op](n, self.cfg)
```

**关键操作**:
- 使用 `graph.inserting_after(n)` 插入新节点
- 使用 `n.replace_all_uses_with(new_node)` 重定向依赖
- 使用 `graph.erase_node(n)` 删除旧节点
- 调用 `traced.recompile()` 重新编译图

---

#### `_build_quantizer(self)`

为每个 `QuantOpr` 构建量化器。

**处理逻辑**:
```python
for node in self.model.graph.nodes:
    if node.op == "call_module":
        module = getattr(self.model, node.target)
        if isinstance(module, QuantOpr):
            # 合并全局配置和特定模块配置
            _config = self.cfg.clone()
            _config.W = _sub_build(self.cfg.W, node.target)
            _config.A = _sub_build(self.cfg.A, node.target)
            module.build_quantizer(_config)
```

---

#### `prepare_calibration(self)`

准备校准流程。注册 forward hook 捕获输入/输出。

**实现**:
```python
def prepare_calibration(self):
    from sparsebit.quantization.tools.calibration import CalibrationRunner
    self.eval()
    self.calibration_runner = CalibrationRunner(self.model)
    self.calibration_runner.prepare_calibration()
```

---

#### `calc_qparams(self, asym=False, w_quant=False, a_quant=False)`

计算量化参数（scale 和 zero_point）。

**参数**:
- `asym`: 是否使用非对称校准（考虑前面层的量化误差）
- `w_quant`: 在校准时是否启用权重量化
- `a_quant`: 在校准时是否启用输入量化

**流程**:
```
1. 移除 forward hook
2. 逐层遍历节点
3. 对每个 QuantOpr:
   - run_feature_calibration: 收集输入统计，计算输入量化参数
   - 前向传播获取浮点输出
   - run_weight_calibration: 收集权重统计，计算权重量化参数
   - 如果使用 AdaRound，执行重建优化
4. 更新后续层的输入数据
```

---

#### `set_quant(self, w_quant=False, a_quant=False)`

全局开关所有量化器的量化状态。

---

#### `export_onnx(self, dummy_data, name, ...)`

导出 QDQ-ONNX 模型。

**关键步骤**:
1. 设置 `export_onnx=True`，使量化器使用 `torch_fake_quant`
2. 调用 `torch.onnx.export`
3. 恢复 `export_onnx=False`
4. 可选：添加额外信息（比特数等）到 ONNX 节点属性

---

## 依赖关系

```
QuantModel
    ├── QTracer (quant_tracer.py)
    ├── simplify (converters/simplifiers)
    ├── fuse_operations (converters/fuse_operations)
    ├── QuantOpr (modules/base.py)
    ├── QMODULE_MAP (modules/__init__.py)
    ├── Quantizer (quantizers/base.py)
    └── CalibrationRunner (tools/calibration.py)
```

---

## 使用示例

```python
from sparsebit.quantization import QuantModel, parse_qconfig

# 1. 解析配置
config = parse_qconfig("config.yaml")

# 2. 创建量化模型
model = ...  # 原始 PyTorch 模型
qmodel = QuantModel(model, config)

# 3. 准备校准
qmodel.prepare_calibration()

# 4. 运行校准数据
for data in calib_dataloader:
    qmodel(data)

# 5. 计算量化参数
qmodel.calc_qparams()

# 6. 启用量化
qmodel.set_quant(w_quant=True, a_quant=True)

# 7. 导出 ONNX
qmodel.export_onnx(dummy_data, "model.onnx")
```
