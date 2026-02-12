# CalibrationRunner - 校准流程控制器

## 文件信息

- **路径**: `sparsebit/quantization/tools/calibration.py`
- **核心类**: `CalibrationRunner`

---

## OCaml Type Signature

```ocaml
type tensor_storage = {
  outputs : (string, torch.tensor list) hashtbl;
}

type calibration_runner = {
  model : fx.GraphModule;
  builder : graph_visitor;
}

val create : fx.GraphModule -> calibration_runner
val prepare_calibration : calibration_runner -> unit
val layerwise_calibration : calibration_runner -> torch.device -> bool -> bool -> bool -> unit
val run_feature_calibration : calibration_runner -> fx.node -> bool -> unit
val run_weight_calibration : calibration_runner -> fx.node -> bool -> bool -> unit
val module_forward : calibration_runner -> int -> fx.node -> torch.device -> bool -> bool -> bool -> torch.tensor list
```

---

## 详细功能说明

### CalibrationRunner 类

`CalibrationRunner` 负责**逐层校准**量化参数。它的核心思想是：

1. 在模型输入处注册 hook，捕获每层输入
2. 逐层遍历计算图节点
3. 对每个 `QuantOpr` 节点：
   - 收集输入统计 → 计算输入量化参数
   - 前向传播获取输出
   - 收集权重统计 → 计算权重量化参数
   - 如果是 AdaRound，执行重建优化

### 为什么需要逐层校准？

```
问题：如果一次性收集所有层的统计信息：
- 第 N 层的输入依赖于第 N-1 层的输出
- 第 N-1 层量化后，其输出分布会改变
- 导致第 N 层观察到的输入分布不准确

解决方案（逐层校准）：
1. 第 1 层：用浮点输入 → 计算量化参数 → 前向得到输出
2. 第 2 层：用第 1 层的浮点输出作为输入 → 计算参数 → 前向...
3. 以此类推

这样每层都基于正确的输入分布计算量化参数。
```

---

## 核心方法

### `__init__(self, model)`

初始化，对模型进行符号化追踪。

```python
def __init__(self, model):
    self.model = fx_symbolic_trace(model)
```

---

### `prepare_calibration(self)`

准备校准，注册 forward hook。

**关键逻辑**:
```python
def prepare_calibration(self):
    input_names_cache = set(
        i.target for i in self.model.graph.nodes if i.op == "placeholder"
    )
    
    def _forward_hook(module, x_in, x_out, node, storage, record_names):
        # 捕获输入张量并存储
        flatten_x_in = flatten(x_in)
        flatten_args = flatten(node.args)
        for pos, (_x_in, _args) in enumerate(zip(flatten_x_in, flatten_args)):
            if isinstance(_args, torch.fx.Node) and _args.target in record_names:
                input_name = _args.target
                datas = storage.get_output(input_name)
                if datas is None:
                    datas = []
                datas.append(to_cpu(to_detach(x_in[pos])))
                storage.set_output(input_name, datas)
    
    def hook_wrapper(node, module, storage):
        # 为需要捕获输入的模块注册 hook
        ...
    
    self.builder = GraphVisitor(self.model, hook_wrapper)
```

---

### `layerwise_calibration(self, device, asym, w_quant, a_quant)`

**主校准流程**。

**参数**:
- `device`: 运行设备 (cuda/cpu)
- `asym`: 是否使用非对称校准（考虑前面层的量化误差）
- `w_quant`: 校准时是否使用权重量化
- `a_quant`: 校准时是否使用输入量化

**实现逻辑**:
```python
def layerwise_calibration(self, device, asym=False, w_quant=False, a_quant=False):
    # 移除 hook
    for handle in self.builder.handles:
        handle.remove()
    
    if asym:
        # 创建量化版本的存储（用于 asym 模式）
        self.builder.qstorage = copy.deepcopy(self.builder.storage)
    
    batch_num = None
    
    for node in self.model.graph.nodes:
        if node.op in ["placeholder", "output"]:
            if batch_num is None:
                batch_num = len(self.builder.storage.get_output(node.target))
            continue
        
        # 1. 特征校准（输入量化）
        self.run_feature_calibration(node, asym)
        
        # 2. 前向传播，获取浮点输出
        float_outputs = self.module_forward(batch_num, node, device)
        self.builder.storage.set_output(node.target, float_outputs)
        
        # 3. 权重校准
        self.run_weight_calibration(node, asym, a_quant=a_quant)
        
        # 4. 如果使用 asym 模式，进行量化前向
        if asym:
            quant_outputs = self.module_forward(
                batch_num, node, device, asym, w_quant, a_quant
            )
            self.builder.qstorage.set_output(node.target, quant_outputs)
            self.builder.qstorage.finish_node(node.target)
        
        # 5. 释放已完成节点的输出（节省内存）
        self.builder.storage.finish_node(node.target)
```

---

### `run_feature_calibration(self, node, asym)`

运行特征（输入）校准。

```python
def run_feature_calibration(self, node, asym):
    module = getattr(self.model, node.target)
    if (isinstance(module, QuantOpr) and 
        getattr(module, "input_quantizer", None) and
        not module.input_quantizer.fake_fused):
        
        # 收集该节点所有输入的统计信息
        for inp_node in node.all_input_nodes:
            inp_tensors = self.builder.storage.get_output(inp_node.target)
            for inp_tensor in inp_tensors:
                if isinstance(inp_tensor, torch.Tensor):
                    module.input_quantizer.update_observer(inp_tensor)
        
        # 计算输入量化参数
        module.input_quantizer.calc_qparams()
        # 清空 observer 的数据缓存
        module.input_quantizer.observer.data_cache.reset()
```

---

### `run_weight_calibration(self, node, asym, a_quant)`

运行权重校准。

```python
def run_weight_calibration(self, node, asym, a_quant):
    module = getattr(self.model, node.target)
    if isinstance(module, QuantOpr) and getattr(module, "weight_quantizer", None):
        # 权重只需要观察一次（不依赖输入数据）
        module.weight_quantizer.update_observer(module.weight)
        module.weight_quantizer.calc_qparams()
        
        # 如果是 AdaRound，执行重建优化
        if module.weight_quantizer.TYPE.lower() == "adaround":
            assert len(node.all_input_nodes) == 1
            # 获取输入和输出用于重建
            _storage = self.builder.qstorage if asym else self.builder.storage
            inp_tensors = _storage.get_output(node.all_input_nodes[0].target)
            out_tensors = self.builder.storage.get_output(node.target)
            
            reconstruct_qlayer(
                module,
                torch.cat(inp_tensors, dim=0),
                torch.cat(out_tensors, dim=0),
                a_quant=a_quant,
            )
```

---

### `module_forward(self, batch_num, node, device, ...)`

执行模块前向传播。

```python
def module_forward(self, batch_num, node, device, asym=False, w_quant=False, a_quant=False):
    module = getattr(self.model, node.target)
    if node.op == "call_module":
        module.eval()
    
    if isinstance(module, QuantOpr) and asym:
        # 启用量化
        module.set_quant(w_quant, a_quant)
    
    with torch.no_grad():
        outputs = []
        for batch_idx in range(batch_num):
            if node.op == "get_attr":  # 常量
                outputs.append(to_cpu(module.data))
                continue
            
            # 从 storage 提取输入参数
            storage = self.builder.qstorage if asym else self.builder.storage
            args = storage.extract_node_args(node.args, batch=batch_idx)
            kwargs = storage.extract_node_kwargs(node.kwargs, batch=batch_idx)
            args = to_device(args, device)
            kwargs = to_device(kwargs, device)
            
            # 前向传播并立即移回 CPU（节省显存）
            outputs.append(to_cpu(module(*args, **kwargs)))
    
    if isinstance(module, QuantOpr):
        # 恢复为浮点模式
        module.set_quant(w_quant=False, a_quant=False)
    
    return outputs
```

---

## 依赖关系

```
CalibrationRunner
    ├── fx_symbolic_trace (tools/graph_wrapper.py)
    ├── GraphVisitor (tools/graph_wrapper.py)
    ├── to_cpu, to_device, to_detach (tools/tensor_wrapper.py)
    ├── QuantOpr (modules/base.py)
    └── reconstruct_qlayer (quantizers/adaround.py)
```

---

## 校准流程图

```
┌─────────────────────────────────────────────────────────────────┐
│                    Layerwise Calibration                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for each node in graph:                                        │
│                                                                 │
│    ┌─────────────────────────────────────────────────────┐     │
│    │  if node is input/output:                           │     │
│    │     skip                                            │     │
│    └─────────────────────────────────────────────────────┘     │
│                                                                 │
│    ┌─────────────────────────────────────────────────────┐     │
│    │  run_feature_calibration(node)                      │     │
│    │  ├── 获取所有输入张量 from storage                  │     │
│    │  ├── 对每个输入: update_observer(input)            │     │
│    │  └── calc_qparams() → 计算输入量化参数              │     │
│    └─────────────────────────────────────────────────────┘     │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────┐     │
│    │  module_forward(batch_num, node, device)            │     │
│    │  └── 执行浮点前向，保存输出到 storage               │     │
│    └─────────────────────────────────────────────────────┘     │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────┐     │
│    │  run_weight_calibration(node)                       │     │
│    │  ├── update_observer(module.weight)                 │     │
│    │  ├── calc_qparams() → 计算权重量化参数              │     │
│    │  └── (if AdaRound) reconstruct_qlayer()             │     │
│    └─────────────────────────────────────────────────────┘     │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────┐     │
│    │  (if asym mode)                                     │     │
│    │  module_forward(..., w_quant=True, a_quant=True)    │     │
│    │  └── 执行量化前向，保存到 qstorage                  │     │
│    └─────────────────────────────────────────────────────┘     │
│                          │                                      │
│                          ▼                                      │
│    ┌─────────────────────────────────────────────────────┐     │
│    │  storage.finish_node(node.target)                   │     │
│    │  └── 释放该节点的输出（如果后续不再使用）           │     │
│    └─────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
