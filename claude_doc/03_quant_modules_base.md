# 03：`sparsebit/quantization/modules/base.py` —— QuantOpr 与多输入量化策略

## 1. 角色定位

该文件定义了量化算子的两种“基类形态”：

- `QuantOpr`：单输入（或带权重）算子量化基类，提供：
  - `input_quantizer`（激活量化）
  - `weight_quantizer`（权重量化，若算子有 `self.weight`）
- `MultipleInputsQuantOpr`：多输入算子量化基类（如 add），自身通常不做量化，
  而是在每个输入前插入 `QIdentity` 来量化输入。

它是 Sparsebit “算子级别量化” 的核心抽象边界。

## 2. QuantOpr：关键行为

### 2.1 `build_quantizer(config)`

- 根据 `config.BACKEND` 设置 backend（virtual/ORT/TRT）。
- 如果 `self.weight is not None`：
  - `config.W.TARGET = (QuantTarget.WEIGHT,)`
  - `self.weight_quantizer = build_quantizer(config.W)`
- 无论如何都会构建 `input_quantizer`：
  - `config.A.TARGET = (QuantTarget.FEATURE,)`
  - `self.input_quantizer = build_quantizer(config.A)`

这里的 `TARGET` 是 `QuantDescriptor` 构造时的关键输入（影响 `DataCache.get_batch_size` 等语义）。

### 2.2 `set_quant(w_quant, a_quant)`

- 同时控制 input/weight quantizer 的开关
- 如果 quantizer 处于 `fake_fused=True`（见 `Quantizer.set_fake_fused`），即使开关为 True 也不会启用

> 这使得图 pass 可以通过 `set_fake_fused()` 来“逻辑上关闭量化”，而不改图结构。

### 2.3 `__repr__`

如果 quantizer 已启用，会在 repr 中打印 quantizer 的 scale/zp 信息（便于调试）。

## 3. MultipleInputsQuantOpr：插 `QIdentity` 的策略

多输入算子（典型：`operator.add` / `torch.add` → `QAdd`）的核心问题是：

> 每个输入可能来自不同路径，不一定共享同一个输入量化器。

Sparsebit 的处理方式：

1) 多输入算子自身通常不继承 `QuantOpr`，而继承 `MultipleInputsQuantOpr`  
2) 在 `QuantModel._build_quantizer()` 中：
   - 如果 node 的 `all_input_nodes` 数量 > 1：调用 `prepare_input_quantizer(node, model)`
3) `prepare_input_quantizer` 会对每个输入插入一个 `QIdentity`：

```text
x1 ----\
        +--> QAdd ---> y
x2 ----/

变为：

x1 -> QIdentity0 --\
                   +--> QAdd ---> y
x2 -> QIdentity1 --/
```

每个 `QIdentity` 都是 `QuantOpr`，因此会被构建 `input_quantizer`，实现“逐输入量化”。

> 是否启用这件事通常由算子实现决定，例如 `QAdd` 会读取 `config.A.QADD.ENABLE_QUANT`。

## 4. OCaml type signature（接口投影）

```ocaml
type cfg
type tensor
type quantizer

class type quant_opr = object
  method build_quantizer : cfg -> unit
  method set_quant : w_quant:bool -> a_quant:bool -> unit
  method forward : tensor -> tensor
end

class type multi_inputs_quant_opr = object
  method prepare_input_quantizer : node:'node -> model:'model -> unit
  method forward : tensor -> tensor -> tensor
end
```

## 5. 重写时的注意事项

- “多输入算子量化”这一点是框架层面而不是单个算子层面的问题：新实现应尽早把策略固定下来（插 identity、共享 quantizer、或对齐到某个输入）。
- `QuantOpr.build_quantizer` 通过修改 config（写 `TARGET`）来驱动下游 `QuantDescriptor`，这种隐式写入在重写时容易出错，建议改为显式参数。

