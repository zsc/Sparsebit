# Sparsebit 现状理解笔记（用于重写）

> 本目录文档全部为中文，面向“将 Sparsebit 重写”前的功能梳理与源码阅读笔记。

## 1. 这个 repo 现在提供什么功能

- 核心库：`sparsebit/`
  - `sparsebit/quantization/`：基于 `torch.fx` 的量化框架（PTQ / QAT），支持：
    - 把原模型 trace 成 FX Graph，再把图里每个算子替换为可量化的 `QuantOpr`（QModule）
    - 针对每个 `QuantOpr` 构建 `Quantizer`/`Observer`，做校准（calibration）得到 `scale/zero_point`
    - 通过 graph pass 做简化（simplifiers）与融合/关量化（fuse_operations）
    - 导出 QDQ-ONNX（可选写入额外 bit 信息）
  - `sparsebit/sparse/`：剪枝/稀疏化原型（structured / unstructured），整体更“demo/实验性”，文档基本缺失。
- 配套与扩展：
  - `examples/`：量化/剪枝的训练与推理示例（CIFAR、ImageNet、YOLO、BERT、wikitext…）
  - `ci/`：回归测试（用于验证 quantization 关键流程）
  - `large_language_models/`：LLM 相关脚本与 CUDA kernel（GPTQ / QLoRA 等），相对独立于 `sparsebit/` 主库

## 2. 核心名词（Nouns）

| 名词 | 主要位置 | 一句话解释 |
|---|---|---|
| `QuantModel` | `sparsebit/quantization/quant_model.py` | 量化生命周期的“总控”，组织 trace→pass→替换→建量化器→融合/导出 |
| `torch.fx.GraphModule` | PyTorch | 中间表示（IR）：以图的形式表达模型计算 |
| `QMODULE_MAP` / 注册器 | `sparsebit/quantization/modules/__init__.py` | “原算子/模块 → QModule”的映射表（由装饰器注册） |
| `QuantOpr` | `sparsebit/quantization/modules/base.py` | 可量化算子基类：持有 `input_quantizer`/`weight_quantizer` |
| `MultipleInputsQuantOpr` | 同上 | 多输入算子基类：通常自己不量化，而是给各输入插 `QIdentity` |
| `Quantizer` | `sparsebit/quantization/quantizers/base.py` | 负责 fake-quant（训练/校准）与导出（ONNX）行为，内部持 `Observer` |
| `Observer` / `DataCache` | `sparsebit/quantization/observers/base.py` | 负责收集统计量并计算量化参数（min/max→scale/zp） |
| `CalibrationRunner` | `sparsebit/quantization/tools/calibration.py` | 负责按层校准：给每层 quantizer 填好 qparams |
| `GraphVisitor` / `SharedData` | `sparsebit/quantization/tools/graph_wrapper.py` | FX 图遍历+hook 引擎：缓存/释放中间结果 |
| `ReplacePatternBase` / `SubgraphMatcher` | `sparsebit/quantization/converters/utils/*` | 子图匹配与替换引擎：实现 simplify/fuse 等 pass |
| `SparseModel` | `sparsebit/sparse/sparse_model.py` | 剪枝生命周期总控：trace→替换为 SparseOpr→构建 Sparser→算 mask |
| `Sparser` | `sparsebit/sparse/sparsers/*` | 计算稀疏 mask 的策略（目前主要是 L1-norm） |

## 3. 核心动词（Verbs）

| 动词 | 入口 | 主要产物 |
|---|---|---|
| trace | `QuantModel._trace()` / `fx.symbolic_trace()` | `torch.fx.GraphModule` |
| simplify（简化） | `converters.simplify()` | 更“干净”的 FX 图（去 Identity 等） |
| replace（替换为 QModule） | `QuantModel._convert2quantmodule()` | 图中 op 变为 `QuantOpr`/QModule |
| build quantizer | `QuantModel._build_quantizer()` / `QuantOpr.build_quantizer()` | 每层拥有 `Quantizer`+`Observer` |
| fuse/disable（后处理） | `converters.fuse_operations()` | fuse BN / 关闭不必要量化等图变换 |
| calibrate（校准） | `prepare_calibration()` + `calc_qparams()` | `scale/zero_point` 落到 quantizer 上 |
| set_quant（开关量化） | `QuantModel.set_quant()` | 控制 forward 走量化/非量化路径 |
| export onnx | `QuantModel.export_onnx()` | QDQ-ONNX（可选额外 bit 标注） |
| subgraph match/replace | `ReplacePatternBase.apply()` | 以模式为单位的图替换 |

## 4. 引擎（Engines）

**(1) FX IR 引擎**：`torch.fx` 提供 trace 和 GraphModule；Sparsebit 的大多数能力都建立在“能把模型变成可编辑的图”之上。

**(2) 图 pass 引擎**：`ReplacePatternBase` + `SubgraphMatcher` + `PruneGraph` 负责：
1) 在 FX 图里匹配子图（按 op 类型、输入关系、checker、joint-checker）  
2) 构造新节点/新 module 替换旧节点  
3) 通过 `PruneGraph` 删除与输出无关的节点

**(3) 校准引擎**：`GraphVisitor` + `SharedData` 提供一种“按 FX 节点注册 hook 并缓存中间结果”的机制，`CalibrationRunner` 在其上实现 layerwise 校准。

**(4) FakeQuant/后端引擎**：`Quantizer` 通过 `quant_tensor.STE` 进行 fake-quant；并根据 `BACKEND` 区分 ORT/TRT 约束（例如 TRT 要求对称量化、zp=0）。

## 5. 点火钥匙（Ignition Key）：最小可跑流程

### 5.1 PTQ（后训练量化）

```python
from sparsebit.quantization import QuantModel
from sparsebit.quantization.quant_config import parse_qconfig

qcfg = parse_qconfig("qconfig.yaml")
qmodel = QuantModel(fp32_model, qcfg).to(qcfg.DEVICE)

qmodel.prepare_calibration()
with torch.no_grad():
    for x, _ in calib_loader:
        qmodel(x.to(qcfg.DEVICE))          # 仅用于收集统计数据
qmodel.calc_qparams()                      # 计算每层 scale/zp
qmodel.set_quant(w_quant=True, a_quant=True)
```

### 5.2 QAT（量化感知训练）

```python
qmodel.prepare_calibration()
...  # 用少量数据跑一遍以初始化 qparams
qmodel.init_QAT()                          # 内部会 calc_qparams + set_quant
train(qmodel)
```

### 5.3 Pruning（稀疏/剪枝）

```python
from sparsebit.sparse import SparseModel
from sparsebit.sparse.sparse_config import parse_sconfig

scfg = parse_sconfig("sconfig.yaml")
smodel = SparseModel(fp32_model, scfg).to(scfg.DEVICE)
smodel.calc_params()                       # 计算并缓存 mask（实现较原型）
```

## 6. 主要模块与调用关系（Call Graph）

### 6.1 QuantModel 初始化阶段

```text
用户 nn.Module
   |
   |  (1) _trace: QTracer.trace / fx.GraphModule
   v
FX GraphModule
   |
   |  (2) simplify: converters.simplify (pre-pass)
   v
FX GraphModule (simplified)
   |
   |  (3) _convert2quantmodule: 用 QMODULE_MAP 替换每个 node 为 QModule
   v
FX GraphModule (QuantOpr/QModule)
   |
   |  (4) _build_quantizer: 给每个 QuantOpr 挂 input/weight quantizer
   v
量化可用模型
   |
   |  (5) _run_fuse_operations: fuse_bn / disable_unnecessary_quant ...
   v
最终 QuantModel.model
```

### 6.2 校准阶段（calc_qparams）

```text
QuantModel.prepare_calibration()
  -> CalibrationRunner.prepare_calibration()
     -> GraphVisitor 注册 hooks (SharedData 缓存输入 placeholder 的 tensor)

用户跑若干 batch forward (仅收集数据)

QuantModel.calc_qparams()
  -> CalibrationRunner.layerwise_calibration()
     for node in fx.graph.nodes:
        - run_feature_calibration()  (输入量化器 observer)
        - module_forward(float)      (得到 float 输出并缓存)
        - run_weight_calibration()   (权重量化器 observer / AdaRound 重建)
        - module_forward(quant, 可选 asym)
```

### 6.3 图 pass（simplify / fuse_operations）

```text
ReplacePatternBase.apply()
  -> SubgraphMatcher.apply()   # 找到一个匹配的子图
  -> get_new_graph()           # 构造新 node / 新 module
  -> replace_input_with()      # 重连输入
  -> PruneGraph.apply()        # 剪掉无用节点
  -> (repeat until no match)   # 可选循环
```

## 7. “Top10 关键文件”索引（每个文件一篇）

1. `claude_doc/01_quant_model.md`（`sparsebit/quantization/quant_model.py`）
2. `claude_doc/02_quant_config.md`（`sparsebit/quantization/quant_config.py`）
3. `claude_doc/03_quant_modules_base.md`（`sparsebit/quantization/modules/base.py`）
4. `claude_doc/04_quantizer_base.md`（`sparsebit/quantization/quantizers/base.py`）
5. `claude_doc/05_quant_tensor.md`（`sparsebit/quantization/quantizers/quant_tensor.py`）
6. `claude_doc/06_observer_base.md`（`sparsebit/quantization/observers/base.py`）
7. `claude_doc/07_calibration_runner.md`（`sparsebit/quantization/tools/calibration.py`）
8. `claude_doc/08_graph_wrapper.md`（`sparsebit/quantization/tools/graph_wrapper.py`）
9. `claude_doc/09_subgraph_matching.md`（`sparsebit/quantization/converters/utils/subgraph_matching.py` 为核心）
10. `claude_doc/10_sparse_model.md`（`sparsebit/sparse/sparse_model.py`）

