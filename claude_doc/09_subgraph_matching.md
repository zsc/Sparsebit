# 09：子图匹配与图替换引擎（以 `subgraph_matching.py` 为核心）

> 核心文件：`sparsebit/quantization/converters/utils/subgraph_matching.py`  
> 配套：`subgraph_matching_node.py`、`subgraph_matching_replace_pattern.py`、`subgraph_matching_utils.py`、`dominator_tree.py`、`bitpartite_graph_matching.py`、`prune.py`

## 1. 角色定位

Sparsebit 的 simplify/fuse 等图 pass 都建立在同一套“子图匹配→替换”框架上：

- **描述模式**：用 `MatchingNode` 列表描述一个子图（带输入连接关系）
- **匹配子图**：`SubgraphMatcher.apply()` 在 FX 图里找一个匹配
- **执行替换**：`ReplacePatternBase.get_new_graph()` 构造新节点并替换旧节点
- **剪枝**：`PruneGraph.apply()` 删除与输出无关的节点

## 2. MatchingNode：如何描述一个子图

定义在 `subgraph_matching_node.py`：

- `name`：模式节点名称（唯一）
- `inputs`：输入节点名称列表（用名字表达连接关系）
  - `None` 表示通配输入（不纳入匹配图或不关心来源）
- `op_type`：允许的实际算子类型列表（class 或 function）
- `checker(node, module) -> bool`：单点额外过滤条件
- `input_match_type`：输入匹配策略
  - `ALL`：严格按顺序对齐输入
  - `SUBSET`：允许子集/乱序（代码里有 Hungary 最大匹配，但注释提示该能力并不成熟）

**隐含要求**：`MatchingNode` 列表必须是拓扑序（引用到的输入节点必须出现在前面）。

## 3. ReplacePatternBase：一个 pass 的抽象

定义在 `subgraph_matching_replace_pattern.py`：

- `make_nodes()`：返回 `MatchingNode list`
- `make_joint_checkers()`：返回联合 checker（可同时检查多个节点/模块）
- `make_matching_strategy()`：返回替换策略（apply once / repeat）
- `get_new_graph(nodes_dict, modules_dict, model, transform_idx)`：构造替换后的新子图

执行入口：

- `apply(m)`：循环调用 `apply_once` 直到不再匹配（或策略指定只跑一次）
- `apply_once(m)`：
  1) `matcher.apply(m)` 找一个匹配
  2) 调用 `get_new_graph` 得到 `{old_name -> new_node}` 的替换映射
  3) 对图中所有节点执行 `replace_input_with(old_node, new_node)`
  4) `PruneGraph.apply(m)` 清理无用节点

## 4. SubgraphMatcher：匹配算法概览

核心步骤（高层）：

1) **coarse filtering**
   - 先按 FX 图节点的“真实 op 类型”分桶（`get_operators_type`）
   - 再对每个 MatchingNode：
     - type 过滤
     - checker 过滤
     - 输入关系过滤（`ALL` 直接比对输入数与输入来源；`SUBSET` 用 Hungary 最大匹配）
2) **pad supported node/operator**
   - 给模式图加一个虚拟 `__root__`，把模式子图的所有“输出节点”连到 root
   - 给真实图也加一个虚拟 operator，方便统一处理“多输出锚点”的情况
3) **match（DFS + 连接关系校验）**
   - 模式图会构建“反向支配树”（`DominatorTree`），并生成一种遍历次序 `rnk`
   - DFS 逐层选择候选 op
   - 在选择时检查：
     - 与前驱节点的连接位置（避免一个输入槽被重复使用）
     - joint-checker（在支配树保证的时机执行）

匹配成功后返回：

- `ops_dict : {pattern_name -> fx.Node}`
- `modules_dict : {pattern_name -> nn.Module | callable | None}`

## 5. 使用示例：fuse_bn / remove_identity

pass 文件一般长这样：

- `make_nodes()` 描述结构（conv/linear → bn）
- `get_new_graph()`：
  - 创建新 module（可能使用 `torch.nn.utils.fusion.*`）
  - 用 `model.graph.inserting_after(...)` 插入新节点
  - 返回 `{ "bn": new_node }` 表示用新 node 替换旧 bn 的输出锚点

## 6. ASCII 图：从模式到替换

```text
模式（MatchingNode）:
  cnn_layer  ->  bn

真实 FX 图:
  ... -> qconv -> bn -> relu -> ...

匹配后:
  nodes_dict["cnn_layer"] = <fx node: qconv>
  nodes_dict["bn"]        = <fx node: bn>

get_new_graph 生成:
  qconv_bn = fuse(qconv, bn)
  new_node = call_module("qconv_bn", x)

替换:
  所有使用 bn 输出的地方，改用 new_node
  再 prune 掉旧 bn 节点
```

## 7. OCaml type signature（接口投影）

```ocaml
type fx_graph_module
type fx_node

type matching_node
type replace_strategy = Apply_repeat | Apply_once

type subgraph_matcher
val create_matcher :
  matching_nodes:matching_node list ->
  joint_checkers:((string list) * 'checker) list ->
  matching_strategy:replace_strategy ->
  subgraph_matcher

val apply_matcher :
  subgraph_matcher ->
  fx_graph_module ->
  ( (string, fx_node) Hashtbl.t * (string, 'module) Hashtbl.t ) option

class type replace_pattern = object
  method apply : fx_graph_module -> bool
end
```

## 8. 重写时的注意事项

- 这套 matcher 假设 FX 图节点是拓扑序；新实现要么保持假设，要么显式排序。
- `InputMatchingType.SUBSET` 在文档里提示“未实装”，但代码里已有 Hungary 最大匹配；重写时应明确是否支持，并补齐测试。
- 当前 matcher 每次只返回“一组匹配”（并按策略 repeat），不支持一次性返回所有匹配；如果需要更强的重写系统，可能要扩展为多匹配/冲突消解。

