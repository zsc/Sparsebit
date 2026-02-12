# 05：`sparsebit/quantization/quantizers/quant_tensor.py` —— FakeQuant、STE 与 CUDA 扩展

## 1. 角色定位

该文件实现了 Sparsebit 的核心“数值引擎”之一：

- fake-quant 的 forward（量化→反量化）
- STE（Straight-Through Estimator）形式的 backward
- 根据 backend（virtual/ORT/TRT）选择不同的 fake-quant 规则
- ONNX 导出时的 `torch_fake_quant`（用于生成 QDQ 节点）

## 2. CUDA 扩展：import 时编译（重要实现细节）

当 `torch.cuda.is_available()` 时，会在 import 阶段执行：

- `torch.utils.cpp_extension.load(...)`
  - sources：`torch_extensions/export.cc`、`torch_extensions/fake_quant_tensor.cu`
  - build 目录：`sparsebit/quantization/torch_extensions/build`

这意味着：

- 第一次 import 可能很慢（编译）
- 运行环境需要完整的 CUDA 编译链

## 3. STE：`torch.autograd.Function`

`STE` 的语义：

- forward：调用 `fake_quant_factory[backend]` 进行 fake-quant
- backward：在 CUDA 上调用自定义 kernel 的 backward（per-channel/per-tensor 两套）

当无 CUDA 时，backward 分支直接 `NotImplementedError`（注释里提到推荐用 CUDA 加速训练）。

## 4. backend 差异：ORT vs TRT

### 4.1 `ort_fake_quant`

- 支持 affine（可非零 zp）
- CPU fallback 时：
  - `zp = zero_point.round()`
  - `x_q = clamp(round(x/scale) + zp, qmin, qmax)`
  - `x_dq = (x_q - zp) * scale`

### 4.2 `trt_fake_quant`

关键约束：**TRT 只支持对称量化**，因此 assert：

- `abs(zero_point).sum() == 0`

CPU fallback 时：

- `x_q = clamp(round(x/scale), qmin, qmax)`
- `x_dq = x_q * scale`

此外还提供 `*_dqrange` 来计算反量化后的浮点范围（用于后端推理的边界推导）。

## 5. ONNX 导出：`torch_fake_quant`

导出 ONNX 时 quantizer 会走 `torch_fake_quant`（而不是自定义 STE），该函数使用：

- `torch.fake_quantize_per_channel_affine`
- `torch.fake_quantize_per_tensor_affine`

导出时，为了兼容 ONNX 原生 QDQ 的 dtype，内部用了固定量化范围：

- 如果是 `uint*`：固定 `[0, 255]`
- 否则固定 `[-128, 127]`

原因：ONNX 对 QDQ 的原生支持主要围绕 8-bit；因此当模型内部使用 <8bit 时，需要额外信息辅助或外部处理（`QuantModel.export_onnx` 也要求 `extra_info=True`）。

## 6. OCaml type signature（接口投影）

```ocaml
type tensor
type backend = Virtual | OnnxRuntime | TensorRT
type qdesc

val ort_fake_quant : tensor -> scale:tensor -> zero_point:tensor -> qdesc -> tensor
val trt_fake_quant : tensor -> scale:tensor -> zero_point:tensor -> qdesc -> tensor

val torch_fake_quant : tensor -> scale:tensor -> zero_point:tensor -> qdesc -> tensor

module STE : sig
  val apply : tensor -> tensor -> tensor -> qdesc -> backend -> tensor
end
```

## 7. 重写时的注意事项

- “import 即编译 CUDA 扩展”会显著影响可用性；重写时建议把 kernel 编译/加载延迟到首次使用或显式安装步骤。
- backend 的约束（尤其 TRT 的对称量化）要在配置层和数值实现层双重校验，否则很难定位错误来源。
