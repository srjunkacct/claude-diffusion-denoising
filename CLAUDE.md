# Claude Code Instructions

## Project Overview
Java port of DDPM (Denoising Diffusion Probabilistic Models) using DJL 0.31.0 + PyTorch.
Build: Gradle 8.14, Java 17 source compat, running on JDK 24.

## Critical: DJL GPU Memory Leak Patterns

### Never extract floats directly from GPU tensors on long-lived managers
Both `getFloat()` and `toFloatArray()` on GPU tensors leak a `PtNDManager` sub-manager per call.
The leak chain: `toFloatArray()` → `toByteBuffer()` → `JniUtils.getByteBuffer()` → `toDevice(cpu)` → `JniUtils.to()` → `newSubManager(cpu)` on the tensor's manager, never closed.

**Fix:** Attach the tensor to a scoped sub-manager before extracting:
```java
try (NDManager scopedMgr = parentManager.newSubManager()) {
    gpuTensor.attach(scopedMgr);
    float val = gpuTensor.toFloatArray()[0];
} // scopedMgr.close() cleans up leaked sub-managers
```
Tensors on a stepManager (closed each iteration) are fine — the leak is cleaned up automatically.

### Always close getGradient() wrappers
`param.getArray().getGradient()` creates a NEW `PtNDArray` wrapper on the parent manager each call. Always use `try (NDArray grad = param.getArray().getGradient()) { ... }`.

### Always close reshape() views on parameters
`param.reshape(...)` creates a view NDArray on the parent manager. Close via try-with-resources.

### Operation result manager rule
`a.mul(b)` result goes to `a.getManager()`. So `paramArray.mul(x)` → parent manager (leak risk), but `stepInput.mul(paramArray)` → stepManager (safe).

### Always zero gradients before backward()
DJL's `PtGradientCollector` does NOT zero gradients. PyTorch's `backward()` accumulates into `.grad` tensors. Without zeroing, gradients from all prior steps accumulate, causing the loss to plateau. DJL's built-in `zeroGradients()` leaks 2 `getGradient()` wrappers per parameter per call. Use a custom implementation with try-with-resources.

### Use ManualAdam, not DJL's built-in Adam
DJL's `optimizer.update()` creates temp NDArrays on the parent manager that are never closed. `ManualAdam` in this project closes all temps explicitly.

## DJL 0.31.0 API Gotchas
- `NDArray.sigmoid()` does NOT exist → use `ndArray.getNDArrayInternal().sigmoid()`
- `NDArray.where()` does NOT exist → use `NDArrays.where(condition, x, y)`
- `NDArray.attachGradient()` does NOT exist → use `NDArray.setRequiresGradient(boolean)`
- After `model.loadParameters()`, re-enable gradients: `param.getArray().setRequiresGradient(true)`
- `NDArray.mean(int[])` only supports single-axis reduction
- `NDArray.toLongArray()` requires INT64 dtype — cast first with `.toType(DataType.INT64, false)`
- `torch.cuda.empty_cache()` not in public API — use `GpuMemoryUtils.cudaEmptyCache()` (reflection)
