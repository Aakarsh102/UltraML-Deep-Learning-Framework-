# UltraML: Exhaustive Technical Document

## 1. High-Level System Overview

**Purpose of the Framework**
UltraML is a low-level, CUDA-backed deep learning building-block library implemented in C++. It is designed to provide highly optimized, GPU-accelerated primitives (layers, activations, norms, and losses) for deep learning. Its primary goal is not to be a high-level Python API (like PyTorch) initially, but to serve as a robust, native C++ backbone upon which a full dynamic computational graph (autograd) can be layered without rewriting existing kernel operations.

**Core Architecture**
The architecture is heavily componentized into logically distinct modules: `core`, `activations`, `layers`, `losses`, and `norms`. At the heart of the system is the `Tensor` abstraction, which manages GPU-allocated memory, shapes, and gradients. Crucially, the execution model separates the forward pass from the backward pass. The user (or a future autograd engine) explicitly manages the lifecycle and flow of tensors, passing saved tensors directly into backward functions.

**Design Philosophy**
*   **Separation of Concerns:** Kernels and math logic are decoupled from graph traversal logic. Operations are mathematically pure and stateless from the perspective of the graph; they take inputs and write to outputs without maintaining internal execution state.
*   **Zero-Overhead Forward Passes:** Forward functions do not allocate memory dynamically. They assume outputs are pre-allocated, giving maximum control to the caller (or memory planner) to optimize VRAM.
*   **Autograd-Ready ABI:** The `AutogradNode` is forward-declared. This allows the core `Tensor` to hold a pointer to a graph node, meaning when the autograd engine is built, it can seamlessly hook into `Tensor` without changing the core ABI or touching the kernels.

**Comparison to PyTorch**
Conceptually, UltraML is currently equivalent to PyTorch's `ATen` (A Tensor Library) C++ backend, mixed with `THC` (Torch CUDA operations). Whereas PyTorch dynamically dispatches ops via a very complex `Dispatcher` and automatically builds an autograd tape natively, UltraML currently exposes direct function calls requiring explicit topological execution and gradient tensor management. While PyTorch's `Function::apply` tracks saved tensors implicitly, UltraML forces explicit argument passing of saved tensors (e.g., `softmax_backward` demands the output tensor).

---

## 2. Directory-Level Breakdown

*   **`core/`**: The foundational abstractions of the framework. It houses the `Tensor` struct, the execution `NNContext` (wrapping cuBLAS and cuDNN), and CUDA error-checking macros. This isolates hardware-level boilerplate and fundamental data types from mathematical operations.
*   **`activations/`**: Contains element-wise non-linearities. It exists as a separate directory because activation functions generally follow a specific structural archetype: they are shape-preserving, heavily rely on simple one-dimensional CUDA threads mapping directly to flat arrays, and require completely custom handwritten kernels rather than cuDNN/cuBLAS calls.
*   *(Other directories will be covered in subsequent sections).*

---

## 3. File-Level Breakdown & 4. Full Code Walkthrough

### **File: `CMakeLists.txt`**

#### File Purpose
Manages the build system for the library and examples. It compiles the source files into a static library `libultraml.a` and optionally builds the MLP example. Without it, linking against CUDA, cuBLAS, and cuDNN securely across platforms would be deeply manual and error-prone.

#### Internal Logic & Design Considerations
*   `set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)`: Targets Volta, Turing, Ampere, Ada, and Hopper. By default, it covers the vast majority of modern datacenters and consumer GPUs.
*   `add_library(ultraml STATIC ${ULTRAML_SOURCES})`: Built statically. **Alternative:** Shared library (`.so`). Rejected because static linking guarantees all inline CUDA kernels and device symbols are resolved at link time without complex `__declspec(dllexport)` logic or missing RPATHs in C++.
*   `POSITION_INDEPENDENT_CODE ON`: Required if this static library is eventually linked into a Python extension (like `pybind11`), which requires `-fPIC`.

---

### **File: `ultraml.h`**

#### File Purpose
The umbrella header file. It acts as the single point of inclusion for end-users, providing a clean, centralized interface that completely obscures the internal directory hierarchy.
*   **Design Rationale:** Following the "facade" pattern (like `#include <torch/torch.h>`), it reduces friction. The user does not need to know where `ReLU` lives versus `BatchNorm2d`.

---

### **File: `core/macros.h`**

#### File Purpose
Defines error-checking macros wrapping all CUDA API calls.

#### Internal Logic & Macros
*   **`ULTRAML_CUDA_CHECK`**, **`ULTRAML_CUBLAS_CHECK`**, **`ULTRAML_CUDNN_CHECK`**:
    *   **What they do:** Execute the wrapped API call, examine the return status. If the status is not `SUCCESS`, print the exact file, line number, and CUDA string error to `stderr`, and immediately terminate (`exit(EXIT_FAILURE)`).
    *   **Why it is necessary:** CUDA operations are asynchronous and silent on failure. If a kernel launch or memory allocation fails, the error will silently propagate, usually causing an incomprehensible SegFault later. Pinpointing the error at the point of origin is mandatory.
    *   **Alternatives rejected:** C++ exceptions (`throw std::runtime_error`). While safer for recovery, deep learning frameworks in C++ frequently use hard exits for irrecoverable hardware errors because recovering a GPU from an invalid memory context mid-training is effectively impossible.
*   **`#define ULTRAML_CUDA_BLOCK 256`**:
    *   **Why 256?** 256 threads per block is a highly robust "goldilocks" size for 1D mapping operations. It is a multiple of the warp size (32), ensuring perfect warp occupancy without hitting hardware limits for max threads per block (1024) or heavily constraining registers per thread.

---

### **File: `core/context.h` & `core/context.cu`**

#### File Purpose
Provides `NNContext`, a localized struct managing heavy hardware library handles.

#### Struct: `NNContext`
*   Contains `cublasHandle_t` and `cudnnHandle_t`.
*   **Why this abstraction exists:** Creating cuBLAS and cuDNN handles involves massive initialization overhead (setting up context, allocating internal workspaces). They must be created *once* per GPU thread and reused.
*   **Alternatives rejected:** Global static singletons. Singletons completely break multi-threading and multi-GPU setups. By explicitly creating and passing an `NNContext*`, the framework guarantees thread-safety and allows users to run different models on different GPU streams.

#### Functions
*   **`create_context` & `destroy_context`:** Wrappers around `cublasCreate` and `cudnnCreate`. Manages explicit lifecycle.

---

### **File: `core/op.h`**

#### File Purpose
Forward-declares `AutogradNode`.
*   **Why this file is separated:** It breaks circular dependencies. `Tensor` needs to know that `AutogradNode` exists to hold a pointer to it, but `AutogradNode` relies on `Tensor`. Forward declaring it in a standalone header keeps `tensor.h` exceptionally clean.

---

### **File: `core/tensor.h` & `core/tensor.cu`**

#### File Purpose
The spine of the framework. Represents a multi-dimensional array residing on GPU memory.

#### Struct: `Tensor`
*   `float* data;`: Raw device pointer. Only `float` is supported currently (FP32).
*   `int* shape; int ndim; int size;`: Metadata mapping the 1D flat array in memory to an N-dimensional mathematical object. Host-side arrays.
*   `float* grad; bool requires_grad;`: Gradient accumulation buffer.
*   `AutogradNode* grad_fn;`: Link to the autograd graph.
*   **Memory/Layout Considerations:** The structural metadata (shape) lives on the CPU (Host), while `data` lives on the GPU (Device). This is crucial: fetching shape info for host-side dispatch logic (like calculating block sizes) must not invoke a slow PCIe device-to-host transfer.

#### Functions
*   **`create_tensor`**:
    *   **Logic:** Allocates the host-side struct, copies the shape array to host memory, and allocates the contiguous `float* data` array on the device using `cudaMalloc`. Lazily calls `alloc_grad` if `requires_grad` is true.
    *   **Why not C++ constructors (`new Tensor`)?** The codebase relies on C-style opaque pointers (`malloc`/`free`) combined with functions, avoiding implicit memory allocation hidden behind C++ RAII bounds. This mimics C-APIs (like Python C extensions).
*   **`free_tensor` / `zero_tensor` / `copy_tensor`**: Standard resource management and wrapper functions over `cudaMemcpy` and `cudaMemset`.

---

### **File: `activations/activations.h` & `activations/activations.cu`**

#### File Purpose
Implements forwards and backwards for element-wise nonlinear activations. It is separated to isolate massive amounts of boilerplate kernel definitions from routing logic.

#### Design Pattern: The `EW_UNARY` Macro
*   **Function Existence Justification:** Many activations are 1D mapping operations mapping $x_i \mapsto y_i$. Writing identical memory calculation block loops 20 times is a maintenance risk.
*   **Macro Mechanics:** The macro wraps kernel launches. It takes the function name, forward kernel, backward kernel, and the required saved argument.
*   **Alternatives Analysis:** C++ templates could be used instead (`launch_ew<ReluFunctor>`). However, C++ device functors require `__device__` annotations, leading to complex template instantiations. Macros allow simple discrete specific kernels to be defined cleanly without complex type traits.

#### GPU Kernels (Deep Dive)

1.  **ReLU (`k_relu_fwd`, `k_relu_bwd`)**
    *   **Logic:** Block-strided 1D loop. `fmaxf(0.0f, x[i])`. Backward routes `dy` if `x > 0`.
    *   **Memory Flow:** Pure global memory reads. Zero shared memory needed since there is no inter-thread dependency.
2.  **Sigmoid (`sigmoid_dev` and Swish/SiLU variants)**
    *   **Hardware specifics:** Uses `expf` (hardware fast math). Note the macro implementation for `softplusf_dev` employs stable math: `x > 0 ? x + log1pf(expf(-x)) : log1pf(expf(x))`. Without this branch, `exp(x)` for $x=80$ produces infinity, resulting in NaNs. 
3.  **GELU (`k_gelu_fwd`)**
    *   **Logic:** Implements the classic $0.5x(1+\tanh(\sqrt{2/\pi}(x+0.044715x^3)))$ approximation.
    *   **Alternatives:** Pre-computed lookup tables or the exact `erf()` calculation. Rejected because `erf()` is highly expensive on the ALU, whereas `tanhf` maps to fast GPU SFU (Special Function Unit) instructions, improving throughput immensely.

4.  **Softmax / LogSoftmax (The hardest activation)**
    *   **Why custom warp logic instead of cuDNN?** cuDNN supports Softmax, but a highly tuned custom row-reduction kernel reduces kernel overhead and avoids relying strictly on cuDNN descriptors.
    *   **Kernel Configuration:** Uses 2D grid allocation where `blockIdx.x` is the row. `threadIdx.x` represents elements in the column.
    *   **Synchronization Decisions (`__syncthreads()`):**
        *   Step 1: Each thread finds maximum in its mapped columns, saves to shared memory (`extern __shared__ float smem[]`), then synchronizes.
        *   Step 2: Performs a parallel tree-reduction `s >>= 1` to find the absolute maximum of the row. This prevents catastrophic cancellation or exponent overflow in $e^x$.
        *   Step 3: Calculates sum of $e^{x - max}$, synchronizes, does another tree-reduction for the sum.
        *   Step 4: Normalizes.
    *   **Memory Hierarchy Optimization:** Utilizing dynamically allocated `extern __shared__ float smem[]` avoids making intermediate trips to global memory, maintaining the entire reduction operation in ultra-fast L1 cache equivalents.

#### Method Signatures in Activations
*   `void softmax_backward(const Tensor* grad_output, const Tensor* output, Tensor* grad_input)`
*   **Signature Design:** Unlike `ReLU`, the `Softmax` derivative heavily reuses the *output* probabilities ($y_i(\delta_{ij} - y_j)$). Therefore, `output` is explicitly required as a parameter, informing the future autograd exactly what to stash.

## 3. File-Level Breakdown & 4. Full Code Walkthrough (Continued)

### **Module: `norms/` (Normalization Layers)**

#### File Purpose (`norms.h`, `batchnorm.cu`, `layernorm.cu`, `rmsnorm.cu`, `groupnorm.cu`)
Normalization is mathematically heavy. Unlike simple activations mapping an element to another element independently, normalization requires computing aggregated statistics (mean, variance) across specific slices of a tensor, synchronizing those reductions, and applying them. The separation is architectural: each file contains specialized kernel logic depending strictly on the physical layout (axis of reduction).

#### Design Pattern: Explicit State Forwarding
*   **The Problem:** Normalization layers need their forward-pass mean and variance to compute exact gradients during the backward pass. PyTorch's `Function::apply` magically tucks these away inside the node graph context.
*   **The Framework's Solution:** Every `forward` function takes `Tensor* saved_mean, Tensor* saved_inv_std` as explicit parameters. The *caller* is responsible for allocating these buffers. When the autograd graph is implemented later, it will capture these explicitly passed pointers.

| Norm Layer | Reduction Dimensionality | Saved State | Feature |
| :--- | :--- | :--- | :--- |
| `BatchNorm1d` | Over $B$ (Batch) per feature. | `mean`, `inv_std` | 1 thread block per feature. |
| `BatchNorm2d` | Over $B \times H \times W$ per channel. | `mean`, `inv_std` | Strided reads across spatial dimensions. |
| `LayerNorm` | Over the last dimension (Feature). | `mean`, `inv_std` | Requires mapping threads per row. |
| `RMSNorm` | Over the last dimension (Feature). | `rrms` | No mean subtraction, saving calculations. |
| `GroupNorm` | Over $C/G \times H \times W$. | `mean`, `inv_std` | Combines Batch and Layer norm properties. |

#### Full Walkthrough: `layernorm.cu` (Representative Example)
1.  **Existence Justification:** `LayerNorm` normalizes inputs across the last dimension, making it completely agnostic to batch size constraints. It is an absolute requirement for Transformers.
2.  **Kernel Design (`ln_fwd`)**:
    *   **Mapping:** 1 Block = 1 Row (i.e., sequence token). `cols` represents the hidden dimension.
    *   **Execution Strategy:**
        *   Calculates $\sum x$ and $\sum x^2$ cooperatively in a single pass holding partials in variables, storing intermediate block sums in dynamically allocated `extern __shared__ float smem[]`.
        *   Uses a binary tree reduction (`s >>= 1`) inside shared memory to securely sum across the warp block.
        *   Avoids a second pass over global memory to calculate the variance because $\text{Var} = E[X^2] - (E[X])^2$.
    *   **Alternative Rejected:** Doing two separate kernel launches (one for mean, one for variance). This would require writing the mean to VRAM, then reading the entire $X$ tensor again from VRAM. Deep learning bottlenecks at VRAM bandwidth; computing both moments in a single fused pass using shared memory prevents an expensive roundtrip.
3.  **Kernel Design (`ln_bwd_dx`)**:
    *   Implements the notoriously complex analytical derivative of LayerNorm.
    *   Calculates the required sums: $\sum (dy \cdot \gamma)$ and $\sum (dy \cdot \gamma \cdot \hat{x})$.
    *   Applies the exact chain rule without retaining intermediate $\frac{d\mu}{dx}$ arrays in memory.

---

### **Module: `layers/` (Core Matrix Operators)**

#### File Purpose (`layers.h`, `linear.cu`, `conv.cu`, `pool.cu`)
These layers perform heavy multiply-accumulate operations mapping inputs to outputs via learnable dense weight tensors. Because writing highly efficient GEMM (General Matrix Multiply) and Convolution kernels is extremely specialized and hardware-specific, this module delegates the heavy lifting entirely to hardware-vendor libraries: **NVIDIA cuBLAS and cuDNN**.

#### Full Walkthrough: `linear.cu`
1.  **Existence Justification:** Implements the fundamental dense $Y = XW^T + b$ layer.
2.  **Internal Framework Hook (`linear_forward`)**:
    *   **Implementation:** Relies directly on `cublasSgemm`. Linear operations are fundamentally matrix multiplications.
    *   **Mathematics to Memory:** The math notation is $Y = XW^T$. However, $X$ is stored row-major in C++, but cuBLAS interprets pointers intrinsically as column-major.
    *   **The Flip Trick:** Instead of transposing physical memory, the framework leverages matrix identity $(AB)^T = B^T A^T$.
    *   Since cuBLAS thinks the row-major $X$ (shape `[B, I]`) is actually column-major $X^T$ (shape `[I, B]`), we tell cuBLAS to compute $W X^T$. The resulting column-major matrix corresponds perfectly to the physical layout of row-major $XW^T$. This completely avoids any expensive $O(N^2)$ physical memory transposition.
3.  **Alternative:** Writing a custom tiled shared-memory GEMM kernel. Rejected because NVIDIA's cuBLAS employs closed-source, specialized assembly instructions targeting specific hardware tensor cores. No open-source kernel can match its throughput trivially.
4.  **Bias Logic:** The bias is added in a separate, lightweight custom kernel (`add_bias_kernel`) via a shape broadcast.

#### Full Walkthrough: `conv.cu` & `pool.cu` (The Descriptor Pattern)
1.  **Existence Justification:** Image-based tasks require 2D spatial feature extraction.
2.  **State Management (`ConvDescriptor`, `PoolDescriptor`)**:
    *   **Why a struct?** cuDNN operates primarily via "Descriptors". A tensor descriptor describes the multidimensional striding of memory. A convolution descriptor explains pad/stride rules.
    *   **Lifecycle:** The framework separates `create_conv_descriptor` from `conv2d_forward`. Descriptor building is exceptionally slow (it may involve benchmarking hardware capabilities to choose the best algorithm, explicitly done via `cudnnGetConvolutionForwardAlgorithm_v7`). The caller allocates the descriptor *once*, maintaining abstraction over the C-style API.
    *   **Memory Management:** The descriptor dynamically requests a `workspace_size` buffer from the GPU. Instead of allocating/freeing this per pass, it resides in the descriptor, maintaining the zero-alloc-on-forward philosophy.
3.  **Forward / Backward Mapping:** These directly map to `cudnnConvolutionForward`, `cudnnConvolutionBackwardData`, and `cudnnConvolutionBackwardFilter`.

---

### **Module: `losses/`**

#### File Purpose (`losses.h`, `losses.cu`)
Loss functions reduce high-dimensional prediction tensors against targets to calculate a final objective scalar (loss) mapping the model's error, and then compute the mathematical derivative of that objective with respect to the prediction.

#### Internal Design Pattern
*   All functions rely on a fundamental design: A templated `reduce_kernel` functor.
*   **Why a Template/Functor?** MSE, L1, Huber, and BCE all perform a block pattern: Element-wise operation $F(P_i, T_i)$ followed by a global array summation reduction. Writing 4 reduction kernels is redundant. Instead, the framework passes a stateless struct `struct MseFn { __device__ float operator()(float p, float t) };` into `reduce_and_mean`.
*   **Global Reduction Logic:** 
    *   Threads compute their elements.
    *   Block reduces to a single float in thread 0.
    *   Thread 0 uses `atomicAdd(out, smem[0])` to safely accumulate the partial sum into a global variable.
*   **Alternative:** cuBLAS `cublasSasum`. Rejected because `cublasSasum` does absolute sum, not custom arbitrary pointwise lambda mapping before sum.

#### Specific Loss Walkthroughs
1.  **`cross_entropy_loss`**
    *   Handles $LogSoftmax + NLL$ mathematically fused.
    *   **Design Decision:** Targets are `int* targets_device`, an explicit 1D array of class indices rather than dense one-hot vectors. This saves an enormous amount of VRAM (e.g., $B \times \text{Vocab}$ bytes down to $B$ bytes).
    *   It manually executes a row-wise LogSumExp (similar to Softmax activation) minus the logit value specifically located at the target index.
2.  **`bce_with_logits_loss`**
    *   **Numerical Stability Justification:** Standard BCE computes $-\log(\sigma(x))$. If $x$ is highly negative, $\sigma(x) = 0$, leading to $\log(0) = -\infty$.
    *   The framework rewrites the math as `max(x, 0) - x*y + log(1 + exp(-abs(x)))` in a custom `BceFn`. This completely evades exponent overflow and zero logarithms while remaining mathematically identical.

## 5. Data Flow Analysis (Execution Path)

#### Tensor Lifecycle (Creation $\rightarrow$ Usage $\rightarrow$ Deletion)
1. **Creation:** User calls `create_tensor(shape, ndim, requires_grad)`. The host allocates a thin C-struct tracking metadata. `cudaMalloc` is invoked exactly once for the data array. If `requires_grad=true`, a zeroed buffer identically sized to `data` is instantiated via `cudaMalloc` and assigned to `t->grad`.
2. **Execution:** Operations operate purely on pointers. A layer reads `input->data`, applies a transformation, and writes to `output->data`.
3. **Destruction:** `free_tensor(t)` explicitly frees `grad`, `data`, and the struct. Standard RAII does not govern destruction, demanding strict memory hygiene.

#### The Forward Pass Path
Following `examples/mlp.cpp`, a forward pass consists of sequenced synchronous calls. 
1. `linear_forward(...)` invokes `cublasSgemm` asynchronously on the hardware stream but explicitly invokes `cudaDeviceSynchronize()` at the boundary or leaves cuBLAS to sequence it.
2. Intermediates (like $H_1$, $H_1\_ln$) must be pre-allocated. Functions map input arrays precisely to output boundaries.
3. Every step saves context (e.g., `ln_mean`, `ln_istd`) explicitly into passed pointers if the operator computes statistics.

#### The Backward Pass Path
The backward tape is traversed in reverse manual topological order.
1. The loss function derivative initiates the chain `cross_entropy_backward`. Note that it writes directly into `d_logits`.
2. `linear_backward` applies the linear layer derivatives. Because $Y = XW^T + b$, $dX = dY \cdot W$ and $dW = dY^T \cdot X$. `cublasSgemm` is invoked with `CUBLAS_OP_T` appropriately to perform matrix multiplication directly into `d_h1_a` and `dW2`.
3. Backpropagating through `layernorm_backward` requires the `ln_mean` and `ln_istd` that were populated during the forward pass. This guarantees exact, bit-identical precision with no re-computation.

---

## 6. Cross-File and System Interactions

* **The Execution Context (`core/context.h`):** The `NNContext` travels through the `layers/` module exclusively. `activations` and `norms` do not require cuBLAS/cuDNN and thus bypass the context. This allows deep splitting of system components.
* **Autograd Hooking (`core/op.h`):** The entire framework operates dynamically via explicit arguments. A future autograd engine needs only to intercept calls to `forward`. The engine will allocate the `saved` intermediate tensors behind the scenes, record the backward function pointer into `AutogradNode`, and release the user from writing explicit `_backward` calls. 

---

## 7. Design Philosophy and Tradeoffs

### Why this architecture was chosen?
UltraML chooses C-style explicit functional purity over C++ OOP complexity. There is no `torch::nn::Linear` class that encapsulates state. Objects do not hold their parameters or their activation buffers. This stateless approach creates a pristine environment for analyzing mathematical operations and drastically simplifies the underlying memory model. If an error occurs, it is explicitly tied to a function line, not hidden behind multiple layers of inheritance dispatch.

### Tradeoffs vs PyTorch

1. **Explicit Memory Allocation (UltraML) vs Implicit (PyTorch):** 
    *   *UltraML:* Forces the user to allocate `h1`, `h1_ln`, `h1_a`. *Benefit:* No memory leaks, perfect tracking of VRAM, opportunity for static memory planning (reusing buffers). *Cost:* Extremely verbose code.
    *   *PyTorch:* A `forward` call returns a dynamically allocated tensor. *Benefit:* Development speed. *Cost:* Deep internal caching structures (c10 allocator) required to prevent massive `cudaMalloc` stalls.

2. **Dispatcher Complexity vs Direct Calling:**
    *   *PyTorch:* Utilizes a massive generic `Dispatcher` that checks types, devices (CPU vs CUDA), and autodiff flags before mapping to a backend kernel. 
    *   *UltraML:* directly calls strongly-typed `.cu` code. *Benefit:* Zero dispatch overhead. A forward pass in UltraML is mathematically bounded by GPU capabilities alone, devoid of CPU routing latency.

3. **cuDNN vs Handwritten Kernels:**
    *   UltraML heavily relies on handwritten custom reductions for Normalization and Softmax. 
    *   *Why?* While cuDNN offers normalization, extracting exact backward statistics predictably across varied GPU architectures is notoriously brittle inside cuDNN descriptors. Handwritten kernels (like `bn2d_stats`) guarantee $100\%$ transparency on memory access patterns. Convolution, conversely, is left to cuDNN because Winograd/Implicit-GEMM convolution algorithms are fundamentally closed-source IP optimized at the assembly level.

### Known Limitations
* **FP32 Only:** The entire framework is hard-coded to `float`. Moving to FP16 or BF16 for tensor cores would require templating the kernels and restructuring the `Tensor` struct to carry a `dtype`.
* **Zero Fusions:** Operations are strictly serial. PyTorch uses `nvfuser` or `Triton` to fuse `Linear + ReLU` into one kernel, preventing the $H_1$ intermediate from being physically written to global memory. UltraML must write every intermediate to VRAM, making it memory-bandwidth bound on small batch training.
