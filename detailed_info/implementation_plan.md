# UltraML Technical Document Generation Plan

This plan outlines the approach to generate the exhaustive, microscopic-level technical document for the UltraML C/C++ CUDA-based deep learning framework.

## User Review Required

> [!IMPORTANT]
> The final document will be extremely long, breaking down the entire codebase line-by-line. Given the size of the codebase (~15-20 files, ranging up to 15KB each), it may exceed output constraints if generated in a single pass. I plan to construct the artifact iteratively, section by section, ensuring no detail is skipped.
> Please approve this approach. 

## Proposed Approach

I will iteratively read the contents of each directory and file, analyze them, and append/write the contents to a single artifact `ultraml_technical_document.md`.

The execution will follow the mandated structure:
1. **High-Level System Overview**
2. **Directory-Level Breakdown**
3. **File-Level Breakdown** (Purpose, Dependencies)
4. **Full Code Walkthrough** (Structs, Functions, Kernels, Macros)
5. **Data Flow Analysis**
6. **Cross-File and System Interactions**
7. **Design Philosophy and Tradeoffs**

### Phases

#### Phase 1: Core System & Architecture
- Analyze `README.md`, `CMakeLists.txt`, `ultraml.h`
- Analyze `core/` (`macros.h`, `context.h/.cu`, `op.h`, `tensor.h/.cu`)
- Write Sections 1, 2, and the beginning of 3/4.

#### Phase 2: Operations and Neural Network Components
- Analyze `activations/` (`activations.h/.cu`)
- Analyze `norms/` (`norms.h`, `batchnorm.cu`, `layernorm.cu`, `rmsnorm.cu`, `groupnorm.cu`)
- Analyze `layers/` (`layers.h`, `linear.cu`, `conv.cu`, `pool.cu`)
- Analyze `losses/` (`losses.h/.cu`)
- Continue appending deep breakdowns to the document.

#### Phase 3: Final Integrations & Tradeoffs
- Analyze `examples/mlp.cpp`
- Complete Sections 5 (Data Flow), 6 (Interactions), and 7 (Philosophy and Tradeoffs).

## Open Questions

> [!WARNING]
> Is there a specific file or component you want me to emphasize (e.g., cuDNN integration vs custom kernels)? 
> Are you perfectly fine with an artifact that might be incredibly long (e.g., 2000+ lines of Markdown)?

## Verification Plan

- Ensure every file in the directory tree is documented.
- Verify that every listed requirement (functions, structs, kernels, macro justifications) is met for every file analyzed.
