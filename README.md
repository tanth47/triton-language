# Triton Kernel Tutorials (with AMD MI300X Notes)

This repo started as my personal Triton playground and is now turning into a small tutorial series on writing and optimizing GPU kernels with [Triton](https://triton-lang.org/), with a special focus on AMD MI300X (ROCm) but still useful on CUDA GPUs.

The goal: go from **“hello world” vector addition** → **Flash Attention kernels** while learning:

- Triton’s programming model (`program_id`, `BLOCK_*`, `num_warps`, `num_stages`, …)
- How GPU hardware (SM/CU, registers, shared memory/LDS, caches) constrains kernel design
- How to reason about **occupancy**, **tiling**, **memory access patterns**, and **auto-tuning**
- How Triton lowers to IR/ISA and how that compares to CUDA/HIP (comming soon)

If you already know PyTorch and basic GPU concepts but want to actually *read and write* Triton kernels, this repo is for you.

---

## Table of Contents / Learning Path

### Part I – Getting Started

1. **Vector Addition – Hello Triton**  
   **Folder:** [`vector_addition/`](./vector_addition)  
   - Simplest possible kernel: `C = A + B`
   - Covers:
     - `tl.program_id`, `tl.arange`, `tl.load`, `tl.store`
     - Masks & bounds checking
     - Basic benchmarking & correctness checks

2. **Intro to Triton (Concepts)** – _planned_  
   **(coming soon: doc/notebook)**  
   - What is Triton? Why not just CUDA?  
   - Triton’s execution model: program vs block vs thread  
   - Reserved keywords & knobs: `BLOCK_*`, `num_warps`, `num_stages`  
   - How Triton maps to GPU hardware (very high level)

---

### Part II – Core Kernels

3. **Softmax**  
   **Folder:** [`softmax/`](./softmax)  
   - Row-wise softmax implementation
   - Topics:
     - Numerically stable softmax (`x - max` trick)
     - Blocking along the row dimension
     - Measuring performance
     - A second version that looks at **occupancy** (registers, shared mem, block size)

4. **Matmul (Tiled & Auto-Tuned)**  
   **Folder:** [`matmul/`](./matmul)  
   - From naive matmul to tiled matmul
   - Topics:
     - Tiling with `BLOCK_M`, `BLOCK_N`, `BLOCK_K`
     - Exploiting L2 cache and shared memory / LDS
     - Using `@triton.autotune` to pick good configs
     - How different block sizes affect throughput and occupancy

5. **LayerNorm Variants**  
   **Folder:** [`layer_norm/`](./layer_norm)  
   - Several implementations of LayerNorm:
     1. “Normal” single-pass version  
     2. 3-stage version: partition → partial reduction → final reduction  
     3. Atomic-ops-based version  
   - Topics:
     - Reduction patterns
     - Numerical stability
     - Trade-offs between simplicity, shared memory usage, and atomics

---

### Part III – Deep Dive & Advanced Topics

6. **Triton Compilation Stages**  
   **Notebook:** [`triton_compilation_stages.ipynb`](./triton_compilation_stages.ipynb)  
   - Walkthrough of Triton’s compilation pipeline for a simple kernel
   - Inspect:
     - Triton IR
     - Lowered IR (TTGIR / LLVM IR)
     - How high-level ops like `tl.dot` turn into lower-level instructions
   - Goal: make the IR less “black-boxy”

7. **Optimizing Triton Kernels on AMD MI300X**  
   **Notebook:** [`amd_optimizing_triton_kernel.ipynb`](./amd_optimizing_triton_kernel.ipynb)  
   - Notes & experiments based on AMD’s optimization guides
   - Topics:
     - MI300X architecture highlights (CUs, SIMDs, LDS, VGPRs)
     - Occupancy math (LDS-limited vs VGPR-limited)
     - Choosing `num_stages`, `num_warps`, tile sizes for MI300X
     - Measuring throughput and interpreting results

8. **Demystified Triton - Triton vs CUDA/HIP at IR level**  _planned_  
   **(coming soon: doc/notebook)**   
   - Side-by-side look at IR generated for a simple kernel
   - Compare Triton IR / LLVM IR and reason about what the compiler is doing

9. **Flash Attention (2D & 3D)**  
   **Folder:** [`flash_attention/`](./flash_attention)  
   - Implementation experiments for Flash Attention-style kernels
   - Topics:
     - 2D kernel: attention over (batch, sequence) with tiling in Q/K/V
     - 3D kernel: partition sequence -> similar compute with 2D version -> reduce segments
     - Fusing matmul + softmax + value matmul

---

## Requirements & Setup

You’ll need:

- Python 3.10+ (or similar)
- [Triton](https://triton-lang.org/) installed, expect version 3.4.0
- A supported GPU:
  - AMD MI300X / ROCm **(main testing target)**  
  - or CUDA GPU (many examples should still work, but MI300X-specific notes obviously won’t apply 1:1)

