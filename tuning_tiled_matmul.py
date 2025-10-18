from itertools import product
import torch

import triton
import triton.language as tl

from tiled_matmul import matmul as naive_triton_matmul

DEVICE = 'cuda:0'


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]

def get_hip_autotune_config():
    # sizes = [
    #     {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_SIZE_M': 6},
    #     {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4},
    #     {'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 6},
    #     {'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 6},
    #     {'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4},
    #     {'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4},
    #     {'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4},
    #     {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 6},

    #     {'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 6},
    # ]

    # return [triton.Config(s | {'matrix_instr_nonkdim': 16}, num_warps=8, num_stages=2) for s in sizes]

    block_m = [32, 64, 128, 256]
    block_n = [32, 64, 128, 256]
    block_k = [16, 32, 64, 128]
    group_size_m = [4, 6, 8, 16]
    num_warps_ = [2, 4, 8]
    num_stages_ = [1, 2]

    # use itertools product to generate all combinations
    configs = []
    for b_m, b_n, b_k, g_m, n_w, n_s in product(block_m[1:], block_n[:1], block_k[:-1], group_size_m[::2], num_warps_, num_stages_[:-1]):
        configs.append(triton.Config({'BLOCK_M': b_m, 'BLOCK_N': b_n, 'BLOCK_K': b_k, 'GROUP_SIZE_M': g_m}, num_warps=n_w, num_stages=n_s))

    return configs

def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_cm: tl.constexpr, stride_cn: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, 
    ACTIVATION: tl.constexpr
):

    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    # Number of program ids along the M axis
    num_pid_m = tl.cdiv(M, BLOCK_M)
    # Number of programs ids along the N axis
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # Number of programs in group
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # Id of the group this program is in
    group_id = pid // num_pid_in_group
    # Row-id of the first program in the group
    first_pid_m = group_id * GROUP_SIZE_M
    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # *Within groups*, programs are ordered in a column-major order
    # Row-id of the program in the *launch grid*
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # Col-id of the program in the *launch grid*
    pid_n = (pid % num_pid_in_group) // group_size_m


    offset_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offset_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offset_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=(offset_am[:, None] < M) & (offset_k[None, :] < K - k * BLOCK_K), other=0.0)
        b = tl.load(b_ptrs, mask=(offset_k[:, None] < K - k * BLOCK_K) & (offset_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # You can fuse arbitrary activation functions here
    if  ACTIVATION == "leaky_relu":
        acc = leaky_relu(acc)

    acc = acc.to(tl.bfloat16)

    c_ptrs = C_ptr + (offset_am[:, None] * stride_cm + offset_bn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offset_am[:, None] < M) & (offset_bn[None, :] < N))


@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(A: torch.Tensor, B: torch.Tensor, activation: str = None) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    assert A.is_cuda and B.is_cuda, "Input tensors must be on CUDA device"
    assert A.dtype == B.dtype, "Input tensors must have the same dtype"
    assert len(A.shape) == 2 and len(B.shape) == 2, "Input tensors must be 2D matrices"

    M, K = A.shape
    K, N = B.shape

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    # BLOCK_M = 128
    # BLOCK_N = 128
    # BLOCK_K = 32
    # GROUP_SIZE_M = 8
    # grid = ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N)
    # grid = (grid,)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), )

    matmul_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        N=N,
        K=K,
        stride_am=A.stride(0),
        stride_ak=A.stride(1),
        stride_bk=B.stride(0),
        stride_bn=B.stride(1),
        stride_cm=C.stride(0),
        stride_cn=C.stride(1),
        # BLOCK_M=BLOCK_M,
        # BLOCK_N=BLOCK_N,
        # BLOCK_K=BLOCK_K,
        # GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=activation if activation is not None else "none"
    )

    return C

def verify_numeric():
    torch.manual_seed(4747)
    M, K, N = 512, 512, 512
    A = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    B = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)

    C_triton = matmul(A, B)
    C_torch = torch.matmul(A, B)
    C_naive_triton = naive_triton_matmul(A, B)

    assert torch.allclose(C_triton, C_torch, atol=1e-3), (C_torch, C_triton)
    assert torch.allclose(C_naive_triton, C_torch, atol=1e-3), (C_torch, C_naive_triton)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # argument names to use as
        x_vals=[2**i for i in range(8, 12)],  # values of `x_name` to plot
        line_arg='provider',  # argument name whose value corresponds to a line
        line_vals=['triton', 'torch', 'naive_triton'],  # values of `line_arg` to plot
        line_names=['Triton', 'PyTorch', 'Naive Triton'],  # display names for the lines
        styles=[('blue', '-'), ('green', '--'), ('orange', ':')],
        ylabel="GB/s",  # label name for the y-axis
        plot_name="matrix-multiplication",  # name for the plot
        args={},
    )
)
def benchmark(M, N, K, provider):
    A = torch.randn((M, K), device='cuda', dtype=torch.bfloat16)
    B = torch.randn((K, N), device='cuda', dtype=torch.bfloat16)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(A, B), quantiles=quantiles)
    elif provider == 'naive_triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: naive_triton_matmul(A, B), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles)

    gbps = lambda ms: 4 * M * N * K / ms / 1e6
    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    verify_numeric()
    benchmark.run(print_data=True, show_plots=True, save_path="./tunning_matmul_perf")
