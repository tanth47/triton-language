import torch
import triton
import triton.language as tl



@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):

    blockidx_m = tl.program_id(0)
    blockidx_n = tl.program_id(1)

    row_offsets = blockidx_m * BLOCK_M + tl.arange(0, BLOCK_M)
    col_offsets = blockidx_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        # flatten the pointers
        # row_offsets[:, None] * K ~ (0, K, 2K, ..., (BLOCK_M-1)*K)
        # (k + tl.arange(0, BLOCK_K))[None, :] ~ (k, k+1, k+2, ..., k+BLOCK_K-1)
        a_ptrs = A_ptr + row_offsets[:, None] * K + (k + tl.arange(0, BLOCK_K))[None, :]
        
        # flatten the pointers
        b_ptrs = B_ptr + (k + tl.arange(0, BLOCK_K))[:, None] * N + col_offsets[None, :]

        a_tile = tl.load(a_ptrs, mask=(row_offsets[:, None] < M) & ((k + tl.arange(0, BLOCK_K))[None, :] < K), other=0.0)
        b_tile = tl.load(b_ptrs, mask=((k + tl.arange(0, BLOCK_K))[:, None] < K) & (col_offsets[None, :] < N), other=0.0)

        acc += tl.dot(a_tile, b_tile)

    c_ptrs = C_ptr + row_offsets[:, None] * N + col_offsets[None, :]
    tl.store(c_ptrs, acc, mask=(row_offsets[:, None] < M)
            & (col_offsets[None, :] < N))


def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    assert A.shape[1] == B.shape[0], "Incompatible matrix dimensions"
    assert A.is_cuda and B.is_cuda, "Input tensors must be on CUDA device"
    assert A.dtype == B.dtype, "Input tensors must have the same dtype"
    assert len(A.shape) == 2 and len(B.shape) == 2, "Input tensors must be 2D matrices"


    M, K = A.shape
    K, N = B.shape

    C = torch.empty((M, N), device=A.device, dtype=A.dtype)

    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid = (grid_m, grid_n)

    matmul_kernel[grid](
        A_ptr=A,
        B_ptr=B,
        C_ptr=C,
        M=M,
        N=N,
        K=K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return C


def verify_numeric():
    torch.manual_seed(4747)
    M, K, N = 512, 512, 512
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)

    C_triton = matmul(A, B)
    C_torch = torch.matmul(A, B)

    if torch.allclose(C_triton, C_torch, atol=1e-3):
        print("Numeric verification passed!")
    else:
        print("Numeric verification failed!")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['M', 'N', 'K'],  # argument names to use as
        x_vals=[2**i for i in range(8, 16)],  # values of `x_name` to plot
        line_arg='provider',  # argument name whose value corresponds to a line
        line_vals=['triton', 'torch'],  # values of `line_arg` to plot
        line_names=['Triton', 'PyTorch'],  # display names for the lines
        styles=[('blue', '-'), ('green', '--')],
        ylabel="GB/s",  # label name for the y-axis
        plot_name="matrix-multiplication",  # name for the plot
        args={},
    )
)
def benchmark(M, N, K, provider):
    A = torch.randn((M, K), device='cuda', dtype=torch.float32)
    B = torch.randn((K, N), device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(A, B), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles)

    gbps = lambda ms: 4 * M * N * K / ms / 1e6
    return gbps(ms), gbps(min_ms), gbps(max_ms)


if __name__ == "__main__":
    verify_numeric()
    benchmark.run(print_data=True, show_plots=True, save_path="./matmul_perf")
