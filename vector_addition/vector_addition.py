import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, z_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    blockidx = tl.program_id(axis=0)
    offsets = blockidx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)

    z = x + y

    tl.store(z_ptr + offsets, z, mask=mask)




def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    assert x.dtype == y.dtype, "Input tensors must have the same dtype"

    z = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (grid,)

    add_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return z


def verify_numeric():
    torch.manual_seed(4747)
    N = 1024
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.rand_like(x)
    z = add(x, y)
    z_ref = x + y

    assert torch.allclose(z, z_ref), "Results do not match!"
    print("Success! The results match.")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],  # argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(10, 28)],  # different values of x to plot
        x_log=True,  # x axis is logarithmic
        line_arg='provider',  # argument name whose value corresponds to a different line in the plot
        line_vals=['triton', 'torch'],  # values of the line argument
        line_names=['Triton', 'PyTorch'],  # name of the lines
        styles=[('blue', '-'), ('green', '-')],  # line styles
        ylabel='GB/s',  # label name for the y-axis
        plot_name='vector-addition-performance',  # name of the plot
        args={} # other arguments for the function to benchmark
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y = torch.rand_like(x)
    quantiles = [0.5, 0.2, 0.8]

    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    else:
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)

    gbps = lambda ms: 12 * N / ms / 1e6
    return gbps(ms), gbps(min_ms), gbps(max_ms)

if __name__ == "__main__":
    verify_numeric()
    benchmark.run(show_plots=True, print_data=True, save_path="./vec_add_perf")
