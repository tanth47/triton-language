import torch

import triton
import triton.language as tl


DEVICE = "cuda:0"

############################### LayerNorm 1 Stage ###############################

@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    Y,  # pointer to the output
    W,  # pointer to the weights
    B,  # pointer to the biases
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    N,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write output
        tl.store(Y + cols, y, mask=mask)


def layer_norm_1_stage(x, normalized_shape, weight, bias, eps):
    # allocate output
    y = torch.empty_like(x)
    # reshape input data into 2D tensor
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape
    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)
    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    # if N > BLOCK_SIZE:
    #     raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
    # heuristics for number of  warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)
    # enqueue kernel
    _layer_norm_fwd_fused[(M, )](  #
        x_arg, y, weight, bias, mean, rstd,  #
        x_arg.stride(0), N, eps,  #
        BLOCK_SIZE=BLOCK_SIZE, 
        num_warps=num_warps, 
        num_ctas=1)

    return y

############################### LayerNorm 3 Stages ###############################

@triton.jit
def _layer_norm_stats_tiles(
    X,
    partial_sum,
    partial_sum_sq,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    row_offset = row_id * stride
    cols = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + row_offset + cols, mask=cols < N, other=0.).to(tl.float32)
    sum = tl.sum(x, axis=0)
    sum_sq = tl.sum(x * x, axis=0)

    idx = row_id * tl.num_programs(1) + tile_id
    tl.store(partial_sum + idx, sum)
    tl.store(partial_sum_sq + idx, sum_sq)

@triton.jit
def _layer_norm_stats_reduce(
    partial_sum,
    partial_sum_sq,
    Mean,
    Rstd,
    N: tl.constexpr,
    T: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    row_id = tl.program_id(0)
    sum = 0.0
    sum_sq = 0.0
    for t in range(0, T, BLOCK_SIZE_T):
        tile_id = t + tl.arange(0, BLOCK_SIZE_T)
        mask = tile_id < T
        idx = row_id * T + tile_id
        p_sum = tl.load(partial_sum + idx, mask=mask, other=0.0)
        p_sum_sq = tl.load(partial_sum_sq + idx, mask=mask, other=0.0)
        sum += tl.sum(p_sum, axis=0)
        sum_sq += tl.sum(p_sum_sq, axis=0)
    mean = sum / N
    var = (sum_sq - 2 * mean * sum + N * mean * mean) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row_id, mean)
    tl.store(Rstd + row_id, rstd)

@triton.jit
def _layer_norm_normalize_kernel(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    row_offset = row_id * stride
    cols = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mean = tl.load(Mean + row_id)
    rstd = tl.load(Rstd + row_id)

    w = tl.load(W + cols, mask=cols < N)
    b = tl.load(B + cols, mask=cols < N)
    x = tl.load(X + row_offset + cols, mask=cols < N, other=0.).to(tl.float32)

    x_hat = (x - mean) * rstd
    y = x_hat * w + b

    tl.store(Y + row_offset + cols, y, mask=cols < N)


def layer_norm_3stages(
    X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float = 1e-5,
):
    """
    X: [M, stride>=N], contiguous in the last dim (or provide proper stride)
    W,B: [N], fp32 weights and biases recommended
    Returns: (Y)
    """
    assert X.dim() == 2
    M, stride = X.shape
    N = W.numel()
    assert B.numel() == N
    assert stride >= N

    device = X.device
    dtype  = X.dtype

    BLOCK_SIZE = 1024

    MAX_FUSED_SIZE = 65536 // X.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
    REDUCE_TILES_SIZE = BLOCK_SIZE // 2

    TILE_SIZE = BLOCK_SIZE
    T = (N + TILE_SIZE - 1) // TILE_SIZE

    # heuristics for number of  warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 8)

    # buffers
    partial_sum   = torch.empty((M * T,), device=device, dtype=torch.float32)
    partial_sumsq = torch.empty_like(partial_sum)
    Mean = torch.empty((M,), device=device, dtype=torch.float32)
    Rstd = torch.empty_like(Mean)
    Y    = torch.empty_like(X)  # compute in fp32; cast later if you want

    # grid shapes
    gridA = (M, T)                      # (row, tile)
    gridB = (M,)                        # per-row reduction
    gridC = (M, T)                      # (row, tile)

    # KERNEL A
    _layer_norm_stats_tiles[gridA](
        X, partial_sum, partial_sumsq,
        stride, N,
        BLOCK_SIZE=TILE_SIZE,
        num_warps=num_warps,  # good default for 256, tune on your GPU
        num_stages=1,
    )

    # KERNEL B
    _layer_norm_stats_reduce[gridB](
        partial_sum, 
        partial_sumsq,
        Mean, Rstd,
        N, T, eps,
        BLOCK_SIZE_T=REDUCE_TILES_SIZE,   # reduce tiles per iteration
        num_warps=4, num_stages=2,
    )

    # KERNEL C
    _layer_norm_normalize_kernel[gridC](
        X, Y, W, B, Mean, Rstd,
        stride, N,
        BLOCK_SIZE=TILE_SIZE,
        num_warps=num_warps, num_stages=1,
    )

    return Y

############################### LayerNorm 3 Stages (Atomic Ops) ###############################

@triton.jit
def _layer_norm_stats_tiles_atomic(
    X,
    sum_shard,
    sumsq_shard,
    stride,
    N,
    NUM_SHARDS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)

    row_offset = row_id * stride
    cols = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    x = tl.load(X + row_offset + cols, mask=cols < N, other=0.).to(tl.float32)

    sum = tl.sum(x, axis=0)
    sum_sq = tl.sum(x * x, axis=0)

    shard_id = tile_id % NUM_SHARDS

    idx = row_id * NUM_SHARDS + shard_id

    tl.atomic_add(sum_shard + idx, sum)
    tl.atomic_add(sumsq_shard + idx, sum_sq)

@triton.jit
def _layer_norm_stats_reduce_atomic(
    sum_shard,
    sumsq_shard,
    Mean,
    Rstd,
    N: tl.constexpr,
    NUM_SHARDS: tl.constexpr,
    eps: tl.constexpr,
):
    row_id = tl.program_id(0)
    shards = tl.arange(0, NUM_SHARDS)
    idx = row_id * NUM_SHARDS + shards
    sum = tl.sum(tl.load(sum_shard + idx), axis=0)
    sum_sq = tl.sum(tl.load(sumsq_shard + idx), axis=0)

    mean = sum / N
    var = (sum_sq - 2 * mean * sum + N * mean * mean) / N
    rstd = 1 / tl.sqrt(var + eps)
    tl.store(Mean + row_id, mean)
    tl.store(Rstd + row_id, rstd)

@triton.jit
def _layer_norm_normalize_kernel_atomic(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    stride,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    row_offset = row_id * stride
    cols = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    mean = tl.load(Mean + row_id)
    rstd = tl.load(Rstd + row_id)

    w = tl.load(W + cols, mask=cols < N)
    b = tl.load(B + cols, mask=cols < N)
    x = tl.load(X + row_offset + cols, mask=cols < N, other=0.).to(tl.float32)

    x_hat = (x - mean) * rstd
    y = x_hat * w + b

    tl.store(Y + row_offset + cols, y, mask=cols < N)
    

def layer_norm_3stages_atomic(
    X: torch.Tensor, W: torch.Tensor, B: torch.Tensor, eps: float = 1e-5,
):
    assert X.dim() == 2
    M, stride = X.shape
    N = W.numel()
    assert B.numel() == N
    assert stride >= N

    device = X.device

    MAX_FUSED_SIZE = 65536 // X.element_size()
    BLOCK_SIZE = min(MAX_FUSED_SIZE // 4, triton.next_power_of_2(N))
    NUM_BLOCKS = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    # NUM_SHARDS = 16 if N >= 16384 else (8 if N >= 8192 else 4)
    NUM_SHARDS = max(NUM_BLOCKS // 128, 32)
    # print(f"Using {NUM_SHARDS} shards for N={N}, BLOCK_SIZE={BLOCK_SIZE}")
    # heuristics for number of  warps
    num_warps = min(max(BLOCK_SIZE // 256, 1), 4)

    sum_shard = torch.zeros((M, NUM_SHARDS), device=device, dtype=torch.float32)
    sumsq_shard = torch.zeros_like(sum_shard)
    Mean = torch.empty((M,), device=device, dtype=torch.float32)
    Rstd = torch.empty_like(Mean)
    Y    = torch.empty_like(X)


    gridA = (M, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    _layer_norm_stats_tiles_atomic[gridA](
        X,
        sum_shard,
        sumsq_shard,
        stride, 
        N,
        NUM_SHARDS,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=1,
    )

    gridB = (M,)
    _layer_norm_stats_reduce_atomic[gridB](
        sum_shard,
        sumsq_shard,
        Mean, Rstd,
        N,
        NUM_SHARDS,
        eps,
        num_warps=4,
        num_stages=2,
    )

    gridC = (M, (N + BLOCK_SIZE - 1) // BLOCK_SIZE)
    _layer_norm_normalize_kernel_atomic[gridC](
        X, Y, W, B, Mean, Rstd,
        stride, N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps, 
        num_stages=1,
    )

    return Y

############################### Testing and Benchmarking ###############################

def test_layer_norm(M, N, dtype, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    # forward pass
    y_tri_1_stage = layer_norm_1_stage(x, w_shape, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    assert torch.allclose(y_tri_1_stage, y_ref, atol=1e-2, rtol=0)

    y_tri_3_stage = layer_norm_3stages(x, weight, bias, eps)

    assert torch.allclose(y_tri_3_stage, y_ref, atol=1e-2, rtol=0), f"{y_ref=}, {y_tri_3_stage=}"

    y_tri_3_stage_atomic = layer_norm_3stages_atomic(x, weight, bias, eps)

    assert torch.allclose(y_tri_3_stage_atomic, y_ref, atol=1e-2, rtol=0), f"{y_ref=}, {y_tri_3_stage_atomic=}"

    print(f"LayerNorm forward ({M=}, {N=}) test passed!")

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512 * i for i in range(2, 64)],
        line_arg='provider',
        line_vals=['triton-1stage', 'torch', 'triton-3stages', 'triton-3stages-atomic'],
        line_names=['Triton-1Stage', 'Torch', 'Triton-3Stages', 'Triton-3Stages-Atomic'],
        styles=[('blue', '-'), ('green', '-'), ('orange', '-'), ('red', '-')],
        ylabel='GB/s',
        plot_name='layer-norm-fwd',
        args={'M': 4096, 'dtype': torch.float16,},
    ))
def bench_layer_norm(M, N, dtype, provider, eps=1e-5, device=DEVICE):
    # create data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    quantiles = [0.5, 0.2, 0.8]

    def y_fwd():

        if provider == "triton-1stage":
            return layer_norm_1_stage(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "torch":
            return torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps)  # noqa: F811, E704

        if provider == "triton-3stages":
            return layer_norm_3stages(x, weight, bias, eps)
        
        if provider == "triton-3stages-atomic":
            return layer_norm_3stages_atomic(x, weight, bias, eps)
        
    # forward pass
    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=500)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    for i in range(2, 64):
        test_layer_norm(4096, 512 * i, torch.float32)
    bench_layer_norm.run(show_plots=True, print_data=True, save_path="layer_norm_fwd")