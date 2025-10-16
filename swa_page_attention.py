from typing import Optional, List

import torch
import aiter
from aiter.ops.triton.unified_attention import unified_attention as aiter_unified_attention
from vllm.attention.ops.triton_unified_attention import unified_attention as vllm_unified_attention

from aiter.test_common import checkAllclose, perftest

from vllm.attention.ops.merge_attn_states import merge_attn_states

from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op

from .scheduler_simulator import get_scheduler_output

######################################################
#  Prepare Inputs
######################################################
def create_inputs(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    sliding_window: Optional[int],
    dtype: torch.dtype,
    block_size: int,
    num_blocks: int,
    common_prefix_len: int = 0,
):
    query_lens = [l[0] for l in seq_lens]
    kv_lens = [l[1] for l in seq_lens]
    common_prefix_len = min(common_prefix_len, min(kv_lens)-1)
    num_seqs = len(query_lens)
    max_seqlen_q = max(query_lens)
    max_kv_len = max(kv_lens)
    max_seqlen_k = max_kv_len
    window_size = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    scale = head_size ** -0.5
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0

    cu_seqlens_q = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    cu_seqlens_k = torch.tensor([0] + kv_lens, dtype=torch.int32).cumsum(
        dim=0, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size

    if common_prefix_len > 0:
        common_prefix_len = (common_prefix_len // block_size) * block_size
        num_common_kv_blocks = common_prefix_len // block_size
    else:
        num_common_kv_blocks = 0

    number_of_unique_blocks = num_seqs * (max_num_blocks_per_seq - num_common_kv_blocks) + num_common_kv_blocks
    num_blocks = max(num_blocks, number_of_unique_blocks + 5)
    pool_value = torch.randperm(num_blocks-1)[:number_of_unique_blocks] + 1
    block_tables = torch.randint(
        1,
        num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    for i in range(num_seqs):
        block_tables[i][:num_common_kv_blocks].copy_(pool_value[:num_common_kv_blocks])
        offset = num_common_kv_blocks + i * (max_num_blocks_per_seq - num_common_kv_blocks)
        block_tables[i][num_common_kv_blocks:].copy_(
            pool_value[offset: offset + (max_num_blocks_per_seq - num_common_kv_blocks)]
        )

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    # first block is used for masking for certain cases (load and ignore rather than masked load)
    # for better testing, but nan there to make sure those values do not propagate during attn. calc.
    # if it propagates to the results, tests will fail for sure when done this way
    key_cache[0] = float("nan")
    value_cache[0] = float("nan")

    sinks = torch.randn(num_query_heads, dtype=torch.bfloat16)
    return (
        query,
        key_cache,
        value_cache,
        query_lens,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_lens,
        max_seqlen_q,
        max_seqlen_k,
        window_size,
        block_tables,
        scale,
        sliding_window,
        sinks,
    )

####################### Triton Unififed Attention(Begin) ###############################

@perftest(num_iters=5, num_warmup=1)
def run_aiter_unified(
    query,
    key_cache,
    value_cache,
    query_lens,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_lens,
    max_seqlen_q,
    max_seqlen_k,
    window_size,
    block_tables,
    scale,
    sliding_window,
    sinks,
):
    output = torch.empty_like(query)
    aiter_unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=kv_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
    )
    return output

@perftest(num_iters=5, num_warmup=1)
def run_vllm_unified(
    query,
    key_cache,
    value_cache,
    query_lens,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_lens,
    max_seqlen_q,
    max_seqlen_k,
    window_size,
    block_tables,
    scale,
    sliding_window,
    sinks,
):
    output = torch.empty_like(query)
    vllm_unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=kv_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        sinks=sinks,
    )
    return output

####################### Triton Unififed Attention(End) ###############################


####################### Paged Attention v1(Begin) ###############################
@triton.jit
def build_window_block_tables(
    block_tables_ptr,          # *int32 [B, max_blocks]
    kv_lens_ptr,               # *int32 [B]
    new_block_tables_ptr,      # *int32 [B, win_blocks]
    p1_kv_id_ptr,              # *int32 [B]
    p2_kv_id_ptr,              # *int32 [B]
    p1_size_ptr,               # *int32 [B]
    p2_size_ptr,               # *int32 [B]
    # need_move_ptr,             # *int8  [B]
    B: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    WIN_BLOCKS: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    i = tl.program_id(0)
    if i >= B:
        return

    # Load kv_len (supports either int32 or int64)
    kv_len_i = tl.load(kv_lens_ptr + i)

    # If kv_len < window_size -> nothing to do for this i
    if kv_len_i <= WINDOW_SIZE:
        # Zero metadata and early return; leave new_block_tables[i] untouched
        tl.store(p1_kv_id_ptr + i, tl.full((), -1, tl.int32))
        tl.store(p2_kv_id_ptr + i, tl.full((), -1, tl.int32))
        tl.store(p1_size_ptr + i, tl.full((), 0, tl.int32))
        tl.store(p2_size_ptr + i, tl.full((), 0, tl.int32))
        # tl.store(need_move_ptr + i, tl.full((), 0, tl.int8))
        return

    # Compute sizes / block ids
    p2_size = (kv_len_i % BLOCK_SIZE).to(tl.int32)
    p1_size = (BLOCK_SIZE - p2_size).to(tl.int32)  # valid even when p2_size==0
    p2_block_id = (kv_len_i - 1) // BLOCK_SIZE
    p1_block_id = (kv_len_i - WINDOW_SIZE) // BLOCK_SIZE
    need_move = p2_size != 0

    # If we need to move the tail, figure out kv_ids for p1 and p2
    p1_kv_id = tl.full((), -1, tl.int32)
    p2_kv_id = tl.full((), -1, tl.int32)
    if need_move:
        p1_kv_id = tl.load(block_tables_ptr + i * MAX_BLOCKS + p1_block_id)
        p2_kv_id = tl.load(block_tables_ptr + i * MAX_BLOCKS + p2_block_id)

    # Where to start copying the window slice in the block table
    start = (p1_block_id + (p2_size != 0)).to(tl.int32)

    # Copy WIN_BLOCKS entries: new_block_tables[i, :] = block_tables[i, start : start+WIN_BLOCKS]
    offs = tl.arange(0, WIN_BLOCKS)
    src_idx = start + offs
    mask = src_idx < MAX_BLOCKS  # safety in case of ragged table tails
    src_ptrs = block_tables_ptr + i * MAX_BLOCKS + src_idx
    dst_ptrs = new_block_tables_ptr + i * WIN_BLOCKS + offs
    vals = tl.load(src_ptrs, mask=mask, other=tl.full((), 0, tl.int32))
    tl.store(dst_ptrs, vals, mask=mask)

    # Write metadata for kernel B
    tl.store(p1_kv_id_ptr + i, p1_kv_id)
    tl.store(p2_kv_id_ptr + i, p2_kv_id)
    tl.store(p1_size_ptr + i, p1_size.to(tl.int32))
    tl.store(p2_size_ptr + i, p2_size.to(tl.int32))
    # tl.store(need_move_ptr + i, need_move.to(tl.int8))

def _launch_move_kv_tail(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    p1_kv_id: torch.Tensor,   # int64 [B]
    p2_kv_id: torch.Tensor,   # int64 [B]
    p1_size: torch.Tensor,    # int32 [B]
    p2_size: torch.Tensor,    # int32 [B]
    # need_move: torch.Tensor,  # int8  [B]
    H: int, D: int,
    max_block_size: int,
):
    # Shapes / strides
    # key_cache: [num_blocks, block_size, H, D]
    assert key_cache.ndim == 4 and value_cache.ndim == 4
    assert key_cache.shape == value_cache.shape
    # s_k0, s_k1, s_k2, s_k3 = [torch.tensor(int(s), dtype=torch.int32, device=key_cache.device) for s in key_cache.stride()]
    # s_v0, s_v1, s_v2, s_v3 = [torch.tensor(int(s), dtype=torch.int32, device=value_cache.device) for s in value_cache.stride()]
    s_k0, s_k1, s_k2, s_k3 = key_cache.stride()
    s_v0, s_v1, s_v2, s_v3 = value_cache.stride()

    B = p1_kv_id.numel()

    # We generate a tiny wrapper kernel that expands the t-loop up to max_block_size.
    @triton.jit
    def _copy_tail(
        k_ptr, v_ptr,
        s_k0, s_k1, s_k2, s_k3,
        s_v0, s_v1, s_v2, s_v3,
        p1_kv_id_ptr, p2_kv_id_ptr,
        p1_size_ptr, p2_size_ptr, 
        # need_move_ptr,
        H, D, B,
        MAX_T: tl.constexpr,
        BLOCK_E: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_e = tl.program_id(1)
        if pid_b >= B:
            return
        # need = tl.load(need_move_ptr + pid_b)
        # if need == 0:
        #     return
        p2_sz = tl.load(p2_size_ptr + pid_b).to(tl.int32)

        if p2_sz == 0:
            return

        p1_id = tl.load(p1_kv_id_ptr + pid_b).to(tl.int32)
        p2_id = tl.load(p2_kv_id_ptr + pid_b).to(tl.int32)
        p1_sz = tl.load(p1_size_ptr + pid_b).to(tl.int32)
        # p2_sz = tl.load(p2_size_ptr + pid_b).to(tl.int32)

        E = H * D
        e_offsets = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
        e_mask = e_offsets < E
        h_idx = e_offsets // D
        d_idx = e_offsets % D

        # Base pointers for K/V for this (p1_id/p2_id)
        # Address: base + b0*s0 + t*s1 + h*s2 + d*s3
        for t in tl.range(0, MAX_T):
            # in_range = t < p1_sz
            # if not tl.any(in_range & e_mask):
            #     continue
            t_src = (MAX_T - p1_sz + t)
            t_dst = (p2_sz + t)

            # Pointer math for K
            k_src = (k_ptr + p1_id * s_k0 + t_src * s_k1 + h_idx * s_k2 + d_idx * s_k3)
            k_dst = (k_ptr + p2_id * s_k0 + t_dst * s_k1 + h_idx * s_k2 + d_idx * s_k3)

            # Pointer math for V
            v_src = (v_ptr + p1_id * s_v0 + t_src * s_v1 + h_idx * s_v2 + d_idx * s_v3)
            v_dst = (v_ptr + p2_id * s_v0 + t_dst * s_v1 + h_idx * s_v2 + d_idx * s_v3)

            mask_t = e_mask & (t < p1_sz)  # guards both E tile and t
            # Copy
            k_val = tl.load(k_src, mask=mask_t, other=0.0)
            v_val = tl.load(v_src, mask=mask_t, other=0.0)
            tl.store(k_dst, k_val, mask=mask_t)
            tl.store(v_dst, v_val, mask=mask_t)

    # Tune tile across features
    E = H * D
    BLOCK_E = 256 if E >= 256 else 128 if E >= 128 else 64
    grid = (B, (E + BLOCK_E - 1) // BLOCK_E)
    _copy_tail[grid](
        key_cache, value_cache,
        s_k0, s_k1, s_k2, s_k3,
        s_v0, s_v1, s_v2, s_v3,
        p1_kv_id, p2_kv_id,
        p1_size, p2_size, 
        # need_move,
        H, D, B,
        MAX_T=max_block_size,   # equals BLOCK_SIZE
        BLOCK_E=BLOCK_E,
        num_warps=4,
        num_stages=2,
    )

def slide_window_and_update_tables_triton(
    key_cache: torch.Tensor,        # [num_blocks, block_size, H, D]
    value_cache: torch.Tensor,      # [num_blocks, block_size, H, D]
    block_tables: torch.Tensor,     # [B, max_num_blocks_per_req]
    kv_lens: torch.Tensor,          # [B]
    new_block_tables: torch.Tensor, # [B, window_size // block_size]
    window_size: int,
    block_size: int,
):
    assert window_size % block_size == 0
    device = block_tables.device
    assert device.type == "cuda", "Use CUDA/ROCm device"

    B, MAX_BLOCKS = block_tables.shape
    WIN_BLOCKS = window_size // block_size
    _, bs, H, D = key_cache.shape
    assert bs == block_size

    # Aux metadata (no host loops)
    p1_kv_id = torch.empty(B, dtype=torch.int32, device=device)
    p2_kv_id = torch.empty(B, dtype=torch.int32, device=device)
    p1_size  = torch.empty(B, dtype=torch.int32, device=device)
    p2_size  = torch.empty(B, dtype=torch.int32, device=device)
    # need_move = torch.empty(B, dtype=torch.int8, device=device)

    # build new_block_tables and emit metadata
    grid_a = (triton.cdiv(B, 1),)
    build_window_block_tables[grid_a](
        block_tables, kv_lens, new_block_tables,
        p1_kv_id, p2_kv_id, p1_size, p2_size, 
        # need_move,
        B=B,
        MAX_BLOCKS=MAX_BLOCKS,
        WIN_BLOCKS=WIN_BLOCKS,
        WINDOW_SIZE=window_size,
        BLOCK_SIZE=block_size,
        num_warps=2,
        num_stages=2,
    )

    # move KV tails only where needed (p2_size > 0)
    # (No allocation; in-place on key_cache/value_cache)
    _launch_move_kv_tail(
        key_cache, value_cache,
        p1_kv_id, p2_kv_id, p1_size, p2_size, 
        # need_move,        
        H=H, D=D,
        max_block_size=block_size,
    )

@perftest(num_iters=5, num_warmup=1)
def run_pa_v1(
    query,
    key_cache,
    value_cache,
    query_lens,
    cu_seqlens_q,
    cu_seqlens_k,
    kv_lens,
    max_seqlen_q,
    max_seqlen_k,
    block_tables,
    scale,
    block_size: int = 16,
    window_size: int = 128,
):
    if window_size and max_seqlen_k > window_size:
        assert window_size % block_size == 0
        B, N = block_tables.shape
        win_blocks = window_size // block_size
        new_block_tables = torch.empty(
            (B, win_blocks),
            dtype=block_tables.dtype,
            device=block_tables.device,
        )

        slide_window_and_update_tables_triton(
            key_cache, value_cache,
            block_tables, kv_lens,
            new_block_tables,
            window_size, block_size,
        )

        kv_lens = torch.clamp(kv_lens, max=window_size)
        block_tables = new_block_tables
        max_seqlen_k = min(max_seqlen_k, window_size)

    return pa_v1_zero_overhead(query, key_cache, value_cache, cu_seqlens_q, kv_lens, max_seqlen_k, block_tables, scale)

def pa_v1_zero_overhead(query, key_cache, value_cache, cu_seqlens_q, kv_lens, max_seqlen_k, block_tables, scale):
    output = torch.empty_like(query)
    num_actual_tokens = int(cu_seqlens_q[-1].item())
    _PARTITION_SIZE_ROCM = 256
    _, num_heads, head_size = query.shape
    nbytes_per_qo_elem = torch.finfo(query.dtype).bits // 8
    num_seqs = kv_lens.shape[0]
    max_num_partitions = (max_seqlen_k + _PARTITION_SIZE_ROCM -
                            1) // _PARTITION_SIZE_ROCM

    workspace_buffer = torch.empty(
        (num_seqs * num_heads * max_num_partitions * head_size) *
        nbytes_per_qo_elem + 2 *
        (num_seqs * num_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=output.device,
    )

    torch.ops.aiter.paged_attention_v1(
        output[:num_actual_tokens],
        workspace_buffer,
        query[:num_actual_tokens],
        key_cache,
        value_cache,
        scale,
        block_tables,
        cu_seqlens_q,
        kv_lens,
        max_seqlen_k,
        None, # alibi_slopes
        "auto",
        "NHD",
        0,  # logits_soft_cap
        torch.tensor(1.0, dtype=torch.float32, device=workspace_buffer.device), # k_scale
        torch.tensor(1.0, dtype=torch.float32, device=workspace_buffer.device), # v_scale
        None,
        _PARTITION_SIZE_ROCM,
    )

    
    # tmp_output, exp_sums, max_logits = unpack_workspace_buffers(workspace_buffer, num_seqs, num_heads, max_num_partitions, head_size, output.dtype)
    # lse = lse_from_partitions(exp_sums, max_logits)
    return output

####################### Paged Attention v1(End) ###############################

########################################################
#  Benchmarking
########################################################
def bench_one_case(seq_lens: list[tuple[int, int]], common_prefix_len: int = 0):
    # gpt-oss shape
    # seq_lens = [(1, 2048), (1, 2048), (1, 2048), (1, 2048)]    # arbitrary seq len
    print(f"Seq Lens: {[(q, kv) for q, kv in seq_lens]}")
    num_heads = (64, 8)
    head_size = 64
    sliding_window = 128
    dtype = torch.bfloat16
    block_size = 16
    num_blocks = 8192   # arbitrary, might be overide in create_inputs

    (
        query,
        key_cache,
        value_cache,
        query_lens,
        cu_seqlens_q,
        cu_seqlens_k,
        kv_lens,
        max_seqlen_q,
        max_seqlen_k,
        window_size,
        block_tables,
        scale,
        sliding_window,
        sinks,
    ) = create_inputs(
        seq_lens=seq_lens,
        num_heads=num_heads,
        head_size=head_size,
        sliding_window=sliding_window,
        dtype=dtype,
        block_size=block_size,
        num_blocks=num_blocks,
        common_prefix_len=common_prefix_len,
    )

    # Turn off SWA & attn sink
    sinks = None
    # sliding_window, window_size = None, (-1, -1)

    aiter_unified_attn_output, aiter_unified_avg_us = run_aiter_unified(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        kv_lens=kv_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        window_size=window_size,
        block_tables=block_tables,
        scale=scale,
        sliding_window=sliding_window,
        sinks=sinks,
    )

    # vllm_unified_attn_output, vllm_unified_avg_us = run_vllm_unified(
    #     query=query,
    #     key_cache=key_cache,
    #     value_cache=value_cache,
    #     query_lens=query_lens,
    #     cu_seqlens_q=cu_seqlens_q,
    #     cu_seqlens_k=cu_seqlens_k,
    #     kv_lens=kv_lens,
    #     max_seqlen_q=max_seqlen_q,
    #     max_seqlen_k=max_seqlen_k,
    #     window_size=window_size,
    #     block_tables=block_tables,
    #     scale=scale,
    #     sliding_window=sliding_window,
    #     sinks=sinks,
    # )

    pa_output, pa_avg_us = run_pa_v1(
    # pa_output, pa_lse = run_pa_v1(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        kv_lens=kv_lens,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        block_tables=block_tables,
        scale=scale,
        window_size=sliding_window,
    )

    torch.testing.assert_close(
        aiter_unified_attn_output, pa_output, atol=2e-2, rtol=2e-2
    ), f"{torch.max(torch.abs(aiter_unified_attn_output - pa_output))}"

    return aiter_unified_avg_us, pa_avg_us

def main():

    batch_size = [32, 128, 256, 128]
    context_len = [1024, 2048, 4096, 16384]
    bench_prefill = False

    for b in batch_size[-1:]:
        for c in context_len[-1:]:
            print(f"==== Batch Size {b}, Context Len {c} ====")
            scheduler_decode_outputs, scheduler_prefill_outputs = get_scheduler_output(context_len=c, batch_size=b)

            if bench_prefill:
                scheduler_outputs = scheduler_prefill_outputs
            else:
                scheduler_outputs = scheduler_decode_outputs

            baseline_results = []
            pa_results = []
            for i, seq_lens in enumerate(scheduler_outputs[:20]):
                baseline_result, pa_result = bench_one_case(seq_lens=seq_lens, common_prefix_len=0)
                baseline_results.append(baseline_result)
                pa_results.append(pa_result)

            
            print("==== Summary ====")
            baseline_avg_latency = sum(baseline_results) / len(baseline_results)
            pa_avg_latency = sum(pa_results) / len(pa_results)
            speedup = baseline_avg_latency / pa_avg_latency
            print(f"Avg Latency Aiter Unified Attention: {baseline_avg_latency:.2f} ms")
            print(f"Avg Latency PA: {pa_avg_latency:.2f} ms")
            print(f"Speedup: {speedup:.2f}x")
            

if __name__ == "__main__":
    torch.set_default_device("cuda")
    torch.manual_seed(42)
    main()
