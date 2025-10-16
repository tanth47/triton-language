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

        for i, kv_len in enumerate(kv_lens):
            if kv_len < window_size:
                continue
            
            p2_size = kv_len % block_size
            p1_size = block_size - p2_size
            p2_block_id = (kv_len - 1) // block_size 
            p1_block_id = (kv_len - window_size) // block_size

            if p2_size != 0:
                p2_kv_id = block_tables[i, p2_block_id].item()
                p1_kv_id = block_tables[i, p1_block_id].item()

                # manipulate k,v to move p1_block_id to the end
                key_cache[p2_kv_id, p2_size:].copy_(key_cache[p1_kv_id, (block_size - p1_size):])
                value_cache[p2_kv_id, p2_size:].copy_(value_cache[p1_kv_id, (block_size - p1_size):])

            # block_tables[i, start:start+win_blocks] with a LongTensor index 
            # creates a new tensor via advanced indexing, not a view
            # so it allocates a temporary before the copy_, 
            # defeating “no-temp / cudagraph-friendly” goal.
            src = block_tables[i].narrow(0, p1_block_id + (p2_size > 0), win_blocks)
            new_block_tables[i].copy_(src)

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
