import torch
import triton
import triton.language as tl

import torch.nn.functional as F

def naive_softmax(x):
    """
    A naive implementation of softmax using PyTorch.
    """
    x_max = torch.max(x, dim=-1, keepdim=True).values
    safe_x = x - x_max
    numerator = torch.exp(safe_x)
    denominator = torch.sum(numerator, dim=-1, keepdim=True)
    sm_out = numerator / denominator
    return sm_out


def online_softmax(x):
    """
    An online implementation of softmax, 2.5x faster.
    """
    assert x.dim() == 2, "Input must be a 2D tensor"

    row_count, col_count = x.shape
    output = torch.empty_like(x)

    for r in range(row_count):
        row_max = float('-inf')
        normalizer = 0.0
        for c in range(col_count):
            curr = x[r, c]
            prev_max = row_max
            row_max = max(row_max, curr)
            # if row_max != prev_max:
            #     print("Adjusting normalizer")
            normalizer = normalizer * torch.exp(prev_max - row_max) + torch.exp(curr - row_max)

        output[r, :] = torch.exp(x[r, :] - row_max) / normalizer
    
    return output


@triton.jit
def _softmax_kernel(
    x,
    x_stride_row,
    out, 
    out_stride_row,
    col_count,
    BLOCK_SIZE: tl.constexpr
):
    # Implement the softmax kernel using Triton
    row_idx = tl.program_id(0)
    input_ptr = x + row_idx * x_stride_row + tl.arange(0, BLOCK_SIZE)
    output_ptr = out + row_idx * out_stride_row + tl.arange(0, BLOCK_SIZE)

    input_x = tl.load(input_ptr, mask=tl.arange(0, BLOCK_SIZE) < col_count, other=float("-inf"))

    safe_x = input_x - tl.max(input_x, axis=0)
    numerator = tl.exp(safe_x)
    denominator = tl.sum(numerator, axis=0)
    output = numerator / denominator
    # tl.static_print("BLOCK size:", BLOCK_SIZE)
    tl.store(output_ptr, output, mask=tl.arange(0, BLOCK_SIZE) < col_count)

def triton_softmax(x):
    assert x.dim() == 2, "Input must be a 2D tensor"
    row, col = x.shape
    out = torch.empty_like(x)

    block_size = triton.next_power_of_2(col)
    grid = (row,)

    num_warps = 4 # *64, warp size of AMD GPUs

    if block_size > 2047:
        num_warps = 8
    elif block_size > 4095:
        num_warps = 16

    _softmax_kernel[grid](
        x,
        x.stride(0),
        out,
        out.stride(0),
        col, 
        BLOCK_SIZE=block_size, 
        num_warps=num_warps
    )

    return out

if __name__ == "__main__":
    sample = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], device='cuda')
    ref_out = F.softmax(sample, dim=-1)
    naive_out = naive_softmax(sample)
    online_out = online_softmax(sample)
    triton_out = triton_softmax(sample)

    print("Reference Output:\n", ref_out)
    print("Naive Softmax Output:\n", naive_out)
    print("Online Softmax Output:\n", online_out)
    print("Triton Softmax Output:\n", triton_out)
