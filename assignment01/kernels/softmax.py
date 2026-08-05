"""问题 7.8（选做）：softmax in Triton（FROM-SCRATCH）。

注：此题可以不用GPU (conftest.py 会自动切到 interpreter 模式)。

contract：
- softmax(x) 接收形状 (M, N) 的 2D tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 自己写，一个 program 处理一行；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意（用 mask 处理），可以假设 N <= 4096，BLOCK_SIZE 用
  triton.next_power_of_2(N) 是常见做法；
- 通过 pytest tests/test_softmax.py 即为完成。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(x_ptr, y_ptr, n_cols: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    columns = tl.arange(0, BLOCK_SIZE)
    mask = columns < n_cols
    values = tl.load(x_ptr + row * n_cols + columns, mask=mask, other=-float("inf"))
    values -= tl.max(values, axis=0)
    numerators = tl.exp(values)
    denominator = tl.sum(numerators, axis=0)
    tl.store(y_ptr + row * n_cols + columns, numerators / denominator, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("softmax expects a 2D tensor")
    rows, columns = x.shape
    if columns == 0 or columns > 4096:
        raise ValueError("row width must be in [1, 4096]")
    output = torch.empty_like(x)
    block_size = triton.next_power_of_2(columns)
    softmax_kernel[(rows,)](
        x, output, n_cols=columns, BLOCK_SIZE=block_size,
        num_warps=8 if block_size >= 2048 else 4,
    )
    return output
