import triton
import triton.language as tl
from typing import Optional
import torch


def weighted_sum_fwd(
    x_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    x_row_stride: tl.uint32,
    output_ptr: tl.pointer_type,
    H: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
):
    # Each instance will compute the weighted sum of a row of x.
    row_idx = tl.program_id(0)
    # Pointer to the first entry of the row this instance sums up.
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    # Pointers to the entries we'll sum up.
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    # Load the data from x given the pointers to its entries,
    # using a mask since BLOCK_SIZE may be > H.
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    output = tl.sum(row * weight)
    # Write back output (a single scalar per instance).
    output_ptr = output_ptr + row_idx
    tl.store(output_ptr, output)


@triton.jit
def weighted_sum_backward(
    grad_output_ptr: tl.pointer_type,
    grad_x_ptr: tl.pointer_type,
    partial_grad_weight_ptr: tl.pointer_type,
    x_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    x_row_stride: tl.uint32,
    H: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    grad_output_ptrs = weight_ptr + offsets
    mask = offsets < H

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0)
    # Gradient with respect to the output of our operation at row_idx.
    grad_output = tl.load(grad_output_ptr + row_idx)  # (scalar)
    # Compute gradient with respect to the current row of x.
    grad_x_row = grad_output * weight  # (See Eq 4)
    # Move grad_x_ptr to the right output row and write the gradient.
    grad_x_ptr = grad_x_ptr + row_idx * x_row_stride
    tl.store(grad_x_ptr + offsets, grad_x_row, mask=mask)

    # Now compute partial gradient with respect to the weight vector.
    # We will write one row to partial_grad_weight_ptr, and later
    # accumulate these rows to compute the gradient w.r.t. the weight vector.
    partial_grad_weight_ptr = partial_grad_weight_ptr + row_idx * x_row_stride + offsets
    row = tl.load(row_start_ptr + offsets, mask=mask, other=0)
    grad_weight_row = row * grad_output  # (See Eq 3)
    tl.store(partial_grad_weight_ptr, grad_weight_row, mask=mask)


class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Remember x and weight for the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        ctx.save_for_backward(x, weight)
        H, output_dims = x.shape[-1], x.shape[:-1]
        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty(output_dims, device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(n_rows,)](
            x, weight, x.stride(0), y, H, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE
        )
        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        N, H = x.shape
        # Allocate output tensors.
        partial_grad_weight = torch.empty_like(x)
        grad_x = torch.empty_like(x)
        weighted_sum_backward[(N,)](
            grad_out,
            grad_x,
            partial_grad_weight,
            x,
            weight,
            x.stride(0),
            H,
            num_warps=16,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        return grad_x, partial_grad_weight.sum(axis=0)


@triton.jit
def rms_norm_fwd(
    x_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    x_row_stride: tl.uint32,
    output_ptr: tl.pointer_type,
    output_row_stride: tl.uint32,
    rms_ptr: tl.pointer_type,
    H: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
):
    # Each instance will compute the weighted sum of a row of x.
    row_idx = tl.program_id(0)
    # Pointer to the first entry of the row this instance sums up.
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    # Pointers to the entries we'll sum up.
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    # Load the data from x given the pointers to its entries,
    # using a mask since BLOCK_SIZE may be > H.
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    rms = tl.sqrt(tl.sum(row * row)/H + 1e-6)
    output = row / rms * weight
    # Write back output (a single scalar per instance).
    output_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_start_ptr+offsets, output, mask=mask)
    tl.store(rms_ptr+row_idx, rms)




@triton.jit
def rms_norm_bwd(
    grad_output_ptr: tl.pointer_type,
    grad_x_ptr: tl.pointer_type,
    grad_g_ptr: tl.pointer_type,
    x_ptr: tl.pointer_type,
    g_ptr: tl.pointer_type,
    rms_ptr: tl.pointer_type,
    x_row_stride: tl.uint32,
    H: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < H
    rms = tl.load(rms_ptr + row_idx)
    row = tl.load(row_start_ptr + offsets, mask=mask, other=0)

    # Gradient with respect to the output of our operation at row_idx.
    grad_output = tl.load(grad_output_ptr+row_idx * H +offsets, mask=mask, other=0)

    # Compute gradient with respect to the current row of x.
    g = tl.load(g_ptr + offsets, mask=mask, other=0)
    grad_rms = -tl.sum(row * grad_output * g / (rms*rms))
    grad_x_row = grad_output * g / rms + grad_rms * row / (rms * H)
    # Move grad_x_ptr to the right output row and write the gradient.
    grad_x_ptrs = grad_x_ptr + row_idx * x_row_stride + offsets
    tl.store(grad_x_ptrs, grad_x_row, mask=mask)

    # Now compute partial gradient with respect to the g vector.
    # We will write one row to partial_grad_g_ptr, and later
    # accumulate these rows to compute the gradient w.r.t. the g vector.
    grad_g_ptrs = grad_g_ptr + row_idx * H + offsets
    grad_g_row = row / rms * grad_output
    tl.store(grad_g_ptrs, grad_g_row, mask=mask)



class RMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g):
        original_shape = x.shape
        x = x.view(-1, x.shape[-1])
        g = g.view(x.shape[-1])

        H = x.shape[-1]
        assert len(g.shape) == 1 and g.shape[0] == H, "Dimension mismatch: %s"%(str(g.shape))
        assert x.is_cuda and g.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"
        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty(x.shape, device=x.device)
        rms = torch.empty(x.shape[0], device=x.device)

        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.shape[0]
        rms_norm_fwd[(n_rows,)](
            x, g, x.stride(0), y, y.stride(0), rms, H, num_warps=16, BLOCK_SIZE=ctx.BLOCK_SIZE
        )
        ctx.save_for_backward(x, g, rms)
        return y.view(original_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x, g, rms = ctx.saved_tensors
        N, H = x.shape
        # Allocate output tensors.
        grad_g = torch.empty_like(x)
        grad_x = torch.empty_like(x)
        rms_norm_bwd[(N,)](
            grad_out,
            grad_x,
            grad_g,
            x,
            g,
            rms,
            x.stride(0),
            H,
            num_warps=16,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )
        return grad_x.view(*grad_out.shape), torch.sum(grad_g, axis=0)


class RMSNorm(torch.nn.Module):
    def __init__(self, H):
        super().__init__()
        self.g = torch.nn.Parameter(torch.ones(H))

    def forward(self, x):
        return RMSNormFunc.apply(x, self.g)


def rmsnorm_backward_g_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor, rms:Optional[torch.Tensor]=None
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to g.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to g. Shape: (H,)
    """
    x = x.view(-1, x.shape[-1])
    grad_output = grad_output.view(-1, x.shape[-1])
    if rms is None:
        rms = torch.sqrt(torch.sum(x ** 2, dim=-1, keepdim=True) / x.shape[-1] + 1e-6)
    return torch.sum(x / rms * grad_output, axis=0)


def rmsnorm_backward_x_pytorch(
    grad_output: torch.Tensor, x: torch.Tensor, g: torch.Tensor, rms:Optional[torch.Tensor]=None
) -> torch.Tensor:
    """
    Compute the gradient of the RMSNorm operation pass with respect to x.

    Args:
        grad_output: torch.Tensor
            Gradient of the loss with respect to the output of the RMSNorm operation.
            This has the same shape as x.
        x: torch.Tensor
            Input to the RMSNorm operation. Shape: (*, H)
        g: torch.Tensor
            The g learnable parameter of the RMSNorm layer. Shape: (H,)

    Returns:
        Gradient of the loss with respect to x. Shape: (*, H)
    """
    original_shape = x.shape
    x = x.view(-1, x.shape[-1])
    grad_output = grad_output.view(-1, x.shape[-1])
    g = g.view(1, -1)
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
    grad_rms = torch.sum(grad_output * x*g *(-1) / (rms**2), axis=-1, keepdim=True)
    a = grad_output*g/rms
    b = grad_rms *x/(rms*x.shape[-1])
    c = a+b
    return c.view(original_shape)

class RMSNormFuncSlow(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, g):
        original_shape = x.shape
        x = x.view(-1, x.shape[-1])
        g = g.view(x.shape[-1])

        H = x.shape[-1]
        assert len(g.shape) == 1 and g.shape[0] == H, "Dimension mismatch: %s"%(str(g.shape))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-5)
        y = x / rms * g
        ctx.save_for_backward(x, g, rms)
        return y.view(original_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x, g, rms = ctx.saved_tensors
        N, H = x.shape
        grad_g = rmsnorm_backward_g_pytorch(grad_out, x, g, rms)
        grad_x = rmsnorm_backward_x_pytorch(grad_out, x, g, rms)
        return grad_x.view(*grad_out.shape), grad_g
