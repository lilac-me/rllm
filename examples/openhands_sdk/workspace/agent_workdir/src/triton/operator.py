import triton
import triton.language as tl
import torch


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Element-wise addition kernel."""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load elements from x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Compute addition
    output = x + y
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)


def forward(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Element-wise addition using Triton kernel.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Element-wise sum of x and y
    """
    # Ensure inputs are contiguous and on the same device
    x = x.contiguous()
    y = y.contiguous()
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.device == y.device, "Input tensors must be on the same device"
    
    # Allocate output tensor
    output = torch.empty_like(x)
    
    # Get total number of elements
    n_elements = x.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    add_kernel[grid](
        x,
        y,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def forward_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation using PyTorch for correctness checking.
    
    Args:
        x: First input tensor
        y: Second input tensor
        
    Returns:
        Element-wise sum of x and y
    """
    return x + y
