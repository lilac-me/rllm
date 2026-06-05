import torch
import torch.nn as nn
import json
import os

class Model(nn.Module):
    """
    Model that performs AdamW optimization step using NPU accelerated Adam.
    Pytorch native implemention
    def forward(self, var: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                grad: torch.Tensor, beta1_power: float, beta2_power: float,
                lr: float, beta1: float, beta2: float, epsilon: float,
                use_locking: bool = False, use_nesterov: bool = False):
        orig_dtype = var.dtype

        var_f32 = var.float()
        m_f32 = m.float()
        v_f32 = v.float()
        grad_f32 = grad.float()

        m_new = beta1 * m_f32 + (1 - beta1) * grad_f32
        v_new = beta2 * v_f32 + (1 - beta2) * grad_f32 * grad_f32

        m_hat = m_new / (1 - beta1_power)
        v_hat = v_new / (1 - beta2_power)

        if use_nesterov:
            m_nesterov = beta1 * m_hat + (1 - beta1) * grad_f32 / (1 - beta1_power)
            var_new = var_f32 - lr * m_nesterov / (torch.sqrt(v_hat) + epsilon)
        else:
            var_new = var_f32 - lr * m_hat / (torch.sqrt(v_hat) + epsilon)

        var_out = var_new.to(orig_dtype)
        m_out = m_new.to(m.dtype)
        v_out = v_new.to(v.dtype)

        return var_out, m_out, v_out
    """
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, var: torch.Tensor, m: torch.Tensor, v: torch.Tensor,
                grad: torch.Tensor, beta1_power: float, beta2_power: float,
                lr: float, beta1: float, beta2: float, epsilon: float,
                use_locking: bool = False, use_nesterov: bool = False):
        """
        Applies Adam optimization step on NPU.

        Args:
            var (torch.Tensor): Variable to be updated.
            m (torch.Tensor): First moment estimates.
            v (torch.Tensor): Second moment estimates.
            grad (torch.Tensor): Gradient tensor.
            beta1_power (float): Beta1 power.
            beta2_power (float): Beta2 power.
            lr (float): Learning rate.
            beta1 (float): Exponential decay rate for first moment.
            beta2 (float): Exponential decay rate for second moment.
            epsilon (float): Small constant for numerical stability.
            use_locking (bool): Whether to use locking.
            use_nesterov (bool): Whether to use Nesterov momentum.

        Returns:
            tuple: Updated (var, m, v).
        """
        import torch_npu
        torch_npu.npu_apply_adam(beta1_power, beta2_power, lr, beta1, beta2, epsilon,
                                  grad, use_locking, use_nesterov, out=(var, m, v))
        return var, m, v


def get_init_inputs():
    return []


def get_input_groups():
    return [
        [torch.randn([4096, 4096], dtype=torch.float32), torch.randn([4096, 4096], dtype=torch.float32), torch.randn([4096, 4096], dtype=torch.float32), torch.randn([4096, 4096], dtype=torch.float32), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([8192, 16384], dtype=torch.float32), torch.randn([8192, 16384], dtype=torch.float32), torch.randn([8192, 16384], dtype=torch.float32), torch.randn([8192, 16384], dtype=torch.float32), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([2048, 13824], dtype=torch.float32), torch.randn([2048, 13824], dtype=torch.float32), torch.randn([2048, 13824], dtype=torch.float32), torch.randn([2048, 13824], dtype=torch.float32), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([789, 12288], dtype=torch.float32), torch.randn([789, 12288], dtype=torch.float32), torch.randn([789, 12288], dtype=torch.float32), torch.randn([789, 12288], dtype=torch.float32), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([4096, 18432], dtype=torch.float32), torch.randn([4096, 18432], dtype=torch.float32), torch.randn([4096, 18432], dtype=torch.float32), torch.randn([4096, 18432], dtype=torch.float32), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([100, 2007], dtype=torch.float32), torch.randn([100, 2007], dtype=torch.float32), torch.randn([100, 2007], dtype=torch.float32), torch.randn([100, 2007], dtype=torch.float32), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([4096, 4096], dtype=torch.bfloat16), torch.randn([4096, 4096], dtype=torch.float32), torch.randn([4096, 4096], dtype=torch.float32), torch.randn([4096, 4096], dtype=torch.bfloat16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([8192, 16384], dtype=torch.bfloat16), torch.randn([8192, 16384], dtype=torch.float32), torch.randn([8192, 16384], dtype=torch.float32), torch.randn([8192, 16384], dtype=torch.bfloat16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([2048, 13824], dtype=torch.bfloat16), torch.randn([2048, 13824], dtype=torch.float32), torch.randn([2048, 13824], dtype=torch.float32), torch.randn([2048, 13824], dtype=torch.bfloat16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([789, 12288], dtype=torch.bfloat16), torch.randn([789, 12288], dtype=torch.float32), torch.randn([789, 12288], dtype=torch.float32), torch.randn([789, 12288], dtype=torch.bfloat16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([4096, 18432], dtype=torch.bfloat16), torch.randn([4096, 18432], dtype=torch.float32), torch.randn([4096, 18432], dtype=torch.float32), torch.randn([4096, 18432], dtype=torch.bfloat16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([100, 2007], dtype=torch.bfloat16), torch.randn([100, 2007], dtype=torch.float32), torch.randn([100, 2007], dtype=torch.float32), torch.randn([100, 2007], dtype=torch.bfloat16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([4096, 4096], dtype=torch.float16), torch.randn([4096, 4096], dtype=torch.float32), torch.randn([4096, 4096], dtype=torch.float32), torch.randn([4096, 4096], dtype=torch.float16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([8192, 16384], dtype=torch.float16), torch.randn([8192, 16384], dtype=torch.float32), torch.randn([8192, 16384], dtype=torch.float32), torch.randn([8192, 16384], dtype=torch.float16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([2048, 13824], dtype=torch.float16), torch.randn([2048, 13824], dtype=torch.float32), torch.randn([2048, 13824], dtype=torch.float32), torch.randn([2048, 13824], dtype=torch.float16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([789, 12288], dtype=torch.float16), torch.randn([789, 12288], dtype=torch.float32), torch.randn([789, 12288], dtype=torch.float32), torch.randn([789, 12288], dtype=torch.float16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([4096, 18432], dtype=torch.float16), torch.randn([4096, 18432], dtype=torch.float32), torch.randn([4096, 18432], dtype=torch.float32), torch.randn([4096, 18432], dtype=torch.float16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
        [torch.randn([100, 2007], dtype=torch.float16), torch.randn([100, 2007], dtype=torch.float32), torch.randn([100, 2007], dtype=torch.float32), torch.randn([100, 2007], dtype=torch.float16), 0.9, 0.999, 0.001, 0.9, 0.999, 1e-08, False, False],
    ]
