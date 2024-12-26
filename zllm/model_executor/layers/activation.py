"""
references: vllm/model_executor/layers/layernorm.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Silu(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_native(x)
