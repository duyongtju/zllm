
import torch
import torch.nn as nn

from zllm import layernorm_ops

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, layer_id = None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.forward_cuda(x)

    def forward_native(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    def forward_cuda(self, x):
        output = torch.randn_like(x, dtype=x.dtype, device=x.device)
        layernorm_ops.rms_norm(output, x, self.weight, self.eps)
        return output

