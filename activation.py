import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ReGLU(nn.Module):
    def reglu(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.relu(b)

    def forward(self, x: Tensor) -> Tensor:
        return self.reglu(x)


class GEGLU(nn.Module):
    def geglu(self, x: Tensor) -> Tensor:
        assert x.shape[-1] % 2 == 0
        a, b = x.chunk(2, dim=-1)
        return a * F.gelu(b)

    def forward(self, x: Tensor) -> Tensor:
        return self.geglu(x)
