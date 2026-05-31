import torch
from torch.autograd import Function
import torch.nn as nn
import math
from loguru import logger
import torch.nn.functional as F

class SparseLinearFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, dummy, weight_grad_hook):
        y = F.linear(x, weight, bias)

        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        ctx.weight_grad_hook = weight_grad_hook

        return y

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors

        grad_x = grad_out.matmul(weight)
        grad_weight = grad_out.reshape(-1, grad_out.size(-1)).t().matmul(x.reshape(-1, x.size(-1)))

        if ctx.weight_grad_hook is not None:
            ctx.weight_grad_hook(grad_weight)

        return grad_x, grad_weight, None, None, None


class SparseLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        self.in_features = in_features
        self.out_features = out_features

        #
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs), requires_grad=False)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        self._dummy = nn.Parameter(torch.zeros(1, **factory_kwargs), requires_grad=True)

        self.reset_parameters()
        self.hook = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def apply_hook(self, hook):
        self.hook = hook

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = SparseLinearFn.apply(input, self.weight, self.bias, self._dummy, self.hook)

        return y

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

