from typing import List, Optional, Union, Tuple

import torch
from torch import Tensor
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
from torch.optim.optimizer import Optimizer
from torch._utils import is_compiling
import math
from torch.utils._foreach_utils import (
    Indices,
    TensorListList,
    _get_foreach_kernels_supported_devices,
    _get_fused_kernels_supported_devices,
)

__all__ = ['Adam', 'adam']
_foreach_supported_types = [torch.Tensor, torch.nn.parameter.Parameter]


def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and is_compiling():
        return x
    else:
        return x

def _dispatch_sqrt(x: float):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return torch.sqrt(x)

class Adam(Optimizer):
    def __init__(self,
                 named_params,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 amsgrad: bool = False,
                 *,
                 foreach: Optional[bool] = None,
                 maximize: bool = False,
                 capturable: bool = False,
                 differentiable: bool = False,
                 fused: Optional[bool] = None,
                 decay_steps: float = 20,
                 decay_max: float = 1.0,
                 masks=None, masks_pre=None
                 ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if isinstance(lr, Tensor) and foreach and not capturable:
            raise ValueError("lr as a Tensor is not supported for capturable=False and foreach=True")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable,
                        differentiable=differentiable, fused=fused)
        # super().__init__(params, defaults)

        # separate params and build name map
        params = []
        self.mask_name_map = {}
        for name, p in named_params:
            params.append(p)
            if masks is not None and name in masks:
                self.mask_name_map[p] = name

        super().__init__([{'params': params}], defaults)

        self.masks = masks
        self.masks_pre = masks_pre

        self.decay_steps = decay_steps
        self.decay_max = decay_max


    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
            group.setdefault('differentiable', False)
            group.setdefault('fused', None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps
    ):
        for p in group['params']:
            if p.grad is not None:
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if both capturable and fused are off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    # state['step'] = (
                    #     torch.zeros((), dtype=torch.float, device=p.device)
                    #     if group['capturable'] or group['fused']
                    #     else torch.tensor(0.)
                    # )
                    state['step'] = torch.zeros_like(p, dtype=torch.float, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                state_steps.append(state['step'])

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            beta1, beta2 = group['betas']

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps)

            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group['amsgrad'],
                beta1=beta1,
                beta2=beta2,
                lr=group['lr'],
                weight_decay=group['weight_decay'],
                eps=group['eps'],
                maximize=group['maximize'],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                mask_name_map=self.mask_name_map,
                masks_pre=self.masks_pre,
                masks=self.masks,
                decay_steps=self.decay_steps,
                decay_max=self.decay_max
            )

        return loss


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         grad_scale: Optional[Tensor] = None,
         found_inf: Optional[Tensor] = None,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: Union[float, Tensor],
         weight_decay: float,
         eps: float,
         maximize: bool,
         mask_name_map,
         masks_pre,
         masks,
         decay_steps,
         decay_max
         ):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.


    func = _single_tensor_adam

    func(params,
         grads,
         exp_avgs,
         exp_avg_sqs,
         max_exp_avg_sqs,
         state_steps,
         amsgrad=amsgrad,
         beta1=beta1,
         beta2=beta2,
         lr=lr,
         weight_decay=weight_decay,
         eps=eps,
         maximize=maximize,
         grad_scale=grad_scale,
         found_inf=found_inf,
         mask_name_map=mask_name_map,
         masks_pre=masks_pre,
         masks=masks,
         decay_steps=decay_steps,
         decay_max=decay_max
         )


def _single_tensor_adam(params: List[Tensor],
                        grads: List[Tensor],
                        exp_avgs: List[Tensor],
                        exp_avg_sqs: List[Tensor],
                        max_exp_avg_sqs: List[Tensor],
                        state_steps: List[Tensor],
                        grad_scale: Optional[Tensor],
                        found_inf: Optional[Tensor],
                        *,
                        amsgrad: bool,
                        beta1: float,
                        beta2: float,
                        lr: Union[float, Tensor],
                        weight_decay: float,
                        eps: float,
                        maximize: bool,
                        mask_name_map,
                        masks_pre,
                        masks,
                        decay_steps,
                        decay_max):


    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # update step
        step_t += 1

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        decay = torch.clamp(step_t / decay_steps, max=decay_max)

        exp_avg.mul_(decay)

        step = _get_value(step_t)

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        step_size = lr / bias_correction1

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

        # param.addcdiv_(exp_avg, denom, value=-step_size)
        param -= step_size * exp_avg / denom



