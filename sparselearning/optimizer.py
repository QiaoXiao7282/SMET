import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import List, Callable, Iterable, Optional, Tuple, Union

import math

class AdamDST(Optimizer):
    r"""Implements Adam for DST (Dynamic Sparse Training).
    The main difference with regular Adam is:
    bias correction is now determined for each individual connection (parameter),
    based on the lifetime of the parameter.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, named_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, masks=None, masks_pre=None, *, foreach: Optional[bool] = None,
                 maximize: bool = False, capturable: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        # separate params and build name map
        params = []
        self.mask_name_map = {}
        for name, p in named_params:
            params.append(p)
            if masks is not None and name in masks:
                self.mask_name_map[p] = name

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        maximize=maximize, foreach=foreach, capturable=capturable)
        super().__init__([{'params': params}], defaults)
        self.masks = masks
        self.masks_pre = masks_pre

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('capturable', False)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(state_values[0]['step'])
        if not step_is_tensor:
            for s in state_values:
                s['step'] = torch.tensor(float(s['step']))

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

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

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        # state['step'] = torch.zeros((1,), dtype=torch.float, device=p.device)
                        state["step"] = torch.ones_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # if p in self.mask_name_map:
                    #     name = self.mask_name_map[p]
                    #     mask = self.masks[name]
                    #     assert mask.size() == p.size()
                    #     # state['step'][mask == 0] = 0
                    #     # state['step'][mask == 1] += 1
                    #     # print("%%%%% {name}")
                    #     state['step'].add_(1)
                    # else:
                    #     # for non-masked (dense, embedding, head) layer
                    #     state['step'].add_(1)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    state_steps.append(state['step'])

            adam(params_with_grad,
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
                 foreach=group['foreach'],
                 capturable=group['capturable'],
                 masks=self.masks,
                 masks_pre=self.masks_pre,
                 mask_name_map=self.mask_name_map)

        return loss


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[Tensor],
         # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
         # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
         foreach: bool = None,
         capturable: bool = False,
         *,
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         maximize: bool,
         masks: Optional[dict],
         masks_pre: Optional[dict],
         mask_name_map: dict):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """

    if not all([isinstance(t, torch.Tensor) for t in state_steps]):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    if foreach is None:
        # Placeholder for more complex foreach logic to be added when value is not set
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach optimizers')

    for i, param in enumerate(params):

        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        assert param.is_cuda and step_t.is_cuda, "If capturable=True, params and state_steps must be CUDA tensors."

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        if step_t.numel() == 1:
            step = step_t.item()
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            step_size = lr / bias_correction1
            bias_correction2_sqrt = math.sqrt(bias_correction2)

            denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            param.addcdiv_(exp_avg, denom, value=-step_size)

        else:
            # decay_mask = torch.ones_like(exp_avg)
            # if masks_pre is not None and param in mask_name_map:
            #     name = mask_name_map[param]
            #     if name in masks and name in masks_pre:
            #         mask_now = masks[name]
            #         mask_old = masks_pre[name]
            #         new_conn = (mask_now == 1) & (mask_old == 0)
            #         decay = torch.clamp(step_t / 20, max=1.0)
            #         decay_mask[new_conn] = decay[new_conn]

            # exp_avg.mul_(decay_mask)

            step = step_t
            step = step.to(dtype=torch.float32)

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            denom = (exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)

            if torch.isnan(grad).any():
                print("### grad has NaN")

            if torch.isnan(exp_avg_sq).any():
                print("### exp_avg_sq has NaN")

            if torch.isnan(bias_correction1).any() or torch.isnan(bias_correction2).any():
                print("### bias correction NaN")

        step_t += 1


class SftAdamW(torch.optim.Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Adapted from Huggingface AdamW optimizer.
    """

    def __init__(
        self,
        named_params: Iterable[torch.nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        decay_steps: float = 20,
        decay_max: float = 1.0,
        correct_bias: bool = True,
        masks=None, masks_pre=None
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}

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

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = torch.ones_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # if p in self.mask_name_map:
                #     name = self.mask_name_map[p]
                #     mask = self.masks[name]
                #     assert mask.size() == p.size()
                #     state['step'][mask == 0] = 1
                #     state['step'][mask == 1] += 1
                #     # print("%%%%% {name}")
                # else:
                #     # for non-masked (dense, embedding, head) layer
                #     state['step'].add_(1)

                age, exp_avg, exp_avg_sq = state["step"], state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                # grad = grad.to(dtype=self.momentum_dtype)
                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                # 1.0 - beta might become 0 in low-precision dtypes like bfloat16
                age = age.to(dtype=torch.float32)

                decay_mask = torch.ones_like(exp_avg)
                # if self.masks_pre and p in self.mask_name_map:
                #     name = self.mask_name_map[p]
                #     if name in self.masks and name in self.masks_pre:
                #         mask_now = self.masks[name]
                #         mask_old = self.masks_pre[name]
                #         new_conn = (mask_now == 1) & (mask_old == 0)
                #         decay = torch.clamp(age / self.decay_steps, max=self.decay_max)
                #         decay_mask[new_conn] = decay[new_conn].to(decay_mask.dtype)

                exp_avg.mul_(decay_mask)

                # Per-parameter bias correction
                bias1_correction = 1.0 - beta1 ** age
                bias2_correction = 1.0 - beta2 ** age
                denom.mul_(bias1_correction)
                denom.div_(torch.sqrt(bias2_correction))

                # exp_avg.mul_(torch.clamp(age / 20, max=1))

                step_size = group["lr"]
                p.addcdiv_(exp_avg, denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

                state['step'].add_(1)

        return loss