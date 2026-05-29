import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing import List, Optional, Union, Tuple
import math


class Adam_block(Optimizer):
    def __init__(self,
                 param_groups,
                 lr: Union[float, Tensor] = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8,
                 weight_decay: float = 0,
                 amsgrad: bool = False,
                 *,
                 block_size: int = 128,
                 block_ratio: float = 0.1,
                 decay_steps: float = 1,
                 decay_max: float = 1.0,
                 use_norm_adjust: bool = False,
                 use_density_scale: bool = False,
                 lr_scale=None,
                 density_dict=None,
                 index_map: Optional[dict] = None):

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(param_groups, defaults)

        self.block_size = block_size
        self.block_ratio = block_ratio

        self.decay_steps = decay_steps
        self.decay_max = decay_max
        self.use_norm_adjust = use_norm_adjust
        self.use_density_scale = use_density_scale

        self.density_dict = density_dict or {}
        self.index_map = index_map or {}

        self.lr_scale = lr_scale

    # ========================
    # update index
    # ========================
    def update_index_map(self, weight, new_index_dst, replace_idx):

        state = self.state.get(weight, None)
        if state is not None:
            state["idx"] = new_index_dst
            state['step'][replace_idx] = 1

            ## zero out: exp_avg; exp_avg_sq; step
            idx = replace_idx[:, None] * self.block_size + torch.arange(self.block_size, device=weight.device)
            state['exp_avg'][idx] = 0
            state['exp_avg_sq'][idx] = 0


    # ========================
    # 初始化 group
    # ========================
    def _init_group(self, group):
        params_with_grad, grads = [], []
        exp_avgs, exp_avg_sqs, state_steps, indices_list = [], [], [], []

        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Adam does not support sparse gradients")

            params_with_grad.append(p)
            grads.append(p.grad)
            state = self.state[p]
            indices_block = self.index_map.get(id(p), None)

            if len(state) == 0:

                # state['step'] = torch.tensor(0.0, device=p.device)
                if indices_block is None:
                    numel = p.numel()
                    num_blocks = (numel + self.block_size - 1) // self.block_size
                    indices_block = torch.arange(num_blocks, device=p.device, dtype=torch.int32)

                state['idx'] = indices_block

                # 初始化动量缓冲区：仅对应稀疏部分
                n = len(indices_block) * self.block_size
                state['exp_avg'] = torch.zeros(n, dtype=p.dtype, device=p.device)
                state['exp_avg_sq'] = torch.zeros(n, dtype=p.dtype, device=p.device)
                # state['step'] = torch.zeros(n, dtype=torch.int16, device=p.device)
                state['step'] = torch.zeros(len(indices_block), dtype=torch.int16, device=p.device)
                # state['step'] = torch.tensor(0.0, device=p.device)


            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])
            state_steps.append(state['step'])
            indices_list.append(state['idx'])

        return params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps, indices_list

    # ========================
    # 主 step 函数
    # ========================
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a block-sparse Adam update (vectorized)."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        bs = self.block_size

        for group in self.param_groups:
            params, grads, exp_avgs, exp_avg_sqs, state_steps, block_ids = self._init_group(group)

            beta1, beta2 = group['betas']
            eps = group['eps']
            lr = group['lr'] #* group.get('lr_sc', 1.0)
            weight_decay = group['weight_decay']

            base = torch.arange(bs, dtype=torch.int32, device=params[0].device)

            for i, p in enumerate(params):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]
                block_idx = block_ids[i]

                # ---- Expand block_id → element indices ----
                numel = p.numel()
                full_idx = (block_idx[:, None] * bs + base[None, :]).reshape(-1) #.flatten()
                full_idx = full_idx[full_idx < numel]  # 防越界
                len_idx = len(full_idx)

                # --- Expand block-wise steps
                step.add_(1)
                # step_t = step.to(torch.float)
                # step_t = step.repeat_interleave(bs)[:len(full_idx)].to(torch.float)
                step_t = step.unsqueeze(1).expand(-1, bs).reshape(-1)[:len(full_idx)].to(torch.float)

                p_data = p.data.view(-1)
                g = grad.view(-1)[full_idx]
                exp_avg_sel = exp_avg[:len_idx]
                exp_avg_sq_sel = exp_avg_sq[:len_idx]

                # ---- weight decay ----
                if weight_decay != 0:
                    g = g.add(p_data[full_idx], alpha=weight_decay)

                # ---- Adam updates ----
                # exp_avg_sel.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sel.lerp_(g, 1 - beta1)
                exp_avg_sq_sel.mul_(beta2).addcmul_(g, g.conj(), value=1 - beta2)

                if self.decay_steps > 1:
                    decay = torch.clamp(step_t / self.decay_steps, max=self.decay_max)
                    exp_avg_sel.mul_(decay)

                if self.lr_scale:
                    scale_factor = len_idx / p_data.numel()
                    lr /= scale_factor

                bias_correction1 = 1 - beta1 ** step_t
                bias_correction2 = 1 - beta2 ** step_t
                step_size = lr / bias_correction1
                denom = (exp_avg_sq_sel / bias_correction2).sqrt().add_(eps)
                update = exp_avg_sel / denom

                # ---- updates ----
                p_data[full_idx] -= step_size * update

        return loss
