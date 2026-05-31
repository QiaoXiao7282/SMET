from __future__ import print_function
import torch
import copy
import numpy as np
import math
import wandb
from sparselearning.decay import CosineDecay, LinearDecay, ConstantDecay, WSDDecay
from transformers import AutoConfig, AutoTokenizer, default_data_collator
import torch.distributed as dist
from peft_pretraining import training_utils, args_utils
from sparselearning.optimizer_dst import Adam_block
import re
import datasets
from loguru import logger
from sparselearning.sparse_layer import SparseLinear


class Masking(object):
    """
    Controls the dynamic sparsity patterns in neural networks during training.

    This class manages the complete lifecycle of sparse training, including initialization,
    weight pruning, weight regrowth across the network.
    It supports various sparse training algorithms through different combinations of
    prune_mode and growth_mode parameters. For example:
    - RigL: prune_mode='magnitude', growth_mode='gradient'
    - SET: prune_mode='magnitude', growth_mode='random'

    """

    def __init__(
            self,
            preprocess_batched,
            pad_idx,
            growth_prune_ratio=1.0,
            redistribution_mode='none',
            threshold=0.001,
            args=None,
            distributed=False,
            device=None,
            global_rank=0,
            world_size=0
    ):
        self.args = args
        self.distributed = distributed
        if device is None:
            self.device = torch.device('cuda')
        else:
            self.device = device

        self.growth_mode = args.growth
        self.prune_mode = args.prune
        self.growth_prune_ratio = growth_prune_ratio
        self.redistribution_mode = redistribution_mode

        self.masks = {}

        self.grads = {}
        self.scores = {}
        self.modules = []
        self.names = []

        self.adjusted_growth = 0
        self.adjustments = []
        self.baseline_nonzero = None
        self.name2baseline_nonzero = {}

        self.preprocess_batched = preprocess_batched
        self.pad_idx = pad_idx
        self.global_rank = global_rank
        self.world_size = world_size

        # stats
        self.momentum_dict = {}
        self.name2variance = {}
        self.name2zeros = {}
        self.name2nonzeros = {}
        self.total_variance = 0
        self.total_removed = 0
        self.total_zero = 0
        self.total_nonzero = 0
        self.prune_rate = args.prune_rate
        self.name2prune_rate = {}
        self.name2density = {}
        self.steps = 0

        # global growth/prune state
        self.threshold = threshold
        self.growth_threshold = threshold
        self.growth_increment = 0.2
        self.increment = 0.2
        self.tolerance = 0.02
        if self.args.fix:
            self.prune_every_k_steps = None
        else:
            self.prune_every_k_steps = args.update_frequency

        self.indices = {}
        self.blocksize = self.args.blocksize

        self.set_prune_rate_decay()

        ##
        self.pre_weights = {}
        self.pre_idx = {}
        self.t_update = False
        self.grads_sparse = {}

    def synchronize_masks(self):
        """ Synchronize masks across GPUs. """
        # if self.distributed:
        # for name in self.masks.keys():
        #     torch.distributed.broadcast(self.masks[name], src=0, async_op=False)

        for name, idx in self.indices.items():
            dist.broadcast(idx, src=0, async_op=False)

    def init_sparse_masks(self):

        if self.args.density_decay == 'constant':
            density = self.args.density
        else:
            density = self.args.initial_density

        if self.sparse_init == 'uniform':
            total_params, total_nonzero = 0, 0

            for module in self.modules:
                for name, weight in module.named_parameters():
                    if name not in self.masks:
                        continue

                    if 'embed' in name or 'lm_head' in name:
                        density = 0.5
                    else:
                        density = self.args.density

                    numel = weight.numel()
                    num_blocks = (numel + self.blocksize - 1) // self.blocksize  # 向上取整

                    k_blocks = int(num_blocks * density)

                    block_mask = (torch.rand(num_blocks, device='cpu') < density)
                    self.indices[name] = block_mask.nonzero(as_tuple=False).flatten().to(torch.int32).cuda()

                    total_params += numel
                    layer_nonzero = min(k_blocks * self.blocksize, numel)

                    layer_density = layer_nonzero / numel
                    total_nonzero += layer_nonzero

                    logger.info(f"  {name:40s} | Density: {layer_density:.4f}")

            overall_density = total_nonzero / total_params
            logger.info(f"\nOverall initial sparsity (density={density}): {overall_density:.4f}")

        ## reinit according sparsity
        if self.args.init_sparse and False:
            self.init_model()

    def init_prune_rate(self, prune_rate):
        for name in self.masks:
            self.name2prune_rate[name] = prune_rate

    def init_density_per_layer(self):
        """Record density per layer."""
        for name, mask in self.masks.items():
            density = mask.sum().item() / mask.numel()
            self.name2density[name] = density

    def init_model(self, initializer_range=0.02, min_scale=0.05):
        self.synchronize_masks()

        for module in self.modules:
            for name, tensor in module.named_parameters():
                # skip layers not sparsified
                if name not in self.masks:
                    continue

                if (name.endswith('proj.weight')):
                    scaling = math.sqrt(self.args.mup_width_multiplier * max(self.args.density, 1e-12))
                    std = initializer_range / scaling
                    torch.nn.init.normal_(tensor, mean=0.0, std=std)
                    logger.info(f"[Init] {name:40s} | density={self.args.density:.4f} | "  f"std={std:.6f}")

    def is_trainable_param(self, name):
        if name.endswith("._dummy"):
            return False
        return True

    def setting_optimizer(self, model):

        self.index_map = {}
        params = []

        for module in self.modules:
            for name, param in module.named_parameters():
                if not self.is_trainable_param(name):
                    logger.info("skipping dummy layers!!!")
                    continue

                params.append(param)
                if name not in self.masks:
                    continue

                self.index_map[id(param)] = self.indices[name]

        if self.args.optimizer.lower() == "adamdst":
            self.optimizer = Adam_block(params, lr=self.args.lr, weight_decay=self.args.weight_decay, index_map=self.index_map, block_size=self.blocksize, decay_steps=self.args.op_decay_steps, index_grads=self.grads_sparse)

        self.lr_scheduler = training_utils.get_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.args.scheduler,
            num_training_steps=self.args.total_training_steps,
            warmup_steps=self.args.warmup_steps,
            min_lr_ratio=self.args.min_lr_ratio,
        )


    def step(self):
        """
        Executes a single optimization step in the sparse training loop.

        """
        self.optimizer.index_grads = self.grads_sparse
        self.optimizer.step()
        self.prune_rate_decay.step()
        self.apply_mask()
        self.grads_sparse = {}

        for name in self.masks:
            if self.args.prune_rate_decay == 'cosine':
                self.name2prune_rate[name] = self.prune_rate_decay.get_current_value()
            elif self.args.prune_rate_decay == 'constant':
                self.name2prune_rate[name] = self.args.prune_rate
            elif self.args.prune_rate_decay == 'WSD':
                self.name2prune_rate[name] = self.prune_rate_decay.get_current_value()
            self.prune_rate = self.name2prune_rate[name]


        self.steps += 1
        self.t_update = False

        if self.prune_every_k_steps is not None:

            if self.steps % self.prune_every_k_steps == 0:

                self.print_sparsity_overall()

                self.optimizer.decay_steps = self.args.op_decay_steps
                self.truncate_weights()
                self.t_update = True
                self.update_hook()

    def update_hook(self):

        # Apply hooks to gather gradients for growth selection
        base = torch.arange(self.blocksize, dtype=torch.int32).cuda()

        for module in self.modules:
            for mname, m in module.named_modules():
                if isinstance(m, SparseLinear):
                    pname = f"{mname}.weight"
                    if pname in self.masks:
                        param = m.weight
                        index = self.index_map[id(param)]

                        numel = param.numel()
                        full_idx = (index[:, None] * self.blocksize + base[None, :]).reshape(-1)  # .flatten()
                        full_idx = full_idx[full_idx < numel]

                        m.apply_hook(self.gradient_accumulation_hook(id(param), full_idx, pname))


    def add_module(self, module, density, sparse_init='ER'):
        self.sparse_init = sparse_init
        self.modules.append(module)
        logger.info('adding module')
        for name, tensor in module.named_parameters():
            logger.info(f'(len: {len(tensor.size())}) size of {name}: {tensor.size()}')


            if self.args.dense_embedding and 'embed' in name:
                logger.info(f'Keeping embedding layer dense: {name}')
                continue  # skip embedding layer, if requested

            if self.args.dense_head and 'lm_head' in name:
                logger.info(f'Keeping lm_head layer dense: {name}')
                continue  # skip fc layer, if requested

            if name.endswith("._dummy"):
                continue

            if len(tensor.size()) == 4 or len(tensor.size()) == 2:
                self.names.append(name)
                self.masks[name] = []


        self.remove_weight_partial_name('bias')
        self.init_sparse_masks()
        self.setting_optimizer(model=module)
        self.apply_mask()  # apply masks
        self.update_hook()


    ## hook grad
    def gradient_accumulation_hook(self, param_id, index, pname):

        @torch.no_grad()
        def _hook(grad):
            """
            grad: dense grad of param, shape == param.shape
            """

            # flatten dense grad
            grad_flat = grad.view(-1)
            sparse_grad = grad_flat.index_select(0, index)

            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sparse_grad, op=dist.ReduceOp.SUM)
                sparse_grad /= dist.get_world_size()

            # gradient accumulate
            if param_id not in self.grads_sparse:
                self.grads_sparse[param_id] = sparse_grad.clone()
            else:
                self.grads_sparse[param_id].add_(sparse_grad)

        return _hook

    def remove_weight(self, name):
        if name in self.masks:
            logger.info('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name].shape,
                                                                      self.masks[name].numel()))
            self.masks.pop(name)
        elif name + '.weight' in self.masks:
            logger.info('Removing {0} of size {1} = {2} parameters.'.format(name, self.masks[name + '.weight'].shape,
                                                                      self.masks[name + '.weight'].numel()))
            self.masks.pop(name + '.weight')
        else:
            logger.info('ERROR', name)

    def remove_weight_partial_name(self, partial_name):
        removed = set()
        for name in list(self.masks.keys()):
            if partial_name in name:
                logger.info('Removing {0} of size {1} with {2} parameters...'.format(name, self.masks[name].shape,
                                                                               np.prod(self.masks[name].shape)))
                removed.add(name)
                self.masks.pop(name)

        logger.info('Removed {0} layers.'.format(len(removed)))

        i = 0
        while i < len(self.names):
            name = self.names[i]
            if name in removed:
                self.names.pop(i)
            else:
                i += 1

    def remove_type(self, nn_type):
        for module in self.modules:
            for name, module in module.named_modules():
                if isinstance(module, nn_type):
                    self.remove_weight(name)

    def apply_mask(self, momentum=True):
        """Apply bool masks to all registered parameters."""
        self.synchronize_masks()

        for module in self.modules:
            for name, tensor in module.named_parameters():

                if name not in self.masks:
                    continue

                flat = tensor.data.view(-1)
                block_idx = self.indices[name]  # shape: [n_blocks]
                bs = self.blocksize
                numel = flat.numel()

                # ---- expand block ids -> element ids ----
                base = torch.arange(bs, device=flat.device, dtype=torch.int32)
                full_idx = (block_idx[:, None] * bs + base[None, :]).flatten()
                full_idx = full_idx[full_idx < numel]

                # ---- keep only selected elements ----
                kept = flat[full_idx].clone()  # only k*bs elements → very small
                flat.zero_()
                flat[full_idx] = kept

                # support SGD and Adam
                if self.args.optimizer.lower() == "adam":
                    state = self.optimizer.state.get(tensor)
                    if not state:
                        continue


    def truncate_weights(self):
        """
        Core function responsible for dynamic neural network topology evolution through structured sparsity.

        This method implements the sparse training paradigm by first pruning (truncating) weights based on
        specified criteria, then activating new parameters to maintain a constant sparsity level. The process
        follows these steps:

        1. Collect network statistics for informed decision-making
        2. Calculate parameter redistribution across layers
        3. Remove parameters based on the specified prune_mode
        4. Regrow parameters using the specified growth_mode

        Common weight pruning strategies include:
        - magnitude: Remove smallest magnitude weights (most common)

        Common weight regrowth strategies include:
        - random: Randomly activate new parameters (used in SET method)
        """

        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks: continue

                block_idx = self.indices[name]
                prune_rate = self.name2prune_rate[name]

                if self.args.compare_update:
                    self.pre_idx[name] = block_idx.clone()

                # prune
                prune_blocks = self.magnitude_prune_block(block_idx, weight, prune_rate)

                # growth
                new_block_idx, replace_idx = self.random_grow_block(block_idx, weight, prune_blocks)

                # update in optimizer
                self.optimizer.update_index_map(weight, new_block_idx, replace_idx)
                self.index_map[id(weight)] = new_block_idx

        self.apply_mask()


    def magnitude_prune_block(self, block_idx, weight, prune_rate):
        flat = weight.data.view(-1).abs()  # flatten

        # expand block indices → element indices
        starts = block_idx * self.blocksize
        ends = (block_idx + 1) * self.blocksize
        ends = torch.clamp(ends, max=flat.numel())

        offsets = torch.arange(self.blocksize, device=flat.device)
        idx = starts[:, None] + offsets
        valid = offsets < (ends - starts)[:, None]
        idx = torch.clamp(idx, max=flat.numel() - 1)
        block_scores = (flat[idx] * valid).sum(dim=1)

        # determine how many blocks to keep
        k = max(1, int(len(block_idx) * (1 - prune_rate)))

        # keep top-k: from large to small
        keep = torch.topk(block_scores, k, largest=True).indices
        keep_blocks = block_idx[keep]

        prune_blocks = block_idx[~torch.isin(block_idx, keep_blocks)]

        return prune_blocks


    def random_grow_block(self, block_idx, weight, prune_blocks):
        numel = weight.numel()
        bs = self.blocksize
        num_blocks = (numel + bs - 1) // bs

        grow_size = prune_blocks.numel()
        if grow_size == 0:
            return block_idx, None

        prune_mask = torch.isin(block_idx, prune_blocks)

        scores = torch.rand(num_blocks, device=block_idx.device)
        scores[block_idx] = float('-inf')

        grow_blocks = torch.topk(scores, grow_size, largest=True).indices
        grow_blocks = grow_blocks.to(block_idx.dtype)

        replace_index = torch.nonzero(prune_mask, as_tuple=False).flatten()
        block_idx[replace_index] = grow_blocks

        return block_idx, replace_index


    def print_sparsity_overall(self):

        total_nonzero = 0
        total_params = 0
        for module in self.modules:
            for name, weight in module.named_parameters():
                if name not in self.masks:
                    continue

                numel = weight.numel()
                layer_nonzero = (weight.data != 0).sum().item()

                flat = weight.data.view(-1)
                bs = self.blocksize
                num_blocks = (numel + bs - 1) // bs
                active_blocks = self.indices[name].to(flat.device).long()

                # ---- check whether weight is block-wise sparse ----
                pad_len = num_blocks * bs - numel
                if pad_len > 0:
                    flat_padded = torch.cat([
                        flat,
                        torch.zeros(pad_len, device=flat.device, dtype=flat.dtype)
                    ])
                else:
                    flat_padded = flat

                blocks = flat_padded.view(num_blocks, bs)
                block_has_nonzero = (blocks != 0).any(dim=1)

                active_mask = torch.zeros(num_blocks, dtype=torch.bool, device=flat.device)
                active_mask[active_blocks] = True

                # inactive blocks should be all zero
                bad_inactive = (block_has_nonzero & ~active_mask).nonzero(as_tuple=False).flatten()

                # active blocks are expected to contain nonzero values
                # but newly grown blocks may temporarily be zero before update
                bad_active = (~block_has_nonzero[active_blocks]).nonzero(as_tuple=False).flatten()
                is_blockwise = bad_inactive.numel() == 0
                active_valid = bad_active.numel() == 0

                layer_density = layer_nonzero / numel
                total_nonzero += layer_nonzero
                total_params += numel

                logger.info(f"  {name:40s} | Density: {layer_density:.4f} | Block-wise: {is_blockwise} | active: {active_valid}")


        overall_density = total_nonzero / total_params
        logger.info(f"\nOverall initial sparsity (density={self.args.density}): {overall_density:.4f}")

    def diagnose_active_zero_blocks(self, name, weight, block_idx, bad_active_pos):
        """
        Diagnose why some active blocks are all-zero.

        Args:
            name: parameter name
            weight: parameter tensor
            block_idx: current active block indices, shape [num_active_blocks]
            bad_active_pos: positions of all-zero active blocks inside block_idx

        Returns:
            dict with diagnostic information
        """
        flat = weight.data.view(-1)
        bs = self.blocksize

        diag = {
            "optimizer_state": False,
            "optimizer_idx_match": None,
            "index_map_match": None,
            "bad_active_step_min": None,
            "bad_active_step_max": None,
            "bad_active_exp_avg_nz": None,
            "bad_active_exp_avg_sq_nz": None,

            # new gradient diagnostics
            "last_grad_nz": None,
            "last_grad_abs_sum": None,
            "bad_active_last_grad_block_nz": None,
            "bad_active_last_grad_abs_sum": None,
            "bad_active_last_grad_all_zero": None,
        }

        if bad_active_pos.numel() == 0:
            return diag

        # Current active block index
        block_idx = block_idx.to(device=flat.device, dtype=torch.long)
        bad_active_pos = bad_active_pos.to(device=flat.device, dtype=torch.long)

        # Check self.index_map
        mask_idx = self.index_map.get(id(weight), None)
        if isinstance(mask_idx, torch.Tensor):
            mask_idx = mask_idx.to(device=flat.device, dtype=torch.long)
            diag["index_map_match"] = (
                    mask_idx.numel() == block_idx.numel()
                    and torch.equal(mask_idx, block_idx)
            )

        # Check optimizer state
        state = self.optimizer.state.get(weight, None)
        if state is None or len(state) == 0:
            return diag

        diag["optimizer_state"] = True

        opt_idx = state.get("idx", None)
        step = state.get("step", None)
        exp_avg = state.get("exp_avg", None)
        exp_avg_sq = state.get("exp_avg_sq", None)

        if isinstance(opt_idx, torch.Tensor):
            opt_idx = opt_idx.to(device=flat.device, dtype=torch.long)
            diag["optimizer_idx_match"] = (
                    opt_idx.numel() == block_idx.numel()
                    and torch.equal(opt_idx, block_idx)
            )

        if isinstance(step, torch.Tensor) and step.numel() > 1:
            bad_steps = step[bad_active_pos].detach()
            diag["bad_active_step_min"] = bad_steps.min().item()
            diag["bad_active_step_max"] = bad_steps.max().item()

        if isinstance(exp_avg, torch.Tensor) and isinstance(exp_avg_sq, torch.Tensor):
            base = torch.arange(bs, device=flat.device, dtype=torch.long)

            state_full_idx = (
                    bad_active_pos[:, None] * bs + base[None, :]
            ).reshape(-1)

            state_full_idx = state_full_idx[state_full_idx < exp_avg.numel()]

            bad_exp_avg = exp_avg[state_full_idx]
            bad_exp_avg_sq = exp_avg_sq[state_full_idx]

            diag["bad_active_exp_avg_nz"] = (bad_exp_avg != 0).sum().item()
            diag["bad_active_exp_avg_sq_nz"] = (bad_exp_avg_sq != 0).sum().item()

        # -------- last gradient diagnostics --------
        last_grad_block_nz = state.get("last_grad_block_nz", None)
        last_grad_block_abs_sum = state.get("last_grad_block_abs_sum", None)
        last_grad_nz = state.get("last_grad_nz", None)
        last_grad_abs_sum = state.get("last_grad_abs_sum", None)

        if isinstance(last_grad_nz, torch.Tensor):
            diag["last_grad_nz"] = last_grad_nz.item()

        if isinstance(last_grad_abs_sum, torch.Tensor):
            diag["last_grad_abs_sum"] = last_grad_abs_sum.item()

        if isinstance(last_grad_block_nz, torch.Tensor):
            last_grad_block_nz = last_grad_block_nz.to(device=flat.device)

            bad_last_grad_nz = last_grad_block_nz[bad_active_pos]
            diag["bad_active_last_grad_block_nz"] = bad_last_grad_nz.sum().item()
            diag["bad_active_last_grad_all_zero"] = bad_last_grad_nz.sum().item() == 0

        if isinstance(last_grad_block_abs_sum, torch.Tensor):
            last_grad_block_abs_sum = last_grad_block_abs_sum.to(device=flat.device)

            bad_last_grad_abs_sum = last_grad_block_abs_sum[bad_active_pos]
            diag["bad_active_last_grad_abs_sum"] = bad_last_grad_abs_sum.sum().item()

        return diag

    def get_bad_active_dense_grad_stats(self, name, bad_active_blocks, numel, device):
        if not hasattr(self, "last_dense_grads"):
            return None

        if name not in self.last_dense_grads or bad_active_blocks.numel() == 0:
            return None

        grad = self.last_dense_grads[name].to(device=device).view(-1)
        bs = self.blocksize

        bad_active_blocks = bad_active_blocks.to(device=device, dtype=torch.long)
        base = torch.arange(bs, device=device, dtype=torch.long)

        idx = (bad_active_blocks[:, None] * bs + base[None, :]).reshape(-1)
        idx = idx[idx < numel]

        if idx.numel() == 0:
            return None

        g = grad.index_select(0, idx)

        # ---- whole dense gradient stats ----
        grad_nz = (grad != 0).sum().item()
        grad_numel = grad.numel()
        grad_density = grad_nz / grad_numel if grad_numel > 0 else 0.0

        return {
            "nz": (g != 0).sum().item(),
            "numel": g.numel(),
            "abs_sum": g.abs().sum().item(),
            "abs_max": g.abs().max().item(),
            "grad_density": grad_density
        }


    def set_prune_rate_decay(self):
        if self.args.prune_rate_decay == 'cosine':
            self.prune_rate_decay = CosineDecay(
                init_value=self.args.prune_rate,
                T_max=self.args.total_training_steps,
                eta_min=0.005,
            )
        elif self.args.prune_rate_decay == 'linear':
            self.prune_rate_decay = LinearDecay(
                init_value=self.args.prune_rate,
                final_value=0.005,
                num_steps=self.args.total_training_steps,
            )
        elif self.args.prune_rate_decay == 'WSD':
            self.prune_rate_decay = WSDDecay(
                init_value=self.args.prune_rate,
                total_steps=self.args.total_training_steps,
            )
        elif self.args.prune_rate_decay == 'constant':
            self.prune_rate_decay = ConstantDecay(self.args.prune_rate)
        else:
            raise Exception(f'Unknown prune_rate_decay mode: {self.args.prune_rate_decay}')


    def check_block_sparsity(self):
        """

        model: nn.Module
        index_map: {id(param): block_idx_tensor}
        block_size: int

        """
        logger.info("\n=== Block Sparsity Check ===")

        for module in self.modules:
            for name, param in module.named_parameters():

                if name not in self.masks:
                    continue  #

                block_idx = self.indices[name]
                flat = param.data.view(-1)
                numel = flat.numel()
                block_size = self.blocksize
                num_blocks = (numel + block_size - 1) // block_size

                # ---- A. reshape/zero-pad to block view ----
                pad_len = num_blocks * block_size - numel
                if pad_len > 0:
                    flat_padded = torch.cat([flat, torch.zeros(pad_len, device=flat.device, dtype=flat.dtype)])
                else:
                    flat_padded = flat

                blocks = flat_padded.view(num_blocks, block_size)

                # block_has_nonzero shape: (num_blocks,)
                block_has_nonzero = (blocks != 0).any(dim=1)

                # --------------------------------------------------------------
                # B. Active blocks must have at least 1 nonzero
                # --------------------------------------------------------------
                active_mask = torch.zeros(num_blocks, dtype=torch.bool, device=flat.device)
                active_mask[block_idx] = True

                active_nonzero = block_has_nonzero[block_idx]
                bad_active = (~active_nonzero).nonzero(as_tuple=False).flatten().tolist()

                if bad_active:
                    logger.info(f"[WARNING] {name}: {len(bad_active)} active blocks are fully zero.")

                # --------------------------------------------------------------
                # C. Inactive blocks must be all zero
                # --------------------------------------------------------------
                bad_inactive = (block_has_nonzero & ~active_mask).nonzero(as_tuple=False).flatten().tolist()

                if bad_inactive:
                    logger.info(f"[WARNING] {name}: {len(bad_inactive)} inactive blocks contain non-zero values.")

        logger.info("=== Block Sparsity Check Completed ===\n")


def weight_init(weight, embedding=False):
    """Initialize weights using the original initialization scheme."""
    if embedding:
        std_embedding = (2 / 5) ** 0.5  # approx 0.632
        weight.data.normal_(mean=0.0, std=std_embedding)
    else:
        # std = config.initializer_range
        std = 0.02  # default value
        weight.data.normal_(mean=0.0, std=std)


