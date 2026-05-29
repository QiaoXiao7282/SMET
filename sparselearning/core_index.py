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
from sparselearning.optimizer_new_block import Adam_block
import re
import datasets
from loguru import logger


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

                    # --- 在 CPU 上生成 block 随机排列，避免大显存占用 ---
                    # perm = torch.randperm(num_blocks, dtype=torch.int32, device='cpu')
                    # block_idx = perm[:k_blocks].to(weight.device, non_blocking=True)
                    # self.indices[name] = block_idx

                    # mask = (torch.rand(weight.shape, device='cpu') < density).float().view(-1)
                    # block_mask = torch.zeros(num_blocks, dtype=torch.bool, device='cpu')
                    # active_idx = mask.nonzero(as_tuple=False)
                    # block_ids = active_idx // self.blocksize
                    # block_mask[block_ids] = True
                    # self.indices[name] = block_mask.nonzero(as_tuple=False).flatten().to(torch.int32).cuda()

                    block_mask = (torch.rand(num_blocks, device='cpu') < density)
                    self.indices[name] = block_mask.nonzero(as_tuple=False).flatten().to(torch.int32).cuda()

                    # mask = (torch.rand(weight.shape) < density).float().data.cuda()  # lsw
                    # self.indices[name] = mask.view(-1).nonzero(as_tuple=False).to(torch.int32)

                    total_params += numel
                    layer_nonzero = min(k_blocks * self.blocksize, numel)

                    layer_density = layer_nonzero / numel
                    total_nonzero += layer_nonzero

                    logger.info(f"  {name:40s} | Density: {layer_density:.4f}")

            overall_density = total_nonzero / total_params
            logger.info(f"\nOverall initial sparsity (density={density}): {overall_density:.4f}")

        ## reinit according sparsity
        if self.args.init_sparse:
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


    def setting_optimizer(self, model):

        self.index_map = {}
        mup_params = []
        nomup_params = []
        oned_params = []

        # 遍历模型的所有参数
        for module in self.modules:
            for name, param in module.named_parameters():

                # if param.dim() >= 2:
                #
                #     if name in self.masks:
                #         mup_params.append(param)
                #     else:
                #         nomup_params.append(param)
                #
                # else:
                #     oned_params.append(param)


                if name not in self.masks:
                    continue

                self.index_map[id(param)] = self.indices[name]


        # optim_groups = [
        #     {'params': mup_params, 'lr_sc': 1.0 / self.args.density},
        #     {'params': nomup_params, 'lr_sc': 1.0 / self.args.density},
        #     {'params': oned_params, 'lr_sc': 1.0 / self.args.density}
        # ]


        if self.args.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        elif self.args.optimizer.lower() == "adamdst":
            self.optimizer = Adam_block(model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay, index_map=self.index_map, block_size=self.blocksize, decay_steps=self.args.op_decay_steps, lr_scale=self.args.lr_scale)

        # for group in self.optimizer.param_groups:
        #     for p in group['params']:
        #         if id(p) not in self.index_map:
        #             print(f"[Skip] Dense layer param {tuple(p.shape)} (no mask)")
        #         else:
        #             assert id(p) in self.index_map, f"Sparse param {p.shape} missing from index_map!"

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
        self.optimizer.step()
        self.prune_rate_decay.step()
        self.apply_mask()

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

        if self.steps % 150 == 0:
            self.check_block_sparsity()

        if self.prune_every_k_steps is not None:

            if self.steps % self.prune_every_k_steps == 0:

                self.print_sparsity_overall()

                self.optimizer.decay_steps = self.args.op_decay_steps
                self.truncate_weights()
                self.t_update = True



    def add_module(self, module, density, sparse_init='ER'):
        self.sparse_init = sparse_init
        self.modules.append(module)
        logger.info('adding module')
        for name, tensor in module.named_parameters():
            logger.info(f'(len: {len(tensor.size())}) size of {name}: {tensor.size()}')

            '''
            if self.args.dense_embedding and 'embed' in name:
                logger.info(f'Keeping embedding layer dense: {name}')
                continue  # skip embedding layer, if requested

            if self.args.dense_head and 'lm_head' in name:
                logger.info(f'Keeping lm_head layer dense: {name}')
                continue  # skip fc layer, if requested
            '''

            if len(tensor.size()) == 4 or len(tensor.size()) == 2:
                self.names.append(name)
                self.masks[name] = []


        self.remove_weight_partial_name('bias')
        self.init_sparse_masks()
        self.setting_optimizer(model=module)
        self.apply_mask()  # apply masks

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
                full_idx = full_idx[full_idx < numel]  # 防越界

                # ---- keep only selected elements ----
                kept = flat[full_idx].clone()  # only k*bs elements → very small
                flat.zero_()
                flat[full_idx] = kept

                # support SGD and Adam
                if self.args.optimizer.lower() == "adam":
                    state = self.optimizer.state.get(tensor)
                    if not state:
                        continue

                    for key in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
                        buf = state.get(key)
                        if isinstance(buf, torch.Tensor):
                            fb = buf.view(-1)
                            kept = fb[full_idx].clone()  # tiny slice
                            fb.zero_()
                            fb[full_idx] = kept


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


        self.apply_mask()


    def magnitude_prune_block(self, block_idx, weight, prune_rate):
        flat = weight.data.view(-1).abs()  # flatten

        # expand block indices → element indices
        starts = block_idx * self.blocksize
        ends = (block_idx + 1) * self.blocksize
        ends = torch.clamp(ends, max=flat.numel())

        block_scores = (flat[starts[:, None] + torch.arange(self.blocksize, device=flat.device)]
                        * (torch.arange(self.blocksize, device=flat.device) < (ends - starts)[:, None])
                        ).sum(dim=1)

        # determine how many blocks to keep
        k = max(1, int(len(block_idx) * (1 - prune_rate)))

        # print(f"len(block_idx)={len(block_idx)}, k={k}, len(block_scores)={block_scores.numel()}")

        # keep top-k: from large to small
        keep = torch.topk(block_scores, k, largest=True).indices

        # try:
        #     keep = torch.topk(block_scores, k, largest=True).indices
        # except RuntimeError as e:
        #     if "selected index k out of range" in str(e):
        #         print(
        #             f"Error: k={k} exceeds block_scores size. Shape: {block_scores.shape}, Numel: {block_scores.numel()}")
        #     raise

        keep_blocks = block_idx[keep]

        # prune = everything else
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

        # 生成每个 block 的随机得分
        scores = torch.rand(num_blocks, device=block_idx.device)

        # 把已有 block 置为 -inf，确保不会被选中
        scores[block_idx] = float('-inf')

        # 直接选 top-k 可用 block → 全向量、无循环
        grow_blocks = torch.topk(scores, grow_size, largest=True).indices
        grow_blocks = grow_blocks.to(block_idx.dtype)

        # 找到需要替换的位置
        replace_index = torch.nonzero(prune_mask, as_tuple=False).flatten()

        # 写回新的 block id
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
                layer_nonzero = len(self.indices[name]) * self.blocksize

                layer_density = layer_nonzero / numel
                total_nonzero += layer_nonzero
                total_params += numel

                logger.info(f"  {name:40s} | Density: {layer_density:.4f}")

        overall_density = total_nonzero / total_params
        logger.info(f"\nOverall initial sparsity (density={self.args.density}): {overall_density:.4f}")


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
        检查模型是否满足 block-wise sparsity。

        model: nn.Module
        index_map: {id(param): block_idx_tensor}
        block_size: int

        返回: None（打印问题）
        """
        logger.info("\n=== Block Sparsity Check ===")

        for module in self.modules:
            for name, param in module.named_parameters():

                if name not in self.masks:
                    continue  # 仅检查稀疏参数

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


