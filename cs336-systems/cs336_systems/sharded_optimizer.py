import torch
import torch.optim as optim
from typing import Any, Type, Optional, Callable
import torch.distributed as dist


class ShardedOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        defaults = kwargs
        self.optimizer = None
        super(ShardedOptimizer, self).__init__(params, defaults=defaults)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.idx = 0
        sub_param_groups = []
        for group in self.param_groups:
            for p in group["params"]:
                group_copy = {k: v for k, v in group.items() if k != "params"}
                if self.idx % self.world_size == self.rank:
                    sub_param_groups.append(dict(params=[p], **group_copy))
                self.idx += 1
        assert (
            len(sub_param_groups) > 0
        ), f"Rank {self.rank} has no parameters to optimize. {len(self.param_groups)}"
        self.optimizer = optimizer_cls(sub_param_groups, **kwargs)

    def step(self, closure: Optional[Callable] = None, **kwargs):
        self.optimizer.step(closure, **kwargs)
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                dist.broadcast(p.data, idx % self.world_size)
                idx += 1

    def add_param_group(self, param_group: dict[str, Any]):
        super().add_param_group(param_group)
        if self.optimizer is None:
            return
        for p in param_group["params"]:
            group_copy = {k: v for k, v in param_group.items() if k != "params"}
            if self.idx % self.world_size == self.rank:
                self.optimizer.add_param_group(dict(params=[p], **group_copy))
            self.idx += 1
