import torch
from torch import nn
import torch.distributed as dist


class DDP(nn.Module):

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.all_reduce_handles = []
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        for param in self.module.parameters():
            dist.scatter(
                param.data,
                (
                    [param.data for _ in range(self.world_size)]
                    if self.rank == 0
                    else None
                ),
                src=0,
            )
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self.average_gradients)

    def average_gradients(self, tensor):
        handle = dist.all_reduce(tensor.grad.data, async_op=True)
        self.all_reduce_handles.append(handle)

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.all_reduce_handles:
            handle.wait()
        self.all_reduce_handles.clear()
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad.data /= self.world_size


class DDPBucketed(nn.Module):

    class Bucket:
        def __init__(self):
            self.params = []
            self.flattened_data = None

        def add_param(self, param):
            self.params.append(param)

        def clear(self):
            self.flattened_data = None

        def complete(self):
            return all([param.grad is not None for param in self.params])

        def all_reduce(self):
            self.flattened_data = torch._utils._flatten_dense_tensors(
                [param.grad.data for param in self.params]
            )
            return dist.all_reduce(self.flattened_data, async_op=True)

        def unflatten_grad_data(self, grad_data):
            unflattened_data = torch._utils._unflatten_dense_tensors(
                grad_data, [param.grad.data for param in self.params]
            )
            for param, data in zip(self.params, unflattened_data):
                param.grad.data = data

    def __init__(self, module: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.all_reduce_handles = []
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.buckets = []
        # self.bucket_id = []
        # self.bucket_sub_id = []
        remaining_size = 0
        for param in reversed(list(self.module.parameters())):
            dist.scatter(
                param.data,
                (
                    [param.data for _ in range(self.world_size)]
                    if self.rank == 0
                    else None
                ),
                src=0,
            )
            if param.requires_grad:
                param_size = torch.numel(param.data) * param.data.element_size()
                if param_size > remaining_size:
                    self.buckets.append(self.Bucket())
                    remaining_size = bucket_size_mb * 1024 * 1024
                remaining_size -= param_size
                param.register_post_accumulate_grad_hook(
                    lambda param, bucket_id=len(
                        self.buckets
                    ) - 1: self.average_gradients(param, bucket_id)
                )
                self.buckets[-1].add_param(param)
                # self.bucket_id.append(len(self.buckets) - 1)
                # self.bucket_sub_id.append(self.buckets[-1].cnt)

    def average_gradients(self, param, bucket_id):
        if self.buckets[bucket_id].complete():
            self.all_reduce_handles.append(self.buckets[bucket_id].all_reduce())

    def before_train_step(self):
        for param in self.module.parameters():
            if param.requires_grad:
                param.grad = None

    def forward(self, *input, **kwargs):
        return self.module(*input, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.all_reduce_handles:
            handle.wait()
        self.all_reduce_handles.clear()
        for bucket in self.buckets:
            bucket.unflatten_grad_data(bucket.flattened_data / self.world_size)
