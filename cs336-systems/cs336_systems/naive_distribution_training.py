import cs336_basics.data
import cs336_basics.nn_utils
import cs336_basics.model
import cs336_basics.tokenizer
import cs336_basics.optimizer
import torch
import argparse
import numpy as np
import timeit
import torch.distributed as dist
import os
from datetime import timedelta
from torch import nn
import copy
import time

parser = argparse.ArgumentParser()
# parser.add_argument("--train_set", type=str)
# parser.add_argument("--vocab_set", type=str)
# parser.add_argument("--merges_set", type=str)
parser.add_argument("--num_steps", type=int)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()


def setup():
    print("CUDA visible devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    # These variables are set via srun
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])
    # MASTER_ADDR and MASTER_PORT should have been set in our sbatch script,
    # so we make sure that's the case.
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]
    # Default timeout is 30 minutes. Reducing the timeout here, so the job fails quicker if there's
    # a communication problem between nodes.
    timeout = timedelta(seconds=60)
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timeout)
    return rank, world_size, local_rank, local_world_size


def average_gradients(model, world_size):
    """Gradient averaging."""
    size = float(world_size)
    for i, param in enumerate(model.parameters()):
        print(i, param.grad.data.shape)
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def check_model_close(non_parallel_model, ddp_model, rank):
    for (non_parallel_param_name, non_parallel_model_parameter), (
        ddp_model_param_name,
        ddp_model_parameter,
    ) in zip(non_parallel_model.named_parameters(), ddp_model.named_parameters()):
        # This parameter was initialized as [2, 2], so we expect its value to remain the same
        assert torch.allclose(non_parallel_model_parameter, ddp_model_parameter)


def one_step(model, stand_model, opt, stand_opt, iter, rank, world_size):
    input = torch.rand(
        (args.batch_size, 2048),
        dtype=torch.float32,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(0),
    )

    # Distributed model
    t0 = time.time()
    per_device_batch_size = args.batch_size // world_size
    assert args.batch_size % world_size == 0
    pred = model.forward(
        input[per_device_batch_size * rank : per_device_batch_size * (rank + 1)]
    )
    loss = torch.mean(pred)
    loss.backward()
    t1 = time.time()
    average_gradients(model, world_size)
    t2 = time.time()
    opt.step()
    opt.zero_grad(set_to_none=True)
    t3 = time.time()
    measured_comm_times = [0 for _ in range(world_size)]
    measured_tot_times = [0 for _ in range(world_size)]
    dist.all_gather_object(measured_tot_times, t3-t0)
    dist.all_gather_object(measured_comm_times, t2-t1)

    # Standalone model
    stand_pred = stand_model.forward(input)
    stand_loss = torch.mean(stand_pred)
    stand_loss.backward()
    stand_opt.step()
    stand_opt.zero_grad(set_to_none=True)

    if rank == 0 and iter % 10 == 0:
        # loss = loss.detach()
        # old_loss = copy.deepcopy(loss)
        # stand_loss = stand_loss.detach()
        # dist.all_reduce(loss, op=dist.reduce_op.SUM)
        if check_model_close(stand_model, model, rank):
            print("Model parameters are close!")
    return sum(measured_tot_times)/world_size, sum(measured_comm_times)/world_size


def main():
    rank, world_size, local_rank, local_world_size = setup()
    print(
        f"World size: {world_size}, global rank: {rank}, "
        f"local rank: {local_rank}, local world size: {local_world_size}"
    )

    torch.manual_seed(0)
    model = ToyModel(2048, 2048).cuda()
    opt = cs336_basics.optimizer.AdamW(model.parameters(), lr=0.0001)

    stand_model = copy.deepcopy(model).cuda()
    stand_opt = cs336_basics.optimizer.AdamW(stand_model.parameters(), lr=0.0001)
    # Warm up
    # one_step(model, stand_model, opt, stand_opt, 0, rank, world_size)
    
    tot_time_tot = 0
    comm_time_tot = 0
    for i in range(args.num_steps):
        tot_time, comm_time = one_step(model, stand_model, opt, stand_opt, i, rank, world_size)
        tot_time_tot += tot_time
        comm_time_tot += comm_time
    print(f"Average one-step time: {tot_time_tot/args.num_steps}")
    print(f"Average communication time: {comm_time_tot/args.num_steps}")


if __name__ == "__main__":
    main()
