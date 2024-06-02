import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str)
parser.add_argument("--data_size_kb", type=int)
parser.add_argument("--device", type=str)
parser.add_argument("--num_proc", type=int)
args = parser.parse_args()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.rand((256, args.data_size_kb)).to(args.device)
    print(f"rank {rank} data (before all-reduce): {torch.mean(data)}")
    t1 = time.time()
    dist.all_reduce(data, async_op=False)
    torch.cuda.synchronize()
    t2 = time.time()
    print(f"rank {rank} data (after all-reduce): {torch.mean(data)}")
    print(f"time {rank}: ", t2-t1)


def main():
    world_size = args.num_proc
    mp.spawn(fn=distributed_demo, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
