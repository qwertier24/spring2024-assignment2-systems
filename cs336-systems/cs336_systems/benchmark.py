import cs336_basics.data
import cs336_basics.nn_utils
import cs336_basics.model
import cs336_basics.tokenizer
import cs336_basics.optimizer
import tiktoken
import torch
import argparse
import numpy as np
import timeit
from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser()
parser.add_argument("--train_set", type=str)
parser.add_argument("--forward_only", type=bool, default=False)
parser.add_argument("--warmup_steps", type=int)
parser.add_argument("--measure_steps", type=int)
args = parser.parse_args()


def one_step(model, train_set, opt, iter, forward_only):
    input, target = cs336_basics.data.get_batch(
        train_set, 32, context_length=256, device="cuda"
    )
    with record_function("forward_pass"):
        with torch.autocast(device_type="cuda"):
            pred = model.forward(input)
        loss = cs336_basics.nn_utils.cross_entropy(pred, target)

    if not forward_only:
        with record_function("backward_pass"):
            loss.backward()
        cs336_basics.nn_utils.clip_gradient(model.parameters(), 0.1)
        with record_function("optimizer"):
            opt.step()
            opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    if iter % 10 == 0:
        print(iter, loss)


def main():
    tokenizer = cs336_basics.data.get_tokenizer_from_vocab_merges_path(
        "/data/stanford_cs336/TinyStoriesV2-GPT4-vocab.json",
        "/data/stanford_cs336/TinyStoriesV2-GPT4-merges.txt",
        ["<|endoftext|>"],
    )
    with open(args.train_set) as f:
        train_set = np.concatenate(
            [
                np.array(tokenizer.encode(l.rstrip()), dtype=np.int64)
                for l in f.readlines()[:100]
            ]
        )

    model = (
        cs336_basics.model.BasicsTransformerLM(
            vocab_size=10000,
            context_length=256,
            d_model=512,
            num_layers=4,
            num_heads=16,
            d_ff=2048,
            attn_pdrop=0.1,
            residual_pdrop=0.1,
        )
        .to("cuda")
        .cuda()
    )
    opt = cs336_basics.optimizer.AdamW(model.parameters(), lr=0.0001)
    for i in range(args.warmup_steps):
        one_step(model, train_set, opt, i, False)
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        for i in range(args.measure_steps):
            one_step(model, train_set, opt, i, args.forward_only)
            prof.step()

    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))


if __name__ == "__main__":
    main()
