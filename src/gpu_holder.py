"""Temporarily reserve GPU memory on a shared host until a RELEASE file appears."""
import os, sys, time
gpu, gb, release = sys.argv[1], float(sys.argv[2]), sys.argv[3]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu
import torch
x = torch.empty(int(gb * 2**30), dtype=torch.uint8, device="cuda")
print("holding", gb, "GB on GPU", gpu, flush=True)
while not os.path.exists(release):
    time.sleep(0.5)
print("released", flush=True)
