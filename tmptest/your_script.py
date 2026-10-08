import sys, os, platform
print("hello from", platform.node())
print("python", sys.version.split()[0])
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(),
      "device_count", torch.cuda.device_count())
