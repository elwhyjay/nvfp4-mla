import subprocess
import torch

p = torch.cuda.get_device_properties(0)
for k in ["name", "major", "minor", "multi_processor_count", "total_memory", "L2_cache_size",
          "shared_memory_per_multiprocessor", "shared_memory_per_block_optin",
          "regs_per_multiprocessor", "max_threads_per_multi_processor"]:
    print(f"{k:34s} {getattr(p, k, 'n/a')}")
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,clocks.max.sm,clocks.max.mem",
                          "--format=csv"], capture_output=True, text=True).stdout)
except FileNotFoundError:
    print("nvidia-smi: n/a")
