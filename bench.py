import os
import pathlib
import re
import shlex
import subprocess
import time

import modal

app = modal.App("kernel-bench")

# GPU=B200 uv run modal run bench.py  (default: SM120 target)
GPU = os.environ.get("GPU", "RTX-PRO-6000")

# ponytail: ncu pinned to 2025.2.1 — 2026.2 (CUDA 13.3) returns NaN counters on Modal B200
NCU = "/opt/nvidia/nsight-compute/2025.2.1/ncu"
image = (
    modal.Image.from_registry("nvidia/cuda:13.3.0-devel-ubuntu24.04", add_python="3.13")
    .apt_install("nsight-compute-2025.2.1", "build-essential")
    .uv_pip_install("ninja~=1.11", "pytest", "torch==2.12.0", "flashinfer-python==0.7.0")
    .env({"TZ": "Asia/Seoul"})  # log filenames in KST
    .add_local_dir(".", remote_path="/root/proj", ignore=[".venv", "out", "sglang", "**/__pycache__", "*.md", "uv.lock"])
)
# flashinfer JIT (~/.cache/flashinfer) + torch cpp_extension builds (~/.cache/torch_extensions)
# + results/ (CSV, raw logs) that local entrypoints pull back into out/
cache = modal.Volume.from_name("kernel-bench-cache", create_if_missing=True)
RESULTS = pathlib.Path("/root/.cache/results")
OUT = pathlib.Path("out")


def gpu_name() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True)
    name = r.stdout.strip().replace("NVIDIA ", "").replace("Blackwell Server Edition", "")
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-") or "gpu"


@app.function(image=image, gpu=GPU, cpu=4, timeout=3600, volumes={"/root/.cache": cache})
def profile(script: str, kernel: str, launch_count: int = 1) -> dict:
    rep = pathlib.Path("/tmp/profile.ncu-rep")
    cmd = [
        NCU, "--set", "full", "--target-processes", "all",
        "--kernel-name", kernel, "--launch-count", str(launch_count),
        "--clock-control", "none", "--replay-mode", "kernel",
        "--force-overwrite", "--export", str(rep),
        "python", *shlex.split(script),
    ]
    r = subprocess.run(cmd, cwd="/root/proj", capture_output=True, text=True)
    if not rep.exists():
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-3000:])
    details = subprocess.run(
        [NCU, "--import", str(rep), "--page", "details"], capture_output=True, text=True
    ).stdout
    return {"gpu": gpu_name(), "rep": rep.read_bytes(), "details": details, "log": r.stdout + r.stderr}


@app.function(image=image, gpu=GPU, cpu=4, timeout=3600, volumes={"/root/.cache": cache})
def run(cmd: str) -> str:
    """Run cmd in /root/proj, stream output live, and tee it to results/logs/ on the Volume."""
    gpu = gpu_name()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^A-Za-z0-9.]+", "_", cmd).strip("_")[:60]
    path = RESULTS / "logs" / f"{stamp}_{gpu}_{slug}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f, subprocess.Popen(
        shlex.split(cmd), cwd="/root/proj", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as p:
        f.write(f"$ {cmd}\n# {gpu}  {stamp}\n\n")
        for line in p.stdout:
            print(line, end="", flush=True)
            f.write(line)
        f.write(f"\n# exit {p.wait()}\n")
    cache.commit()
    return str(path.relative_to("/root/.cache"))


def pull_results() -> None:
    """Copy results/*.csv from the Volume into out/results/."""
    (OUT / "results").mkdir(parents=True, exist_ok=True)
    for e in cache.listdir("results"):
        if e.path.endswith(".csv"):
            (OUT / "results" / pathlib.Path(e.path).name).write_bytes(b"".join(cache.read_file(e.path)))


@app.local_entrypoint()
def sh(cmd: str):
    """uv run modal run bench.py::sh --cmd "python baselines.py" """
    rel = run.remote(cmd)
    local = OUT / "logs" / pathlib.Path(rel).name
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(b"".join(cache.read_file(rel)))
    pull_results()
    print(f"saved {local}")


@app.local_entrypoint()
def main(script: str = "kernels/vec_add.py", kernel: str = "regex:vec_add", launch_count: int = 1):
    """uv run modal run bench.py::main --script kernels/x.py --kernel regex:name"""
    r = profile.remote(script, kernel, launch_count)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", kernel.removeprefix("regex:")).strip("_")
    stem = OUT / "ncu" / f"{pathlib.Path(shlex.split(script)[0]).stem}_{slug}_{r['gpu']}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".ncu-rep").write_bytes(r["rep"])
    stem.with_suffix(".txt").write_text(r["details"])
    stem.with_suffix(".log").write_text(r["log"])
    print(r["details"][-4000:])
    print(f"saved {stem}.ncu-rep / .txt / .log")
