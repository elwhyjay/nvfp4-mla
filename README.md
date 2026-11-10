# kernel benchmark via modal

```sh
uv run modal run bench.py::main --script kernels/vec_add.py --kernel regex:vec_add   # ncu profile
uv run modal run bench.py::sh --cmd "python kernels/vec_add.py"                      # plain run
uv run modal run bench.py::sh --cmd "pytest -q test_decode.py"                       # NVFP4 MLA decode tests
uv run modal run bench.py::sh --cmd "python bench_local.py"                          # ours vs FlashInfer baselines
uv run modal run bench.py::sh --cmd "python kernels/step0.py"                        # achievable BW / mma peak
```

`GPU=B200` prefix for SM100. Status and results: NVFP4_MLA_DECODE_PLAN.md §17.
