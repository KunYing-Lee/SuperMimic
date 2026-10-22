# RTX 5090 runtime measurements — 2026-10-22

Mars5090: single RTX 5090 (32607 MiB), Core Ultra 9 285K. Source `487ed02`, locked dependencies; OMP/MKL threads set to 1.

All four candidates passed native backend and dataset-reset checks and completed 30 finite training iterations. Measurement fills replay with 1000 warmup steps; rates average iterations 11–30. Peak memory includes initialization and saving. Formal training retains eight warmup steps and 320000 iterations.

| Replay | Activation recomputation | Iterations/s | Peak GPU MiB |
|---|---|---:|---:|
| GPU | Enabled | 1.605 | 18728 |
| GPU | Disabled | **2.171** | 30142 |
| CPU | Disabled | 1.423 | 26420 |
| CPU | Enabled | 1.165 | 14980 |

Selected GPU replay without recomputation, the fastest measured candidate. Network dimensions, batches, 32-step imagination, sampling and optimizer settings remain unchanged. Short measurements establish runtime feasibility, not policy quality or a globally optimal configuration.

Evidence: `/home/leeky/projects/SuperMimic/logs/benchmark_5090_20261022_r2/results.json` and adjacent logs/GPU samples.

The kernel driver is 595.84 while system user-space libraries are 595.99.02. Process-local `LD_LIBRARY_PATH` selects extracted official 595.84 libraries; system drivers are unchanged. Both unversioned and versioned CUDA library links select the same release. Torch/Warp shared-stream checks passed. Official installer SHA256: `99c404e52131bf27bca6da0268dcce3ebedc7a73ab174071a995cf1a6c5eba06`.

The A800 run was stopped after its complete iteration-2000 checkpoint was saved and remains archived there. The Mars run starts independently from seed 0.
