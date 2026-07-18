# Performance Log

Timestamped performance results for optimization work, newest at the bottom.

Format (one row per measured step):

```
YYYY-MM-DD HH:MM:SS | commit_hash | step_description | total_time=X.XXs | metric1 | metric2
```

## How to measure (GPU offload path, NVIDIA A100 / Perlmutter)

This branch's changes are in the `USE_HIJ_OMP_OFFLOAD` matrix-free path, which only builds/runs on
GPU with nvc++. Capture each row on Perlmutter, not locally:

1. Build: `cmake -DSBD_GPU=ON -DCMAKE_CXX_COMPILER=nvc++ -DSBD_GPU_ARCH=cc80 ..`
2. Run the N1 debug case with `OMP_TARGET_OFFLOAD=MANDATORY` (forces a hard abort if a target
   region can't run on the device — so a completed run proves the kernels executed on the GPU).
3. `total_time` = the driver's `Elapsed time for davidson` line (and/or `Elapsed time for mult`).
4. GPU activity: `gpu_sm_clock_max_mhz` from the diagnostics header (should sustain near boost, not
   the ~210 MHz idle floor) and DCGM power/clock from the sidecar.
5. Profile: Nsight Systems (`nsys profile`) / `ncu` for kernel vs memcpy time. (The repo CLAUDE.md
   mentions `rocprof --stats`; that is AMD/ROCm — this hardware is NVIDIA, so use nsys/ncu here.)
6. Correctness gate before trusting any number: converged energy must match the CPU build within
   **relative** tolerance (repo CLAUDE.md rule 1).

## Results

<!-- TEMPLATE ROWS — replace placeholders with real measurements on Perlmutter -->

<!--
BASELINE (pre-change, commit 8eada1a "added code modifications from IBM Research"):
YYYY-MM-DD HH:MM:SS | 8eada1a | baseline N1 debug (L20,U1,t1, batch=1024) | total_time=___s | davidson=___s | mult=___s | gpu_sm_clock_max=___MHz | gpu_power_max=___W

AFTER host<->GPU transfer reduction (this branch, commit ______):
YYYY-MM-DD HH:MM:SS | _______ | resident connectivity + Wb; per-task T only | total_time=___s | davidson=___s | mult=___s | gpu_sm_clock_max=___MHz | gpu_power_max=___W | speedup=__x
-->
