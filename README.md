# Hull–White 1F Monte Carlo on GPU

CUDA program that runs a Monte Carlo workflow for the Hull–White one-factor short-rate model: curve construction (zero-coupon + forward), ZBC pricing, and ZBC vega estimation (pathwise + finite-difference check).

Project description refers to `Sujets2026.pdf`.

---

## Key implementation points

Model parameters are stored in a single `HWParams` struct on host, then copied to device constant memory (`__constant__ HWParams d_params`) for fast read-only access.

Deterministic time-grid data are precomputed once on CPU (`theta[t_k]` and `drift[k]`), uploaded to GPU global memory, and exposed to device code through device symbols (`__device__ double* d_theta`, `__device__ double* d_drift`).

Random states are allocated as a per-path array (`RNGState* d_states`) and initialized by a dedicated kernel (`init_rng`) using Philox.

Monte Carlo results are accumulated on GPU into global sums:
`sumP[k]` for the curve (array accumulator) and scalar accumulators for option price / vega. Accumulation uses `atomicAdd` to combine contributions across threads.

The workflow is organized as host-side driver functions that own the “CPU precompute → upload → GPU kernels → copy back” sequence:
`simulate_zero_coupon_curve`, `recover_theta_from_forward_on_gpu`, `simulate_zbc_option_on_gpu`, `simulate_zbc_price_and_vega_on_gpu`, and `zbc_sigma_fd_on_gpu`.

For the vega comparison, the finite-difference run reuses the same curve inputs and uses common random numbers via fixed seeds to reduce noise.

---

## Build

```bash
nvcc -O3 -arch=sm_75 HWbuilt.cu -o HW
```

---

## Run
```bash
./HW [nPaths] [nSteps]
```

---

## Authors

Yanheng Liu, Sorbonne Université  

Long Qian, Sorbonne Université
