# Hull–White 1F Monte Carlo on GPU (CUDA)

CUDA implementation of a Monte Carlo engine for the Hull–White one-factor short-rate model, covering curve construction, option pricing, and sensitivity (vega) estimation for a zero-coupon bond call (ZBC). This project follows the 2026 assignment statement for “Monte Carlo simulation of Hull-White model and sensitivities computation (Q1–Q3)”. :contentReference[oaicite:0]{index=0}

---

## Model

Hull–White one-factor short rate:
\[
dr(t) = (\theta(t) - a\,r(t))dt + \sigma\,dW_t,\quad r(0)=0.012
\]

- Mean reversion: `a = 1.0`
- Volatility: `sigma = 0.1`
- Time horizon: `T_max = 10`
- Uniform grid with `nSteps` points (`dt = T_max/(nSteps-1)`)

The time integral \(\int r\,ds\) is approximated with the trapezoidal rule on the uniform grid. :contentReference[oaicite:1]{index=1}

---

## What the program computes

### Q1 — Curve construction (Monte Carlo)
For maturities \(T \in [0,10]\), compute:
- Zero-coupon curve \(P(0,T) = \mathbb{E}[\exp(-\int_0^T r_s ds)]\)
- Forward curve \(f(0,T) = -\partial_T \ln P(0,T)\)

### Q2 — Theta recovery + ZBC pricing
Using the curves from Q1 (same discretization):
- **(a)** Recover \(\theta(T)\) from \(f(0,T)\) using:
  \[
  \theta(T) = \partial_T f(0,T) + a f(0,T) + \frac{\sigma^2(1-e^{-2aT})}{2a}
  \]
- **(b)** Price the European call on a zero-coupon bond:
  \[
  ZBC(S_1,S_2,K)=\mathbb{E}\left[e^{-\int_0^{S_1} r_s ds}\,(P(S_1,S_2)-K)_+\right]
  \]
  with the assignment setup: `S1=5`, `S2=10`, `K=exp(-0.1)`.

### Q3 — Vega of ZBC (two methods)
Assuming \(\{P(0,T)\}\) and \(\{f(0,T)\}\) are **fixed market data**:
- **Pathwise / tangent method** on GPU (propagates \(u(t)=\partial_\sigma r(t)\) using the same Gaussian increments as the rate simulation)
- **Finite difference** baseline with common random numbers:
  \[
  \frac{ZBC(\sigma+h)-ZBC(\sigma-h)}{2h}
  \]

All of the above match the formulas and intent described in the statement. :contentReference[oaicite:2]{index=2}

---

## Implementation notes (high level)

- **One GPU thread = one Monte Carlo path** over the whole time grid.
- **cuRAND Philox** (`curandStatePhilox4_32_10_t`) for per-path RNG.
- Deterministic terms are **precomputed on CPU once** and uploaded:
  - `theta(t_k)` on the grid
  - step drift integral used in the conditional Gaussian transition
- Monte Carlo totals are accumulated using `atomicAdd` into global sums.
- For Q3 (“market data fixed”), `theta` is rebuilt from the fixed forward curve via Eq.(10), which makes it depend on `sigma`.

---

## Build

Requirements:
- NVIDIA GPU + CUDA toolkit (nvcc)
- A supported compute capability (set `-arch=` accordingly)

Example:
```bash
nvcc -O3 -arch=sm_75 HWbuilt.cu -o HW
