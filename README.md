# Hull–White 1F Monte Carlo on GPU (CUDA)

CUDA implementation of a Monte Carlo engine for the Hull–White one-factor short-rate model. The code builds the zero-coupon curve and forward curve on a uniform time grid, prices a zero-coupon bond call (ZBC), and estimates the ZBC vega with a pathwise (tangent) method and a finite-difference check.

---

## Model

Hull–White one-factor short rate:

$$
dr(t) = \big(\theta(t) - a\,r(t)\big)\,dt + \sigma\,dW_t,\qquad r(0)=r_0
$$

Mean reversion: \(a = 1.0\)  
Volatility: \(\sigma = 0.1\)  
Initial short rate: \(r_0 = 0.012\)  
Time horizon: \(T_{\max} = 10\)  
Uniform grid: \(n_{\text{steps}}\) points with \(dt = \frac{T_{\max}}{n_{\text{steps}}-1}\)

The integral \(\int r_s\,ds\) is approximated with the trapezoidal rule on the uniform grid.

---

## What the program computes

### Q1 — Curve construction (Monte Carlo)

For maturities \(T \in [0,10]\), estimate:

$$
P(0,T) = \mathbb{E}\left[\exp\left(-\int_0^T r_s\,ds\right)\right]
$$

and the forward curve from the bond curve:

$$
f(0,T) = -\frac{d}{dT}\ln P(0,T)
$$

Finite differences are used on the time grid (forward/backward at boundaries, central inside).

---

### Q2 — Theta recovery and ZBC pricing

Theta recovery from the forward curve uses:

$$
\theta(T) = \frac{d}{dT}f(0,T) + a\,f(0,T) + \frac{\sigma^2\big(1-e^{-2aT}\big)}{2a}
$$

The ZBC option priced is a European call on a zero-coupon bond:

$$
\mathrm{ZBC}(S_1,S_2,K)=
\mathbb{E}\left[
\exp\left(-\int_0^{S_1} r_s\,ds\right)\,
\big(P(S_1,S_2)-K\big)_+
\right]
$$

Assignment setup used in `main`:

\(S_1 = 5\)  
\(S_2 = 10\)  
\(K = e^{-0.1}\)

Bond pricing under Hull–White uses the analytic form:

$$
P(t,T) = A(t,T)\,\exp\big(-B(t,T)\,r(t)\big),\qquad
B(t,T)=\frac{1-e^{-a(T-t)}}{a}
$$

---

### Q3 — Vega of ZBC (two methods)

Market data are treated as fixed: the curves \(P(0,T)\) and \(f(0,T)\) from Q1 stay unchanged.

Method 1: pathwise / tangent vega on GPU, propagating \(u(t)=\partial_\sigma r(t)\) using the same Gaussian increments as the rate simulation.

Method 2: central finite difference with common random numbers:

$$
\partial_\sigma \mathrm{ZBC} \approx
\frac{\mathrm{ZBC}(\sigma+h)-\mathrm{ZBC}(\sigma-h)}{2h}
$$

---

## Implementation summary

One GPU thread simulates one full short-rate path.

Deterministic terms are precomputed on CPU and uploaded once per run: \(\theta(t_k)\) on the grid and the one-step drift integral used in the conditional Gaussian transition.

Random numbers are generated per path using cuRAND Philox (`curandStatePhilox4_32_10_t`).

Monte Carlo totals are accumulated with `atomicAdd` into global sums.

---

## Build

Example:

```bash
nvcc -arch=sm_75 HWbuilt.cu -o HW
```

## Run
```bash
./HW
```
