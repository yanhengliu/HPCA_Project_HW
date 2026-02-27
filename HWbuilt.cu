/**
 * Monte Carlo Simulation of the Hull–White One-Factor Short-Rate Model (CUDA)
 *
 * Project: Monte Carlo simulation of Hull-White model and sensitivities computation (Q1–Q3)
 * File: Single translation unit (e.g., HWbuilt.cu)
 *
 * Author:
 *   - Yanheng Liu 21410437 Sorbonne Université
 *   - Long Qian 21400529 Sorbonne Université
 *
 * Brief description:
 *   This program implements GPU-accelerated Monte Carlo simulation under the Hull–White one-factor
 *   short-rate model:
 *       dr(t) = (theta(t) - a r(t)) dt + sigma dW_t
 *   on a uniform time grid over [0, T_max]. It computes:
 *     (Q1) Zero-coupon bond curve P(0,T) and forward curve f(0,T) for T in [0,10]
 *     (Q2) (a) Recovery of theta(T) from the forward curve using Eq.(10)
 *          (b) Pricing of a ZBC European call option using the analytic bond form under Hull–White
 *     (Q3) Sensitivity (vega) of ZBC w.r.t sigma using a pathwise/tangent method on GPU,
 *          and comparison against a central finite-difference estimate (common random numbers).
 *
 * Numerical / implementation notes:
 *   - Time integrals of r are approximated by the trapezoidal rule on the uniform grid.
 *   - The short-rate transition is simulated via the conditional Gaussian form:
 *         r_{k+1} = r_k e^{-a dt} + drift[k] + sigma_step * G
 *     where drift[k] is a deterministic integral term depending only on k and theta(t),
 *     precomputed once on the host and uploaded to the GPU.
 *   - Random numbers are generated per-path using cuRAND Philox (curandStatePhilox4_32_10_t).
 *   - Monte Carlo sums are accumulated using atomicAdd into global sums.
 *
 * High-level architecture (zones):
 *
 *   1) Library zone
 *      - Standard C/C++ headers and CUDA/cuRAND headers.
 *
 *   2) Utility macro zone
 *      - CUDA_CHECK: error-checking wrapper for CUDA runtime calls.
 *      - RNGState: cuRAND Philox state type alias.
 *
 *   3) Model parameters zone
 *      - HWParams: parameter pack for (a, sigma, r0, T_max, dt, nSteps, sigma_step).
 *      - d_params: HWParams stored in __constant__ memory (read-only on device).
 *      - d_theta / d_drift: device symbols holding pointers to deterministic precomputed arrays.
 *
 *   4) Host precomputation zone
 *      - theta_host(t): piecewise theta(t) used by the statement (for Q1 baseline + diagnostics).
 *      - build_time_grid_and_theta(...):
 *          * builds uniform grid params.dt
 *          * precomputes h_theta[k] = theta(t_k)
 *          * precomputes h_drift[k] = ∫_{t_k}^{t_{k+1}} e^{-a(t_{k+1}-u)} theta(u) du (trapezoid)
 *          * computes sigma_step for the one-step conditional variance
 *      - upload_constants_to_device(...):
 *          * copies HWParams to constant memory
 *          * allocates/copies theta and drift arrays
 *          * publishes device pointers via device symbols
 *      - build_time_grid_and_theta_from_forward(...):
 *          * for “market data fixed” (Q3): rebuilds theta from fixed f(0,T) using Eq.(10)
 *            (theta depends on sigma) and recomputes drift accordingly.
 *
 *   5) Device helper functions zone
 *      - accumulate_trap(I_old, r_old, r_new):
 *          trapezoidal accumulation for ∫ r ds.
 *      - step_short_rate(k, r_curr, state):
 *          one-step update using precomputed drift[k] + Gaussian shock.
 *      - safe_log(x):
 *          numerical guard for log(P) when computing forward rates.
 *
 *   6) Kernel zone
 *      - init_rng(states, seed, nPaths):
 *          initializes Philox RNG state per thread/path.
 *
 *      - mc_zero_coupon(states, sumP, nSteps, nPaths):
 *          Q1: simulates r(t) for each path over the whole grid,
 *          accumulates discount factors exp(-∫_0^{T_k} r ds) into sumP[k] via atomicAdd.
 *
 *      - average_prices_kernel(sumP, P, nSteps, nPaths):
 *          computes P(0,T_k) = sumP[k] / nPaths, with P(0)=1.
 *
 *      - forward_rate_kernel(P, f, nSteps):
 *          computes f(0,T_k) from finite differences of -ln P(0,T) over the grid.
 *
 *      - theta_from_forward_kernel(f, theta, nSteps):
 *          Q2(a): computes theta(T_k) using Eq.(10) by finite differences df/dT and extra term.
 *
 *      - mc_zbc(states, sumPayoff, nPaths, kS1, A12, B12, K):
 *          Q2(b): simulates r up to S1, computes P(S1,S2)=A12 exp(-B12 r(S1)),
 *          and accumulates discounted payoff exp(-∫_0^{S1} r ds) * (P-K)+.
 *
 *      - mc_zbc_sigma_sens(states, sumPayoff, sumDeriv, ...):
 *          Q3: pathwise/tangent method for vega. Propagates r and u=∂σ r using the SAME Gaussian
 *          increment per step, accumulates I=∫ r ds and J=∫ u ds, and adds:
 *              disc * ( ∂σP * 1_{P>K} - J * (P-K)+ )
 *          Also accumulates price in sumPayoff for convenience.
 *
 *   7) Host driver helper zone
 *      - simulate_zero_coupon_curve(...):
 *          orchestrates Q1 end-to-end: precompute -> upload -> RNG init -> MC -> P -> f -> copy back.
 *
 *      - recover_theta_from_forward_on_gpu(f_curve, theta_curve):
 *          Q2(a) driver: runs theta_from_forward_kernel and copies results.
 *
 *      - simulate_zbc_option_on_gpu(...):
 *          Q2(b) driver: computes A12/B12 from market curves and launches mc_zbc.
 *
 *      - simulate_zbc_price_and_vega_on_gpu(...):
 *          Q3 driver: rebuilds theta/drift from fixed forward curve (Eq.10),
 *          launches mc_zbc_sigma_sens to get both price and vega.
 *
 *      - zbc_sigma_fd_on_gpu(...):
 *          Q3 FD driver: central difference using sigma±h and common random numbers (same seed).
 *
 *   8) main()
 *      - Reads optional CLI args: nPaths, nSteps
 *      - Runs Q1, prints full curve (T, P(0,T), f(0,T))
 *      - Runs Q2(a), prints diagnostic errors against the statement theta(t)
 *      - Runs Q2(b), prints ZBC(S1,S2,K)
 *      - Runs Q3, prints price, vega (pathwise), vega (FD), and absolute difference
 *      - Resets CUDA device before exit.
 *
 * Build / run (example):
 *   nvcc -arch=sm_75 HWbuilt.cu -o HW
 *   ./HW 200000 1001
 */


// ------------------------- Library zone -------------------------
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <curand_kernel.h>

// ------------------------- Utility macro zone -------------------------
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err__ = (call);                                     \
        if (err__ != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                 \
                    cudaGetErrorString(err__), __FILE__, __LINE__);     \
            std::exit(EXIT_FAILURE);                                    \
        }                                                               \
    } while (0)

// Type alias for the chosen cuRAND state
using RNGState = curandStatePhilox4_32_10_t;

// ----------------------- Model parameters zone -----------------------

/**
 * Parameters of the Hull-White short-rate model and time grid.
 *
 * dr(t) = (theta(t) - a r(t)) dt + sigma dW_t
 */
struct HWParams {
    double a;          // mean reversion speed
    double sigma;      // volatility of short rate
    double r0;         // initial short rate r(0)
    double T_max;      // maximum maturity (here is 10.0)
    double dt;         // time step for uniform grid
    int    nSteps;     // number of time points, including t=0
    double sigma_step; // Σ_{k,k+1}: one-step std dev of r
};

// Model parameters read-only on device, so transfer them in constant memory.
__constant__ HWParams d_params;

// Device-side pointers for precomputed deterministic arrays:
//   d_theta[k]  = theta(t_k)
//   d_drift[k]  = ∫_{t_k}^{t_{k+1}} e^{-a(t_{k+1}-u)} θ(u) du  for 0 <= k < nSteps-1
__device__ double *d_theta = nullptr;
__device__ double *d_drift = nullptr;

// ----------------------- Host precomputation zone --------------------

/**
 * Piecewise definition of theta(t) on [0, 10], taken from the statement:
 *
 *   θ(t) = 0.012 + 0.0014 * t            for 0 <= t < 5
 *        = 0.019 + 0.001 * (t - 5)       for 5 <= t <= 10
 */
static inline double theta_host(double t)
{
    if (t < 5.0) {
        return 0.012 + 0.0014 * t;
    } else {
        return 0.019 + 0.001 * (t - 5.0);
    }
}

/**
 * Build the time grid and precompute:
 *   - theta(t_k) on the uniform grid
 *   - drift[k] ≈ ∫_{t_k}^{t_{k+1}} e^{-a(t_{k+1}-u)} θ(u) du  (trapezoidal rule)
 *   - sigma_step = Σ_{t_k, t_{k+1}} for the conditional Gaussian of r
 *
 * These quantities are deterministic and depend only on the step index k,
 * not on the Monte Carlo path, so we precompute them once on the host.
 */
void build_time_grid_and_theta(
    double a,
    double sigma,
    double r0,
    double T_max,
    int    nSteps,
    HWParams &params,
    std::vector<double> &h_theta,
    std::vector<double> &h_drift
)
{
    params.a      = a;
    params.sigma  = sigma;
    params.r0     = r0;
    params.T_max  = T_max;
    params.nSteps = nSteps;

    // Uniform time step: last index nSteps-1 corresponds to T_max
    params.dt = T_max / static_cast<double>(nSteps - 1);
    const double dt = params.dt;

    h_theta.resize(nSteps);
    h_drift.resize(nSteps - 1);

    // 1) Precompute theta(t_k) on the grid
    for (int k = 0; k < nSteps; ++k) {
        double t_k = k * dt;
        h_theta[k] = theta_host(t_k);
    }

    // 2) Precompute drift integrals for one-step transition:
    //
    //    m_{k,k+1} = r_k * exp(-a dt) + ∫_{t_k}^{t_{k+1}} e^{-a(t_{k+1}-u)} θ(u) du
    //
    // We store only the integral term:
    //    drift[k] ≈ 0.5 * dt * ( e^{-a dt} θ(t_k) + 1 * θ(t_{k+1}) )
    //
    // This is a 2-point trapezoidal rule on [t_k, t_{k+1}] with the exponential weight.
    for (int k = 0; k < nSteps - 1; ++k) {
        double theta_k   = h_theta[k];
        double theta_kp1 = h_theta[k + 1];

        double w_left  = std::exp(-a * dt); // e^{-a dt} at u = t_k
        double w_right = 1.0;               // e^{0}     at u = t_{k+1}

        h_drift[k] = 0.5 * dt * (w_left * theta_k + w_right * theta_kp1);
    }

    // 3) Precompute one-step standard deviation of r:
    //
    // Σ_{t_k,t_{k+1}} = sqrt( σ^2 (1 - e^{-2 a dt}) / (2 a) )
    params.sigma_step = sigma * std::sqrt((1.0 - std::exp(-2.0 * a * dt)) / (2.0 * a));

}

/**
 * Upload model parameters and precomputed arrays to the device.
 *
 * Steps:
 *   - copy HWParams to constant memory (d_params)
 *   - allocate and copy d_theta and d_drift
 *   - set the device-side pointer symbols d_theta and d_drift
 */

void upload_constants_to_device(
    const HWParams            &h_params,
    const std::vector<double> &h_theta,
    const std::vector<double> &h_drift,
    double                   **d_theta_out,
    double                   **d_drift_out
)
{    
    // Copy parameters to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(d_params, &h_params, sizeof(HWParams)));
    
    const int nSteps = h_params.nSteps;

    // Allocate and copy theta
    double *d_theta_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_theta_raw, nSteps * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_theta_raw, h_theta.data(),
                          nSteps * sizeof(double),
                          cudaMemcpyHostToDevice));
    
    // Allocate and copy drift (nSteps - 1 entries)
    double *d_drift_raw = nullptr;
    CUDA_CHECK(cudaMalloc(&d_drift_raw, (nSteps - 1) * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_drift_raw, h_drift.data(),
                          (nSteps - 1) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Expose them to device code via global device symbols
    CUDA_CHECK(cudaMemcpyToSymbol(d_theta, &d_theta_raw, sizeof(double *)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_drift, &d_drift_raw, sizeof(double *)));

    if (d_theta_out) *d_theta_out = d_theta_raw;
    if (d_drift_out) *d_drift_out = d_drift_raw;
}

/**
 * For Q3: Build theta(T_k) from the (fixed) market forward curve f(0,T) using Eq.(10):
 *   theta(T) = df/dT + a f(0,T) + sigma^2 (1 - exp(-2 a T)) / (2 a)
 *
 * Then precompute drift[k] for r transition:
 *   drift[k] ≈ ∫_{t_k}^{t_{k+1}} e^{-a(t_{k+1}-u)} theta(u) du
 *           ≈ 0.5*dt*( e^{-a dt}*theta_k + 1*theta_{k+1} )
 *
 * This matches the "market data fixed" setup: f(0,T) is fixed, theta depends on sigma.
 */
void build_time_grid_and_theta_from_forward(
    double a,
    double sigma,
    double r0,
    double T_max,
    int    nSteps,
    const std::vector<double> &f_curve,   // fixed market data
    HWParams &params,
    std::vector<double> &h_theta,
    std::vector<double> &h_drift
)
{
    params.a      = a;
    params.sigma  = sigma;
    params.r0     = r0;
    params.T_max  = T_max;
    params.nSteps = nSteps;

    params.dt = T_max / static_cast<double>(nSteps - 1);
    const double dt = params.dt;

    h_theta.resize(nSteps);
    h_drift.resize(nSteps - 1);

    // 1) df/dT finite differences on uniform grid
    //    - forward at k=0, backward at k=nSteps-1, central inside
    auto dfdT_at = [&](int k) -> double {
        if (k == 0) return (f_curve[1] - f_curve[0]) / dt;
        if (k == nSteps - 1) return (f_curve[nSteps - 1] - f_curve[nSteps - 2]) / dt;
        return (f_curve[k + 1] - f_curve[k - 1]) / (2.0 * dt);
    };

    // 2) theta(T_k) from Eq.(10)
    for (int k = 0; k < nSteps; ++k) {
        double T = k * dt;
        double extra = (sigma * sigma) * (1.0 - std::exp(-2.0 * a * T)) / (2.0 * a);
        h_theta[k] = dfdT_at(k) + a * f_curve[k] + extra;
    }

    // 3) drift[k] trapezoid with exponential weight
    for (int k = 0; k < nSteps - 1; ++k) {
        double w_left  = std::exp(-a * dt);
        double w_right = 1.0;
        h_drift[k] = 0.5 * dt * (w_left * h_theta[k] + w_right * h_theta[k + 1]);
    }

    // 4) one-step std dev Σ_{k,k+1}
    params.sigma_step = sigma * std::sqrt((1.0 - std::exp(-2.0 * a * dt)) / (2.0 * a));
}

// ----------------------- Device helper functions ---------------------
/**
 * Trapezoidal accumulation for the time integral of r:
 *
 *   I_{k+1} = I_k + 0.5 * (r_k + r_{k+1}) * dt
 *
 * This approximates ∫_{t_k}^{t_{k+1}} r_s ds.
 */
__device__ inline double accumulate_trap(double I_old,
                                         double r_old,
                                         double r_new)
{
    return I_old + 0.5 * (r_old + r_new) * d_params.dt;
}

/**
 * One-step evolution of the short rate under Hull-White.
 *
 * Using the conditional Gaussian representation:
 *   r_{k+1} = m_{k,k+1} + Σ_{k,k+1} G
 *   m_{k,k+1} = r_k * exp(-a dt) + drift[k]
 *
 * where:
 *   - drift[k] is the precomputed integral term,
 *   - Σ_{k,k+1} = d_params.sigma_step,
 *   - G ~ N(0, 1) from cuRAND.
 */
__device__ inline double step_short_rate(int k,
                                         double r_curr,
                                         RNGState &state)
{
    // integral term for this step
    double drift_k = d_drift[k];    
    double m = r_curr * exp(-d_params.a * d_params.dt) + drift_k;

    // Standard normal random variable
    double G = curand_normal_double(&state);
    
    // Next short rate
    double r_next = m + d_params.sigma_step * G;
    return r_next;
}
/**
 * Safely compute logarithm of P with a small floor to avoid log(0).
 */
__device__ inline double safe_log(double x)
{
    const double eps = 1e-14;
    return log(fmax(x, eps));
}

// --------------------------- Kernel zone -----------------------------
/**
 * RNG initialization kernel.
 *
 * Each path (thread) gets its own Philox state.
 */
__global__
void init_rng(RNGState *states,
              unsigned long long seed,
              int nPaths)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nPaths) return;

    // sequence = id makes each thread use a different subsequence
    curand_init(seed, id, 0ULL, &states[id]);
}

/**
 * Monte Carlo kernel for zero-coupon bond pricing from time 0.
 *
 * Parallelisation strategy:
 *   - One thread = one full short-rate path over the entire time grid.
 *   - For each time step, the thread:
 *       1) evolves r from t_k to t_{k+1},
 *       2) updates integral I ≈ ∫_0^{t_{k+1}} r_s ds via trapezoidal rule,
 *       3) computes discount = exp(-I),
 *       4) atomically adds it to sumP[k+1].
 *
 * Inputs:
 *   states : RNG states (size nPaths)
 *   sumP   : device array of length nSteps, pre-initialised to 0
 *   nSteps : number of time points
 *   nPaths : number of Monte Carlo paths
 *
 * Output:
 *   sumP[k] = sum_{i=1..nPaths} exp( - ∫_0^{t_k} r_s^{(i)} ds )
 */
__global__
void mc_zero_coupon(RNGState *states,
                    double   *sumP,
                    int       nSteps,
                    int       nPaths)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= nPaths) return;

    // Local copy of RNG state for this path
    RNGState localState = states[pid];

    // Initial short rate and integral
    double r = d_params.r0;
    double I = 0.0;

    // At t=0, discount factor is exactly 1; we will set P[0]=1 on host.
    // We only accumulate for k >= 0 --> maturity t_{k+1}.
    for (int k = 0; k < nSteps - 1; ++k) {
        double r_next = step_short_rate(k, r, localState);
        I = accumulate_trap(I, r, r_next);
        
        // discount factor for maturity T = t_{k+1}
        double disc = exp(-I); 

        // Accumulate contribution of this path to sumP[k+1]
        atomicAdd(&sumP[k + 1], disc);

        r = r_next;
    }

    // Write back RNG state
    states[pid] = localState;
}

/**
 * Kernel to compute Monte Carlo estimates of P(0, T_k) from sumP[k].
 *
 * P[k] = sumP[k] / nPaths, with P[0] = 1 exactly.
 */
__global__
void average_prices_kernel(const double *sumP,
                           double       *P,
                           int           nSteps,
                           int           nPaths)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nSteps) return;

    if (k == 0) {
        P[0] = 1.0; 
    } else {
        P[k] = sumP[k] / static_cast<double>(nPaths);
    }
}

/**
 * Kernel to compute the forward rate curve f(0, T_k) from P(0, T_k).
 *
 * Definition:
 *   f(0, T) = d/dT [ ln P(0, T) ].
 *
 * We use finite differences on the time grid:
 *   - forward difference at T_0,
 *   - backward difference at T_{nSteps-1},
 *   - central difference for interior points.
 */
__global__
void forward_rate_kernel(const double *P,
                         double       *f,
                         int           nSteps)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nSteps) return;

    double dt = d_params.dt;
    
    if (k == 0) {
        // forward difference at the left boundary
        double logP0 = safe_log(P[0]);
        double logP1 = safe_log(P[1]);
        f[0] = -(logP1 - logP0) / dt;
    } else if (k == nSteps - 1) {
        // backward difference at the right boundary
        double logPm1 = safe_log(P[nSteps - 2]);
        double logP   = safe_log(P[nSteps - 1]);
        f[nSteps - 1] = -(logP - logPm1) / dt;
    } else {
        // central difference for interior points
        double logPm1 = safe_log(P[k - 1]);
        double logPp1 = safe_log(P[k + 1]);
        f[k] = -(logPp1 - logPm1) / (2.0 * dt);
    }
}

/**
 * Q2(a): Recover theta(T) from Eq. (10) using the computed forward curve f(0,T):
 *
 *   theta(T) = df(0,T)/dT + a f(0,T) + sigma^2 (1 - exp(-2 a T)) / (2 a)
 *
 * We compute df/dT with finite differences on the uniform time grid.
 */
__global__
void theta_from_forward_kernel(const double *f,
                               double *theta,
                               int nSteps)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nSteps) return;

    const double dt = d_params.dt;
    const double a  = d_params.a;
    const double s  = d_params.sigma;

    // Time value T_k
    const double T = k * dt;

    // Finite difference for df/dT
    double dfdT = 0.0;
    if (k == 0) {
        dfdT = (f[1] - f[0]) / dt;
    } else if (k == nSteps - 1) {
        dfdT = (f[nSteps - 1] - f[nSteps - 2]) / dt;
    } else {
        dfdT = (f[k + 1] - f[k - 1]) / (2.0 * dt);
    }

    // Eq. (10)
    const double extra = (s * s) * (1.0 - exp(-2.0 * a * T)) / (2.0 * a);
    theta[k] = dfdT + a * f[k] + extra;
}

/**
 * Q2(b): Monte Carlo simulation of the ZBC option:
 *
 *   ZBC(S1, S2, K) = E[ exp(-∫_0^{S1} r_s ds) * ( P(S1, S2) - K )_+ ].
 *
 * Under Hull-White (given in the statement), we can use:
 *   P(t, T) = A(t,T) * exp( -B(t,T) * r(t) )
 *   B(t,T) = (1 - exp(-a (T-t))) / a
 *   A(t,T) = P(0,T)/P(0,t) * exp( B(t,T) f(0,t) - (sigma^2(1-exp(-2aT))/(4a)) * B(t,T)^2 )
 *
 * For ZBC we only need r(t) and ∫ r ds up to t=S1.
 */

/**
 * Monte Carlo kernel for ZBC(S1, S2, K).
 *
 * Each thread simulates r up to S1 and accumulates the integral I1 = ∫_0^{S1} r ds.
 * Then it computes P(S1,S2) using A12 and B12, and adds the discounted payoff.
 */
__global__
void mc_zbc(RNGState *states,
            double   *sumPayoff,
            int       nPaths,
            int       kS1,     // index of S1 on the time grid
            double    A12,     // A(S1,S2)
            double    B12,     // B(S1,S2)
            double    K)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= nPaths) return;

    RNGState localState = states[pid];

    double r = d_params.r0;
    double I = 0.0;

    // Simulate until time S1 (kS1 steps from 0 to kS1)
    // After the loop, r corresponds to r(S1) and I approximates ∫_0^{S1} r_s ds.
    for (int k = 0; k < kS1; ++k) {
        double r_next = step_short_rate(k, r, localState);
        I = accumulate_trap(I, r, r_next);
        r = r_next;
    }
    
    // Bond price at time S1 for maturity S2
    double P12 = A12 * exp(-B12 * r);

    // Option payoff (P12 - K)+ discounted by exp(-I)
    double payoff = fmax(P12 - K, 0.0);
    double disc   = exp(-I);

    atomicAdd(sumPayoff, disc * payoff);

    states[pid] = localState;
}

/**
 * Q3: Monte Carlo kernel for ZBC sigma-sensitivity using pathwise/tangent method.
 *
 * One thread = one path simulated up to S1 (kS1 steps).
 * We propagate:
 *   r(t)  and  I ≈ ∫_0^{S1} r_s ds,
 *   u(t)=∂σ r(t) and J ≈ ∫_0^{S1} u_s ds,
 * using the SAME Gaussian G at each step (as required by the statement).
 *
 * Bond at S1:
 *   P12 = P(S1,S2) = A12 * exp(-B12 * r(S1)).
 *
 * Derivative:
 *   dP12 = ∂σ P12 = P12 * ( dlnA - B12 * u(S1) ),
 *   dlnA = ∂σ ln A12 = -[ σ(1-exp(-2a S2))/(2a) ] * B12^2   (market data fixed).
 *
 * Contribution per path:
 *   payoff = (P12 - K)+,  disc = exp(-I),
 *   dZ = disc * ( dP12 * 1_{P12>K} - J * payoff ).
 *
 * Outputs (atomic sums):
 *   sumPayoff += disc * payoff
 *   sumDeriv  += dZ
 */

__global__
void mc_zbc_sigma_sens(RNGState *states,
                       double   *sumPayoff,
                       double   *sumDeriv,
                       int       nPaths,
                       int       kS1,
                       double    S2,     // exp(-2a S2) for d/dσ ln A12
                       double    A12,
                       double    B12,
                       double    K)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= nPaths) return;

    RNGState localState = states[pid];

    const double dt = d_params.dt;
    const double a  = d_params.a;
    const double s  = d_params.sigma;
    const double sig_step = d_params.sigma_step;
    const double exp_adt = exp(-a * dt);

    double r  = d_params.r0;
    double I  = 0.0;   // ∫ r ds
    double u  = 0.0;   // u = ∂σ r
    double J  = 0.0;   // ∫ u ds

    for (int k = 0; k < kS1; ++k) {
        // Get drift[k] (already on the device)
        double drift_k = d_drift[k];

        // same  G
        double G = curand_normal_double(&localState);

        // r_{k+1}
        double m = r * exp_adt + drift_k;
        double r_next = m + sig_step * G;

        // u_{k+1} : induction（s=t_k, t=t_{k+1}）
        double t_k   = k * dt;
        double t_kp1 = (k + 1) * dt;

        double extra_det = (2.0 * s * exp(-a * t_kp1) *
                           (cosh(a * t_kp1) - cosh(a * t_k))) / (a * a);

        double u_next = u * exp_adt + extra_det + (sig_step / s) * G;

        // trapezoidal method
        I += 0.5 * (r + r_next) * dt;
        J += 0.5 * (u + u_next) * dt;

        r = r_next;
        u = u_next;
    }

    // P(S1,S2)
    double P12 = A12 * exp(-B12 * r);

    // payoff & discount
    double payoff = fmax(P12 - K, 0.0);
    double disc   = exp(-I);

    // price contribution
    atomicAdd(sumPayoff, disc * payoff);

    // d/dσ ln A12（market data under fixed conditions）
    double dterm = s * (1.0 - exp(-2.0 * a * S2)) / (2.0 * a);
    double dlnA  = - dterm * (B12 * B12);

    // dP12/dσ
    double dP12 = P12 * (dlnA - B12 * u);

    // indicator（The derivative of max）
    double ind = (P12 > K) ? 1.0 : 0.0;

    // dZ = disc*( dP*1_{P>K} - J*(P-K)_+ )
    double dZ = disc * (dP12 * ind - J * payoff);

    atomicAdd(sumDeriv, dZ);

    states[pid] = localState;
}

// ---------------------- Host driver helper zone ----------------------
/**
 * Host helper for Q1
 * High-level driver:
 *   - builds time grid and precomputes theta & drift
 *   - uploads constants to device
 *   - runs Monte Carlo on GPU to compute P(0, T_k)
 *   - computes forward rates f(0, T_k)
 *   - copies results back to host
 */
void simulate_zero_coupon_curve(
    double a,
    double sigma,
    double r0,
    double T_max,
    int    nSteps,
    int    nPaths,
    std::vector<double> &T_grid,
    std::vector<double> &P_curve,
    std::vector<double> &f_curve
)
{
    // 1. Build time grid + precompute theta and drift on host
    HWParams params;
    std::vector<double> h_theta;
    std::vector<double> h_drift;

    build_time_grid_and_theta(a, sigma, r0, T_max, nSteps,
                              params, h_theta, h_drift);

    // 2. Upload parameters and precomputed arrays to device
    double *d_theta_raw = nullptr;
    double *d_drift_raw = nullptr;
    upload_constants_to_device(params, h_theta, h_drift,
                               &d_theta_raw, &d_drift_raw);

    // 3. Allocate device arrays for RNG states and Monte Carlo accumulators
    RNGState *d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, nPaths * sizeof(RNGState)));

    double *d_sumP = nullptr;
    double *d_P    = nullptr;
    double *d_f    = nullptr;

    CUDA_CHECK(cudaMalloc(&d_sumP, nSteps * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_P,    nSteps * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_f,    nSteps * sizeof(double)));

    // Initialize sumP to zero
    CUDA_CHECK(cudaMemset(d_sumP, 0, nSteps * sizeof(double)));

    // Grid/block configuration
    const int blockSize  = 256;
    const int gridPaths  = (nPaths + blockSize - 1) / blockSize;
    const int gridSteps  = (nSteps + blockSize - 1) / blockSize;

    // 4. Initialize RNG
    unsigned long long seed = 1234ULL; // any fixed seed for reproducibility
    init_rng<<<gridPaths, blockSize>>>(d_states, seed, nPaths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Run Monte Carlo simulation to fill sumP
    mc_zero_coupon<<<gridPaths, blockSize>>>(d_states, d_sumP, nSteps, nPaths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Compute Monte Carlo estimates of P(0, T_k)
    average_prices_kernel<<<gridSteps, blockSize>>>(d_sumP, d_P, nSteps, nPaths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 7. Compute forward rate curve f(0, T_k)
    forward_rate_kernel<<<gridSteps, blockSize>>>(d_P, d_f, nSteps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. Copy results back to host
    P_curve.resize(nSteps);
    f_curve.resize(nSteps);
    T_grid.resize(nSteps);

    CUDA_CHECK(cudaMemcpy(P_curve.data(), d_P,
                          nSteps * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(f_curve.data(), d_f,
                          nSteps * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Build time grid on host for convenience
    for (int k = 0; k < nSteps; ++k) {
        T_grid[k] = k * params.dt;
    }

    // 9. Free device memory
    cudaFree(d_states);
    cudaFree(d_sumP);
    cudaFree(d_P);
    cudaFree(d_f);
    cudaFree(d_theta_raw);
    cudaFree(d_drift_raw);
}

/**
 * Host driver for Q2(a): run theta_from_forward_kernel on GPU.
 */
void recover_theta_from_forward_on_gpu(
    const std::vector<double> &f_curve,
    std::vector<double> &theta_curve
)
{
    const int nSteps = static_cast<int>(f_curve.size());
    theta_curve.resize(nSteps);

    double *d_f = nullptr;
    double *d_theta = nullptr;

    CUDA_CHECK(cudaMalloc(&d_f, nSteps * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_theta, nSteps * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_f, f_curve.data(),
                          nSteps * sizeof(double),
                          cudaMemcpyHostToDevice));
    
    const int blockSize = 256;
    const int gridSteps = (nSteps + blockSize - 1) / blockSize;

    theta_from_forward_kernel<<<gridSteps, blockSize>>>(d_f, d_theta, nSteps);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(theta_curve.data(), d_theta,
                          nSteps * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_f);
    cudaFree(d_theta);
}

/**
 * Host helper for Q2(b): run MC for ZBC(5,10,exp(-0.1)).
 *
 * We purposely keep it decoupled from Q1. It reuses:
 *   - build_time_grid_and_theta()
 *   - upload_constants_to_device()
 *   - step_short_rate() and accumulate_trap() in the kernel
 *
 * Inputs:
 *   P_curve, f_curve are taken from Q1 as "market data" on the same grid.
 */

double simulate_zbc_option_on_gpu(
    double a,
    double sigma,
    double r0,
    double T_max,
    int    nSteps,
    int    nPaths,
    const std::vector<double> &T_grid,
    const std::vector<double> &P_curve,
    const std::vector<double> &f_curve,
    double S1,
    double S2,
    double K
)
{
    // Re-upload constants and drift arrays for the kernels.
    // (We do not change simulate_zero_coupon_curve, we keep this function self-contained.)
    HWParams params;
    std::vector<double> h_theta;
    std::vector<double> h_drift;
    build_time_grid_and_theta(a, sigma, r0, T_max, nSteps, params, h_theta, h_drift);

    double *d_theta_raw = nullptr;
    double *d_drift_raw = nullptr;
    upload_constants_to_device(params, h_theta, h_drift, &d_theta_raw, &d_drift_raw);

    const double dt = params.dt;
    
    int kS1 = static_cast<int>(std::llround(S1 / dt));
    int kS2 = static_cast<int>(std::llround(S2 / dt));

    if (kS1 < 1 || kS1 >= nSteps) {
        fprintf(stderr, "Invalid S1 index: kS1=%d (nSteps=%d)\n", kS1, nSteps);
        std::exit(EXIT_FAILURE);
    }
    if (kS2 < 1 || kS2 >= nSteps) {
        fprintf(stderr, "Invalid S2 index: kS2=%d (nSteps=%d)\n", kS2, nSteps);
        std::exit(EXIT_FAILURE);
    }

    // Pull "market data" at t=S1 and T=S2
    const double P0S1 = P_curve[kS1];
    const double P0S2 = P_curve[kS2];
    const double f0S1 = f_curve[kS1];

    // Compute B(S1,S2)
    const double B12 = (1.0 - std::exp(-a * (S2 - S1))) / a;

    // Compute A(S1,S2) using the given analytic expression
    const double term = (sigma * sigma) * (1.0 - std::exp(-2.0 * a * S2)) / (4.0 * a);
    const double A12  = (P0S2 / P0S1) * std::exp(B12 * f0S1 - term * (B12 * B12));

    // Allocate RNG states and payoff accumulator
    RNGState *d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, nPaths * sizeof(RNGState)));

    double *d_sumPayoff = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sumPayoff, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sumPayoff, 0, sizeof(double)));

    // Launch configuration
    const int blockSize = 256;
    const int gridPaths = (nPaths + blockSize - 1) / blockSize;

    // RNG init
    unsigned long long seed = 5678ULL;
    init_rng<<<gridPaths, blockSize>>>(d_states, seed, nPaths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Monte Carlo for ZBC
    mc_zbc<<<gridPaths, blockSize>>>(d_states, d_sumPayoff, nPaths, kS1, A12, B12, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    double h_sumPayoff = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_sumPayoff, d_sumPayoff, sizeof(double), cudaMemcpyDeviceToHost));

    // Average
    double zbc = h_sumPayoff / static_cast<double>(nPaths);

    cudaFree(d_states);
    cudaFree(d_sumPayoff);
    cudaFree(d_theta_raw);
    cudaFree(d_drift_raw);

    return zbc;
}

/**
 * Host helper For Q3(Method 1, price + vega) Compute ZBC(S1,S2,K) and its sigma sensitivity ∂σ ZBC on GPU.
 * Market data (P_curve, f_curve) are fixed (from Q1) on the same grid.
 *
 * Output:
 *   price = E[ exp(-∫_0^{S1} r ds) * (P(S1,S2)-K)+ ]
 *   vega  = ∂σ price (pathwise / tangent method kernel)
 */
void simulate_zbc_price_and_vega_on_gpu(
    double a,
    double sigma,
    double r0,
    double T_max,
    int    nSteps,
    int    nPaths,
    const std::vector<double> &T_grid,   // fixed grid from Q1 (only used for dt check if you want)
    const std::vector<double> &P_curve,  // fixed market data
    const std::vector<double> &f_curve,  // fixed market data
    double S1,
    double S2,
    double K,
    double &price,
    double &vega
)
{
    // 1) Build theta/drift CONSISTENT with fixed forward curve (Eq.10 depends on sigma)
    HWParams params;
    std::vector<double> h_theta, h_drift;
    build_time_grid_and_theta_from_forward(a, sigma, r0, T_max, nSteps, f_curve,
                                          params, h_theta, h_drift);

    double *d_theta_raw = nullptr;
    double *d_drift_raw = nullptr;
    upload_constants_to_device(params, h_theta, h_drift, &d_theta_raw, &d_drift_raw);

    const double dt = params.dt;

    // 2) Map S1,S2 to grid indices
    int kS1 = static_cast<int>(std::llround(S1 / dt));
    int kS2 = static_cast<int>(std::llround(S2 / dt));

    if (kS1 < 1 || kS1 >= nSteps || kS2 < 1 || kS2 >= nSteps) {
        fprintf(stderr, "Invalid S1/S2 indices: kS1=%d kS2=%d (nSteps=%d)\n", kS1, kS2, nSteps);
        std::exit(EXIT_FAILURE);
    }

    // 3) Market data at S1,S2
    const double P0S1 = P_curve[kS1];
    const double P0S2 = P_curve[kS2];
    const double f0S1 = f_curve[kS1];

    // 4) Bond coefficients B12, A12 (given formula)
    const double B12 = (1.0 - std::exp(-a * (S2 - S1))) / a;
    const double term = (sigma * sigma) * (1.0 - std::exp(-2.0 * a * S2)) / (4.0 * a);
    const double A12  = (P0S2 / P0S1) * std::exp(B12 * f0S1 - term * (B12 * B12));

    // 5) Allocate RNG and accumulators (scalars)
    RNGState *d_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_states, nPaths * sizeof(RNGState)));

    double *d_sumPayoff = nullptr;
    double *d_sumDeriv  = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sumPayoff, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sumDeriv,  sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sumPayoff, 0, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sumDeriv,  0, sizeof(double)));

    // 6) Launch config
    const int blockSize = 256;
    const int gridPaths = (nPaths + blockSize - 1) / blockSize;

    // 7) RNG init (fixed seed => reproducible + good for FD common random numbers)
    unsigned long long seed = 9999ULL;
    init_rng<<<gridPaths, blockSize>>>(d_states, seed, nPaths);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8) Main kernel: price + vega
    mc_zbc_sigma_sens<<<gridPaths, blockSize>>>(
        d_states, d_sumPayoff, d_sumDeriv,
        nPaths, kS1, S2, A12, B12, K
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 9) Copy back and average
    double h_sumPayoff = 0.0, h_sumDeriv = 0.0;
    CUDA_CHECK(cudaMemcpy(&h_sumPayoff, d_sumPayoff, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_sumDeriv,  d_sumDeriv,  sizeof(double), cudaMemcpyDeviceToHost));

    price = h_sumPayoff / static_cast<double>(nPaths);
    vega  = h_sumDeriv  / static_cast<double>(nPaths);

    // 10) Free
    CUDA_CHECK(cudaFree(d_states));
    CUDA_CHECK(cudaFree(d_sumPayoff));
    CUDA_CHECK(cudaFree(d_sumDeriv));
    CUDA_CHECK(cudaFree(d_theta_raw));
    CUDA_CHECK(cudaFree(d_drift_raw));
}

/**
 * Host helper Q3(Method 2): Central finite difference approximation of ∂σ ZBC at sigma0:
 *   FD ≈ (ZBC(sigma0+h) - ZBC(sigma0-h)) / (2h)
 *
 * Uses the SAME market data P_curve, f_curve (fixed), and the SAME RNG seed
 * inside simulate_zbc_price_and_vega_on_gpu (so CRN reduces noise).
 */
double zbc_sigma_fd_on_gpu(
    double a,
    double sigma0,
    double r0,
    double T_max,
    int    nSteps,
    int    nPaths,
    const std::vector<double> &T_grid,
    const std::vector<double> &P_curve,
    const std::vector<double> &f_curve,
    double S1,
    double S2,
    double K,
    double h
)
{
    double price_p=0.0, vega_dummy=0.0;
    double price_m=0.0;

    simulate_zbc_price_and_vega_on_gpu(a, sigma0 + h, r0, T_max, nSteps, nPaths,
                                      T_grid, P_curve, f_curve, S1, S2, K,
                                      price_p, vega_dummy);

    simulate_zbc_price_and_vega_on_gpu(a, sigma0 - h, r0, T_max, nSteps, nPaths,
                                      T_grid, P_curve, f_curve, S1, S2, K,
                                      price_m, vega_dummy);

    return (price_p - price_m) / (2.0 * h);
}


// ------------------------------ main ---------------------------------

int main(int argc, char **argv)
{
    // Model parameters from the project
    const double a     = 1.0;
    const double sigma = 0.1;
    const double r0    = 0.012;
    const double T_max = 10.0;

    // Default discretization
    int nPaths = 200000;  // number of Monte Carlo paths
    int nSteps = 1001;    // time steps (dt = 0.01, T_max = 10)

    // Optional command-line overrides: ./hw_mc [nPaths] [nSteps]
    if (argc >= 2) nPaths = std::atoi(argv[1]);
    if (argc >= 3) nSteps = std::atoi(argv[2]);

    if (nPaths <= 0 || nSteps < 2) {
        fprintf(stderr, "Usage: %s [nPaths>0] [nSteps>=2]\n", argv[0]);
        return EXIT_FAILURE;
    }

    printf("Hull-White Monte Carlo (CUDA)\n");
    printf("  a      = %.6f\n", a);
    printf("  sigma  = %.6f\n", sigma);
    printf("  r0     = %.6f\n", r0);
    printf("  T_max  = %.2f\n", T_max);
    printf("  nPaths = %d\n", nPaths);
    printf("  nSteps = %d\n\n", nSteps);

    std::vector<double> T_grid;
    std::vector<double> P_curve;
    std::vector<double> f_curve;

    // -------------------- Question 1 (existing behavior) --------------------
    simulate_zero_coupon_curve(a, sigma, r0, T_max,
                               nSteps, nPaths,
                               T_grid, P_curve, f_curve);

    // Print results as three columns: T, P(0,T), f(0,T)
    printf("# Q1 output: T\tP(0,T)\tf(0,T)\n");
    for (int k = 0; k < nSteps; ++k) {
        printf("%.6f\t%.10f\t%.10f\n",
               T_grid[k],
               P_curve[k],
               f_curve[k]);
    }

    // -------------------- Question 2(a): theta recovery --------------------
    std::vector<double> theta_rec;
    recover_theta_from_forward_on_gpu(f_curve, theta_rec);

    // Print a compact diagnostic (do not spam full curve again)
    // We also compare to the "true" theta(t) given in the statement.
    double maxAbsErr = 0.0;
    double maxAbsTheta = 0.0;

    for (int k = 0; k < nSteps; ++k) {
        double t = T_grid[k];
        double th_true = theta_host(t);
        double err = fabs(theta_rec[k] - th_true);
        if (err > maxAbsErr) maxAbsErr = err;
        if (fabs(th_true) > maxAbsTheta) maxAbsTheta = fabs(th_true);
    }

    printf("\n# Q2(a) theta recovery using Eq.(10)\n");
    printf("# Diagnostics: max|theta_rec - theta_true| = %.6e (relative %.6e)\n",
           maxAbsErr, (maxAbsTheta > 0.0 ? maxAbsErr / maxAbsTheta : 0.0));

    // Print a few key points around the kink t  =5
    auto print_theta_point = [&](double t_query) {
        double dt = T_grid[1] - T_grid[0];
        int k = static_cast<int>(std::llround(t_query / dt));
        if (k < 0) k = 0;
        if (k >= nSteps) k = nSteps - 1;
        double t = T_grid[k];
        double th_true = theta_host(t);
        printf("  t=%.2f (grid %.6f): theta_rec=%.10f, theta_true=%.10f, err=%.3e\n",
               t_query, t, theta_rec[k], th_true, fabs(theta_rec[k] - th_true));
    };

    print_theta_point(0.0);
    print_theta_point(2.0);
    print_theta_point(4.99);
    print_theta_point(5.00);
    print_theta_point(7.0);
    print_theta_point(10.0);

    // -------------------- Question 2(b): ZBC pricing --------------------
    const double S1 = 5.0;
    const double S2 = 10.0;
    const double K  = std::exp(-0.1);

    double zbc = simulate_zbc_option_on_gpu(
        a, sigma, r0, T_max, nSteps, nPaths,
        T_grid, P_curve, f_curve,
        S1, S2, K
    );

    printf("\n# Q2(b) ZBC pricing on GPU\n");
    printf("  ZBC(S1=%.2f, S2=%.2f, K=exp(-0.1)=%.10f) = %.10f\n", S1, S2, K, zbc);

    // -------------------- Question 3: sigma sensitivity of ZBC --------------------
    double price = 0.0, vega = 0.0;
    simulate_zbc_price_and_vega_on_gpu(
        a, sigma, r0, T_max, nSteps, nPaths,
        T_grid, P_curve, f_curve,
        S1, S2, K,
        price, vega
    );

    double h = 1e-3;
    double fd = zbc_sigma_fd_on_gpu(
        a, sigma, r0, T_max, nSteps, nPaths,
        T_grid, P_curve, f_curve,
        S1, S2, K, h
    );

    printf("\n# Q3 sigma sensitivity of ZBC (sigma=%.6f)\n", sigma);
    printf("  ZBC price  = %.10f\n", price);
    printf("  vega (MC)  = %.10f\n", vega);
    printf("  vega (FD)  = %.10f   (h=%.1e)\n", fd, h);
    printf("  abs diff   = %.3e\n", fabs(vega - fd));

    CUDA_CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}