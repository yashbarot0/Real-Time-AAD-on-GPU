// ===== numerical_stability.cu =====
// Enhanced numerical stability and edge case handling for GPU AAD

#include "AADTypes.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>

// Forward declarations for functions used in blackscholes_aad_kernels.cu
__device__ bool check_array_bounds(int index, int max_size);
__device__ double safe_array_read(const double* array, int index, int max_size, double default_value);
__device__ void safe_array_write(double* array, int index, int max_size, double value);
__device__ bool validate_and_sanitize_parameters(double* S, double* K, double* T, double* r, double* sigma);
__device__ void handle_black_scholes_edge_cases(double S, double K, double T, double r, double sigma,
    double* price, double* delta, double* vega, double* gamma, double* theta, double* rho);
__device__ void apply_graceful_degradation(double* price, double* delta, double* vega, double* gamma, double* theta, double* rho);
__device__ int detect_numerical_errors(double S, double K, double T, double r, double sigma,
    double price, double delta, double vega, double gamma, double theta, double rho);

// Numerical constants for stability
__constant__ double EPSILON = 1e-15;
__constant__ double MIN_POSITIVE = 1e-100;
__constant__ double MAX_EXP_ARG = 700.0;
__constant__ double MIN_EXP_ARG = -700.0;
__constant__ double MAX_VOLATILITY = 10.0;
__constant__ double MIN_VOLATILITY = 1e-8;
__constant__ double MAX_TIME = 100.0;
__constant__ double MIN_TIME = 1e-8;
__constant__ double MAX_RATE = 1.0;
__constant__ double MIN_RATE = -0.5;

// Enhanced safe mathematical operations with comprehensive error handling
__device__ double safe_log_enhanced(double x) {
    if (x <= 0.0) return log(EPSILON);
    if (x < MIN_POSITIVE) return log(MIN_POSITIVE);
    if (!isfinite(x)) return log(EPSILON);
    return log(x);
}

__device__ double safe_exp_enhanced(double x) {
    if (!isfinite(x)) return (x > 0) ? DBL_MAX : 0.0;
    if (x > MAX_EXP_ARG) return exp(MAX_EXP_ARG);
    if (x < MIN_EXP_ARG) return exp(MIN_EXP_ARG);
    return exp(x);
}

__device__ double safe_sqrt_enhanced(double x) {
    if (x < 0.0) return 0.0;
    if (!isfinite(x)) return (x > 0) ? sqrt(DBL_MAX) : 0.0;
    return sqrt(x);
}

__device__ double safe_divide_enhanced(double numerator, double denominator) {
    if (!isfinite(numerator) || !isfinite(denominator)) {
        if (isfinite(numerator) && numerator != 0.0) {
            return (denominator > 0) ? DBL_MAX : -DBL_MAX;
        }
        return 0.0;
    }
    
    if (fabs(denominator) < EPSILON) {
        if (fabs(numerator) < EPSILON) return 0.0;
        return (numerator * denominator > 0) ? DBL_MAX : -DBL_MAX;
    }
    
    double result = numerator / denominator;
    if (!isfinite(result)) {
        return (numerator * denominator > 0) ? DBL_MAX : -DBL_MAX;
    }
    
    return result;
}

// Enhanced normal CDF with extreme value handling
__device__ double safe_norm_cdf(double x) {
    if (!isfinite(x)) return (x > 0) ? 1.0 : 0.0;
    
    // Handle extreme values with high precision
    if (x < -10.0) return 0.0;
    if (x > 10.0) return 1.0;
    
    // Use built-in erf for better numerical stability
    const double sqrt2 = 1.4142135623730951;
    double erf_arg = x / sqrt2;
    
    // Clamp erf argument to prevent overflow
    if (erf_arg > 6.0) return 1.0;
    if (erf_arg < -6.0) return 0.0;
    
    return 0.5 * (1.0 + erf(erf_arg));
}

// Enhanced normal PDF with overflow protection
__device__ double safe_norm_pdf(double x) {
    if (!isfinite(x)) return 0.0;
    
    // Handle extreme values
    if (fabs(x) > 10.0) return 0.0;
    
    const double inv_sqrt_2pi = 0.3989422804014327;
    double exp_arg = -0.5 * x * x;
    
    // Prevent underflow
    if (exp_arg < MIN_EXP_ARG) return 0.0;
    
    return inv_sqrt_2pi * exp(exp_arg);
}

// Parameter validation and sanitization
__device__ bool validate_and_sanitize_parameters(
    double* S, double* K, double* T, double* r, double* sigma) {
    
    bool valid = true;
    
    // Spot price validation
    if (!isfinite(*S) || *S <= 0.0) {
        *S = 100.0;  // Default reasonable spot price
        valid = false;
    }
    
    // Strike price validation
    if (!isfinite(*K) || *K <= 0.0) {
        *K = *S;  // At-the-money default
        valid = false;
    }
    
    // Time to expiry validation
    if (!isfinite(*T) || *T <= 0.0) {
        *T = MIN_TIME;
        valid = false;
    } else if (*T > MAX_TIME) {
        *T = MAX_TIME;
        valid = false;
    }
    
    // Risk-free rate validation
    if (!isfinite(*r)) {
        *r = 0.05;  // Default 5% rate
        valid = false;
    } else if (*r > MAX_RATE) {
        *r = MAX_RATE;
        valid = false;
    } else if (*r < MIN_RATE) {
        *r = MIN_RATE;
        valid = false;
    }
    
    // Volatility validation
    if (!isfinite(*sigma) || *sigma <= 0.0) {
        *sigma = MIN_VOLATILITY;
        valid = false;
    } else if (*sigma > MAX_VOLATILITY) {
        *sigma = MAX_VOLATILITY;
        valid = false;
    }
    
    return valid;
}

// Bounds checking for array accesses
__device__ bool check_array_bounds(int index, int max_size) {
    return (index >= 0 && index < max_size);
}

// Safe array access with bounds checking
__device__ double safe_array_read(const double* array, int index, int max_size, double default_value = 0.0) {
    if (check_array_bounds(index, max_size)) {
        double value = array[index];
        return isfinite(value) ? value : default_value;
    }
    return default_value;
}

__device__ void safe_array_write(double* array, int index, int max_size, double value) {
    if (check_array_bounds(index, max_size)) {
        array[index] = isfinite(value) ? value : 0.0;
    }
}

// Enhanced Black-Scholes edge case handling
__device__ void handle_black_scholes_edge_cases(
    double S, double K, double T, double r, double sigma,
    double* price, double* delta, double* vega, double* gamma, double* theta, double* rho) {
    
    // Zero time to expiry
    if (T <= MIN_TIME) {
        double intrinsic = fmax(S - K, 0.0);
        *price = intrinsic;
        *delta = (S > K) ? 1.0 : 0.0;
        *vega = 0.0;
        *gamma = 0.0;
        *theta = 0.0;
        *rho = 0.0;
        return;
    }
    
    // Zero or very low volatility
    if (sigma <= MIN_VOLATILITY) {
        double discount_factor = safe_exp_enhanced(-r * T);
        double forward_price = S / discount_factor;
        double intrinsic = fmax(forward_price - K, 0.0) * discount_factor;
        
        *price = intrinsic;
        *delta = (forward_price > K) ? discount_factor : 0.0;
        *vega = 0.0;
        *gamma = 0.0;
        *theta = -r * intrinsic;
        *rho = (forward_price > K) ? T * K * discount_factor : 0.0;
        return;
    }
    
    // Very high volatility (numerical instability region)
    if (sigma >= MAX_VOLATILITY) {
        // For very high volatility, option approaches S - K*exp(-rT)
        double discount_factor = safe_exp_enhanced(-r * T);
        double discounted_strike = K * discount_factor;
        
        *price = fmax(S - discounted_strike, 0.0);
        *delta = 1.0;
        *vega = 0.0;  // Vega approaches 0 for extreme volatility
        *gamma = 0.0;
        *theta = r * discounted_strike;
        *rho = T * discounted_strike;
        return;
    }
    
    // Extreme moneyness cases
    double moneyness = S / K;
    
    // Deep out-of-the-money (S << K)
    if (moneyness < 0.01) {
        *price = 0.0;
        *delta = 0.0;
        *vega = 0.0;
        *gamma = 0.0;
        *theta = 0.0;
        *rho = 0.0;
        return;
    }
    
    // Deep in-the-money (S >> K)
    if (moneyness > 100.0) {
        double discount_factor = safe_exp_enhanced(-r * T);
        *price = S - K * discount_factor;
        *delta = 1.0;
        *vega = 0.0;
        *gamma = 0.0;
        *theta = r * K * discount_factor;
        *rho = T * K * discount_factor;
        return;
    }
}

// Graceful degradation for numerical issues
__device__ void apply_graceful_degradation(
    double* price, double* delta, double* vega, double* gamma, double* theta, double* rho) {
    
    // Check and fix any non-finite values
    if (!isfinite(*price)) *price = 0.0;
    if (!isfinite(*delta)) *delta = 0.0;
    if (!isfinite(*vega)) *vega = 0.0;
    if (!isfinite(*gamma)) *gamma = 0.0;
    if (!isfinite(*theta)) *theta = 0.0;
    if (!isfinite(*rho)) *rho = 0.0;
    
    // Apply reasonable bounds
    *price = fmax(*price, 0.0);  // Price cannot be negative
    *delta = fmax(fmin(*delta, 1.0), 0.0);  // Delta between 0 and 1 for calls
    *vega = fmax(*vega, 0.0);  // Vega cannot be negative for calls
    *gamma = fmax(*gamma, 0.0);  // Gamma cannot be negative
    
    // Theta and Rho can be negative, but should be finite
    if (fabs(*theta) > 1000.0) *theta = (*theta > 0) ? 1000.0 : -1000.0;
    if (fabs(*rho) > 1000.0) *rho = (*rho > 0) ? 1000.0 : -1000.0;
}

// Comprehensive error detection and reporting
__device__ int detect_numerical_errors(
    double S, double K, double T, double r, double sigma,
    double price, double delta, double vega, double gamma, double theta, double rho) {
    
    int error_flags = 0;
    
    // Input parameter errors
    if (!isfinite(S) || S <= 0) error_flags |= 0x01;
    if (!isfinite(K) || K <= 0) error_flags |= 0x02;
    if (!isfinite(T) || T <= 0) error_flags |= 0x04;
    if (!isfinite(r)) error_flags |= 0x08;
    if (!isfinite(sigma) || sigma <= 0) error_flags |= 0x10;
    
    // Output value errors
    if (!isfinite(price)) error_flags |= 0x20;
    if (!isfinite(delta)) error_flags |= 0x40;
    if (!isfinite(vega)) error_flags |= 0x80;
    if (!isfinite(gamma)) error_flags |= 0x100;
    if (!isfinite(theta)) error_flags |= 0x200;
    if (!isfinite(rho)) error_flags |= 0x400;
    
    // Logical consistency errors
    if (price < 0) error_flags |= 0x800;
    if (delta < 0 || delta > 1) error_flags |= 0x1000;
    if (vega < 0) error_flags |= 0x2000;
    if (gamma < 0) error_flags |= 0x4000;
    
    return error_flags;
}