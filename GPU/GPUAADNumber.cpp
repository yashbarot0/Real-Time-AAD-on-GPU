
// ===== GPUAADNumber.cpp =====
#include "GPUAADNumber.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>

GPUAADTape* GPUAADNumber::active_tape_ = nullptr;

GPUAADTape* GPUAADNumber::get_tape_safe() {
    if (!active_tape_) {
        throw std::runtime_error("No active GPUAADTape set - call GPUAADNumber::set_active_tape() first");
    }
    return active_tape_;
}

GPUAADNumber::GPUAADNumber(double val) {
    GPUAADTape* tape = get_tape_safe();
    var_index_ = tape->create_variable(val);
}

GPUAADNumber::GPUAADNumber(int idx) : var_index_(idx) {
    get_tape_safe(); // Verify tape is set
}

GPUAADNumber::GPUAADNumber(const GPUAADNumber& other) : var_index_(other.var_index_) {
    get_tape_safe(); // Verify tape is set
}

GPUAADNumber& GPUAADNumber::operator=(const GPUAADNumber& other) {
    if (this != &other) {
        var_index_ = other.var_index_;
        get_tape_safe(); // Verify tape is set
    }
    return *this;
}

void GPUAADNumber::set_active_tape(GPUAADTape* tape) {
    active_tape_ = tape;
}

GPUAADTape* GPUAADNumber::get_active_tape() {
    return active_tape_;
}

double GPUAADNumber::val() const {
    return get_tape_safe()->get_value(var_index_);
}

double GPUAADNumber::adj() const {
    return get_tape_safe()->get_adjoint(var_index_);
}

int GPUAADNumber::index() const {
    return var_index_;
}

// Enhanced arithmetic operators matching CPU implementation patterns
GPUAADNumber GPUAADNumber::operator+(const GPUAADNumber& other) const {
    GPUAADTape* tape = get_tape_safe();
    double this_val = val();
    double other_val = other.val();
    double result_val = this_val + other_val;
    
    int result_idx = tape->record_binary_op(
        AADOpType::ADD, var_index_, other.var_index_, result_val, 1.0, 1.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator-(const GPUAADNumber& other) const {
    GPUAADTape* tape = get_tape_safe();
    double this_val = val();
    double other_val = other.val();
    double result_val = this_val - other_val;
    
    int result_idx = tape->record_binary_op(
        AADOpType::SUB, var_index_, other.var_index_, result_val, 1.0, -1.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator*(const GPUAADNumber& other) const {
    GPUAADTape* tape = get_tape_safe();
    double this_val = val();
    double other_val = other.val();
    double result_val = this_val * other_val;
    
    int result_idx = tape->record_binary_op(
        AADOpType::MUL, var_index_, other.var_index_, result_val, other_val, this_val);
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator/(const GPUAADNumber& other) const {
    GPUAADTape* tape = get_tape_safe();
    double this_val = val();
    double other_val = other.val();
    
    // Check for division by zero
    if (std::abs(other_val) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Division by zero in GPUAADNumber");
    }
    
    double result_val = this_val / other_val;
    double partial1 = 1.0 / other_val;
    double partial2 = -this_val / (other_val * other_val);
    
    int result_idx = tape->record_binary_op(
        AADOpType::DIV, var_index_, other.var_index_, result_val, partial1, partial2);
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator-() const {
    GPUAADTape* tape = get_tape_safe();
    double this_val = val();
    double result_val = -this_val;
    
    int result_idx = tape->record_unary_op(
        AADOpType::NEG, var_index_, result_val, -1.0);
    return GPUAADNumber(result_idx);
}

// Operators with constants on the right
GPUAADNumber GPUAADNumber::operator+(double rhs) const {
    GPUAADTape* tape = get_tape_safe();
    double this_val = val();
    double result_val = this_val + rhs;
    
    // For constant addition, only this variable contributes to derivative
    int const_idx = tape->record_constant(rhs);
    int result_idx = tape->record_binary_op(
        AADOpType::ADD, var_index_, const_idx, result_val, 1.0, 0.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator-(double rhs) const {
    GPUAADTape* tape = get_tape_safe();
    double this_val = val();
    double result_val = this_val - rhs;
    
    int const_idx = tape->record_constant(rhs);
    int result_idx = tape->record_binary_op(
        AADOpType::SUB, var_index_, const_idx, result_val, 1.0, 0.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator*(double rhs) const {
    GPUAADTape* tape = get_tape_safe();
    double this_val = val();
    double result_val = this_val * rhs;
    
    int const_idx = tape->record_constant(rhs);
    int result_idx = tape->record_binary_op(
        AADOpType::MUL, var_index_, const_idx, result_val, rhs, 0.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator/(double rhs) const {
    if (std::abs(rhs) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Division by zero constant in GPUAADNumber");
    }
    
    GPUAADTape* tape = get_tape_safe();
    double this_val = val();
    double result_val = this_val / rhs;
    
    int const_idx = tape->record_constant(rhs);
    int result_idx = tape->record_binary_op(
        AADOpType::DIV, var_index_, const_idx, result_val, 1.0 / rhs, 0.0);
    return GPUAADNumber(result_idx);
}

// Friend operators for constants on the left side
GPUAADNumber operator+(double lhs, const GPUAADNumber& rhs) {
    GPUAADTape* tape = GPUAADNumber::get_active_tape();
    if (!tape) {
        throw std::runtime_error("No active tape for constant + GPUAADNumber operation");
    }
    
    double rhs_val = rhs.val();
    double result_val = lhs + rhs_val;
    
    int const_idx = tape->record_constant(lhs);
    int result_idx = tape->record_binary_op(
        AADOpType::ADD, const_idx, rhs.index(), result_val, 0.0, 1.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber operator-(double lhs, const GPUAADNumber& rhs) {
    GPUAADTape* tape = GPUAADNumber::get_active_tape();
    if (!tape) {
        throw std::runtime_error("No active tape for constant - GPUAADNumber operation");
    }
    
    double rhs_val = rhs.val();
    double result_val = lhs - rhs_val;
    
    int const_idx = tape->record_constant(lhs);
    int result_idx = tape->record_binary_op(
        AADOpType::SUB, const_idx, rhs.index(), result_val, 0.0, -1.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber operator*(double lhs, const GPUAADNumber& rhs) {
    GPUAADTape* tape = GPUAADNumber::get_active_tape();
    if (!tape) {
        throw std::runtime_error("No active tape for constant * GPUAADNumber operation");
    }
    
    double rhs_val = rhs.val();
    double result_val = lhs * rhs_val;
    
    int const_idx = tape->record_constant(lhs);
    int result_idx = tape->record_binary_op(
        AADOpType::MUL, const_idx, rhs.index(), result_val, 0.0, lhs);
    return GPUAADNumber(result_idx);
}

GPUAADNumber operator/(double lhs, const GPUAADNumber& rhs) {
    GPUAADTape* tape = GPUAADNumber::get_active_tape();
    if (!tape) {
        throw std::runtime_error("No active tape for constant / GPUAADNumber operation");
    }
    
    double rhs_val = rhs.val();
    if (std::abs(rhs_val) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Division by zero in constant / GPUAADNumber");
    }
    
    double result_val = lhs / rhs_val;
    double partial2 = -lhs / (rhs_val * rhs_val);
    
    int const_idx = tape->record_constant(lhs);
    int result_idx = tape->record_binary_op(
        AADOpType::DIV, const_idx, rhs.index(), result_val, 0.0, partial2);
    return GPUAADNumber(result_idx);
}

// Comparison operators (compare values only, no derivatives)
bool GPUAADNumber::operator<(const GPUAADNumber& other) const {
    return val() < other.val();
}

bool GPUAADNumber::operator>(const GPUAADNumber& other) const {
    return val() > other.val();
}

bool GPUAADNumber::operator<=(const GPUAADNumber& other) const {
    return val() <= other.val();
}

bool GPUAADNumber::operator>=(const GPUAADNumber& other) const {
    return val() >= other.val();
}

bool GPUAADNumber::operator==(const GPUAADNumber& other) const {
    return std::abs(val() - other.val()) < std::numeric_limits<double>::epsilon();
}

bool GPUAADNumber::operator!=(const GPUAADNumber& other) const {
    return !(*this == other);
}

// Enhanced math functions with numerical stability
GPUAADNumber log(const GPUAADNumber& x) {
    GPUAADTape* tape = GPUAADNumber::get_active_tape();
    if (!tape) {
        throw std::runtime_error("No active tape for log operation");
    }
    
    double x_val = x.val();
    if (x_val <= 0.0) {
        throw std::runtime_error("log() called with non-positive argument: " + std::to_string(x_val));
    }
    
    double result_val = std::log(x_val);
    double derivative = 1.0 / x_val;
    
    int result_idx = tape->record_unary_op(
        AADOpType::LOG, x.index(), result_val, derivative);
    return GPUAADNumber(result_idx);
}

GPUAADNumber exp(const GPUAADNumber& x) {
    GPUAADTape* tape = GPUAADNumber::get_active_tape();
    if (!tape) {
        throw std::runtime_error("No active tape for exp operation");
    }
    
    double x_val = x.val();
    
    // Check for overflow
    if (x_val > 700.0) {
        throw std::runtime_error("exp() overflow: argument too large: " + std::to_string(x_val));
    }
    
    double result_val = std::exp(x_val);
    double derivative = result_val;
    
    int result_idx = tape->record_unary_op(
        AADOpType::EXP, x.index(), result_val, derivative);
    return GPUAADNumber(result_idx);
}

GPUAADNumber sqrt(const GPUAADNumber& x) {
    GPUAADTape* tape = GPUAADNumber::get_active_tape();
    if (!tape) {
        throw std::runtime_error("No active tape for sqrt operation");
    }
    
    double x_val = x.val();
    if (x_val < 0.0) {
        throw std::runtime_error("sqrt() called with negative argument: " + std::to_string(x_val));
    }
    
    double result_val = std::sqrt(x_val);
    double derivative = (result_val > 0.0) ? 0.5 / result_val : 0.0;
    
    int result_idx = tape->record_unary_op(
        AADOpType::SQRT, x.index(), result_val, derivative);
    return GPUAADNumber(result_idx);
}

GPUAADNumber erf(const GPUAADNumber& x) {
    GPUAADTape* tape = GPUAADNumber::get_active_tape();
    if (!tape) {
        throw std::runtime_error("No active tape for erf operation");
    }
    
    double x_val = x.val();
    double result_val = std::erf(x_val);
    
    // Derivative: d/dx erf(x) = (2/√π) * exp(-x²)
    double derivative = (2.0 / std::sqrt(M_PI)) * std::exp(-x_val * x_val);
    
    int result_idx = tape->record_unary_op(
        AADOpType::ERF, x.index(), result_val, derivative);
    return GPUAADNumber(result_idx);
}

GPUAADNumber norm_cdf(const GPUAADNumber& x) {
    // Use the same step-by-step approach as CPU implementation
    // Φ(x) = 0.5 * (1 + erf(x/√2))
    GPUAADNumber sqrt2(std::sqrt(2.0));
    GPUAADNumber x_scaled = x / sqrt2;
    GPUAADNumber erf_result = erf(x_scaled);
    GPUAADNumber one(1.0);
    GPUAADNumber half(0.5);
    
    return half * (one + erf_result);
}

// Additional math functions for Black-Scholes
GPUAADNumber pow(const GPUAADNumber& base, double exponent) {
    GPUAADTape* tape = GPUAADNumber::get_active_tape();
    if (!tape) {
        throw std::runtime_error("No active tape for pow operation");
    }
    
    double base_val = base.val();
    if (base_val <= 0.0 && exponent != std::floor(exponent)) {
        throw std::runtime_error("pow() with negative base and non-integer exponent");
    }
    
    double result_val = std::pow(base_val, exponent);
    double derivative = (base_val != 0.0) ? exponent * std::pow(base_val, exponent - 1.0) : 0.0;
    
    int result_idx = tape->record_unary_op(
        AADOpType::EXP, base.index(), result_val, derivative); // Reuse EXP type for now
    return GPUAADNumber(result_idx);
}

GPUAADNumber abs(const GPUAADNumber& x) {
    double x_val = x.val();
    if (x_val >= 0.0) {
        return x;
    } else {
        return -x;
    }
}

GPUAADNumber max(const GPUAADNumber& a, const GPUAADNumber& b) {
    return (a.val() >= b.val()) ? a : b;
}

GPUAADNumber min(const GPUAADNumber& a, const GPUAADNumber& b) {
    return (a.val() <= b.val()) ? a : b;
}

// Safe math functions with numerical stability
GPUAADNumber safe_log(const GPUAADNumber& x) {
    double x_val = x.val();
    const double min_val = 1e-15;
    
    if (x_val <= min_val) {
        // Return log of minimum value to avoid -inf
        return GPUAADNumber(std::log(min_val));
    }
    
    return log(x);
}

GPUAADNumber safe_sqrt(const GPUAADNumber& x) {
    double x_val = x.val();
    
    if (x_val < 0.0) {
        // Return sqrt of 0 for negative values
        return GPUAADNumber(0.0);
    }
    
    return sqrt(x);
}

GPUAADNumber safe_divide(const GPUAADNumber& numerator, const GPUAADNumber& denominator) {
    double denom_val = denominator.val();
    const double min_denom = 1e-15;
    
    if (std::abs(denom_val) < min_denom) {
        // Return 0 for division by very small numbers
        return GPUAADNumber(0.0);
    }
    
    return numerator / denominator;
}
