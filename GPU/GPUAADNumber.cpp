
// ===== GPUAADNumber.cpp =====
#include "GPUAADNumber.h"
#include <cmath>
#include <stdexcept>

GPUAADTape* GPUAADNumber::active_tape_ = nullptr;

GPUAADNumber::GPUAADNumber(double val) {
    if (!active_tape_) {
        throw std::runtime_error("No active tape set");
    }
    var_index_ = active_tape_->create_variable(val);
}

GPUAADNumber::GPUAADNumber(int idx) : var_index_(idx) {
    if (!active_tape_) {
        throw std::runtime_error("No active tape set");
    }
}

void GPUAADNumber::set_active_tape(GPUAADTape* tape) {
    active_tape_ = tape;
}

GPUAADTape* GPUAADNumber::get_active_tape() {
    return active_tape_;
}

double GPUAADNumber::val() const {
    return active_tape_->get_value(var_index_);
}

double GPUAADNumber::adj() const {
    return active_tape_->get_adjoint(var_index_);
}

int GPUAADNumber::index() const {
    return var_index_;
}

GPUAADNumber GPUAADNumber::operator+(const GPUAADNumber& other) const {
    double result_val = val() + other.val();
    int result_idx = active_tape_->record_binary_op(
        AADOpType::ADD, var_index_, other.var_index_, result_val, 1.0, 1.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator-(const GPUAADNumber& other) const {
    double result_val = val() - other.val();
    int result_idx = active_tape_->record_binary_op(
        AADOpType::SUB, var_index_, other.var_index_, result_val, 1.0, -1.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator*(const GPUAADNumber& other) const {
    double result_val = val() * other.val();
    int result_idx = active_tape_->record_binary_op(
        AADOpType::MUL, var_index_, other.var_index_, result_val, other.val(), val());
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator/(const GPUAADNumber& other) const {
    double other_val = other.val();
    double result_val = val() / other_val;
    int result_idx = active_tape_->record_binary_op(
        AADOpType::DIV, var_index_, other.var_index_, result_val, 
        1.0 / other_val, -val() / (other_val * other_val));
    return GPUAADNumber(result_idx);
}

GPUAADNumber GPUAADNumber::operator-() const {
    double result_val = -val();
    int result_idx = active_tape_->record_unary_op(
        AADOpType::NEG, var_index_, result_val, -1.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber operator+(double lhs, const GPUAADNumber& rhs) {
    double result_val = lhs + rhs.val();
    int const_idx = GPUAADNumber::get_active_tape()->record_constant(lhs);
    int result_idx = GPUAADNumber::get_active_tape()->record_binary_op(
        AADOpType::ADD, const_idx, rhs.index(), result_val, 0.0, 1.0);
    return GPUAADNumber(result_idx);
}

GPUAADNumber operator*(double lhs, const GPUAADNumber& rhs) {
    double result_val = lhs * rhs.val();
    int const_idx = GPUAADNumber::get_active_tape()->record_constant(lhs);
    int result_idx = GPUAADNumber::get_active_tape()->record_binary_op(
        AADOpType::MUL, const_idx, rhs.index(), result_val, 0.0, lhs);
    return GPUAADNumber(result_idx);
}

GPUAADNumber log(const GPUAADNumber& x) {
    double x_val = x.val();
    double result_val = std::log(x_val);
    int result_idx = GPUAADNumber::get_active_tape()->record_unary_op(
        AADOpType::LOG, x.index(), result_val, 1.0 / x_val);
    return GPUAADNumber(result_idx);
}

GPUAADNumber exp(const GPUAADNumber& x) {
    double result_val = std::exp(x.val());
    int result_idx = GPUAADNumber::get_active_tape()->record_unary_op(
        AADOpType::EXP, x.index(), result_val, result_val);
    return GPUAADNumber(result_idx);
}

GPUAADNumber sqrt(const GPUAADNumber& x) {
    double result_val = std::sqrt(x.val());
    int result_idx = GPUAADNumber::get_active_tape()->record_unary_op(
        AADOpType::SQRT, x.index(), result_val, 0.5 / result_val);
    return GPUAADNumber(result_idx);
}

GPUAADNumber erf(const GPUAADNumber& x) {
    double x_val = x.val();
    double result_val = std::erf(x_val);
    double derivative = 2.0 / std::sqrt(M_PI) * std::exp(-x_val * x_val);
    int result_idx = GPUAADNumber::get_active_tape()->record_unary_op(
        AADOpType::ERF, x.index(), result_val, derivative);
    return GPUAADNumber(result_idx);
}

GPUAADNumber norm_cdf(const GPUAADNumber& x) {
    return 0.5 * (GPUAADNumber(1.0) + erf(x / sqrt(GPUAADNumber(2.0))));
}
