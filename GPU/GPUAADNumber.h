// ===== GPUAADNumber.h =====
#pragma once

#include "GPUAADTape.h"

class GPUAADNumber {
private:
    static GPUAADTape* active_tape_;
    int var_index_;
    
public:
    GPUAADNumber(double val = 0.0);
    GPUAADNumber(int idx);
    
    static void set_active_tape(GPUAADTape* tape);
    static GPUAADTape* get_active_tape();
    
    double val() const;
    double adj() const;
    int index() const;
    
    // Arithmetic operators
    GPUAADNumber operator+(const GPUAADNumber& other) const;
    GPUAADNumber operator-(const GPUAADNumber& other) const;
    GPUAADNumber operator*(const GPUAADNumber& other) const;
    GPUAADNumber operator/(const GPUAADNumber& other) const;
    GPUAADNumber operator-() const;
    
    // Friend operators for constants
    friend GPUAADNumber operator+(double lhs, const GPUAADNumber& rhs);
    friend GPUAADNumber operator*(double lhs, const GPUAADNumber& rhs);
    
    // Math functions
    friend GPUAADNumber log(const GPUAADNumber& x);
    friend GPUAADNumber exp(const GPUAADNumber& x);
    friend GPUAADNumber sqrt(const GPUAADNumber& x);
    friend GPUAADNumber erf(const GPUAADNumber& x);
    friend GPUAADNumber norm_cdf(const GPUAADNumber& x);
};
