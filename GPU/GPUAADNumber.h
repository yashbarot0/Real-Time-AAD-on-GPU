// ===== GPUAADNumber.h =====
#pragma once

#include "GPUAADTape.h"

class GPUAADNumber {
private:
    static GPUAADTape* active_tape_;
    int var_index_;
    
    // Helper method for safe tape access
    static GPUAADTape* get_tape_safe();
    
public:
    GPUAADNumber(double val = 0.0);
    GPUAADNumber(int idx);
    
    // Copy constructor and assignment operator
    GPUAADNumber(const GPUAADNumber& other);
    GPUAADNumber& operator=(const GPUAADNumber& other);
    
    static void set_active_tape(GPUAADTape* tape);
    static GPUAADTape* get_active_tape();
    
    double val() const;
    double adj() const;
    int index() const;
    
    // Enhanced arithmetic operators matching CPU implementation
    GPUAADNumber operator+(const GPUAADNumber& other) const;
    GPUAADNumber operator-(const GPUAADNumber& other) const;
    GPUAADNumber operator*(const GPUAADNumber& other) const;
    GPUAADNumber operator/(const GPUAADNumber& other) const;
    GPUAADNumber operator-() const;
    
    // Operators with constants (both sides)
    GPUAADNumber operator+(double rhs) const;
    GPUAADNumber operator-(double rhs) const;
    GPUAADNumber operator*(double rhs) const;
    GPUAADNumber operator/(double rhs) const;
    
    // Friend operators for constants on left side
    friend GPUAADNumber operator+(double lhs, const GPUAADNumber& rhs);
    friend GPUAADNumber operator-(double lhs, const GPUAADNumber& rhs);
    friend GPUAADNumber operator*(double lhs, const GPUAADNumber& rhs);
    friend GPUAADNumber operator/(double lhs, const GPUAADNumber& rhs);
    
    // Comparison operators (for completeness)
    bool operator<(const GPUAADNumber& other) const;
    bool operator>(const GPUAADNumber& other) const;
    bool operator<=(const GPUAADNumber& other) const;
    bool operator>=(const GPUAADNumber& other) const;
    bool operator==(const GPUAADNumber& other) const;
    bool operator!=(const GPUAADNumber& other) const;
    
    // Math functions with enhanced numerical stability
    friend GPUAADNumber log(const GPUAADNumber& x);
    friend GPUAADNumber exp(const GPUAADNumber& x);
    friend GPUAADNumber sqrt(const GPUAADNumber& x);
    friend GPUAADNumber erf(const GPUAADNumber& x);
    friend GPUAADNumber norm_cdf(const GPUAADNumber& x);
    
    // Additional math functions for Black-Scholes
    friend GPUAADNumber pow(const GPUAADNumber& base, double exponent);
    friend GPUAADNumber abs(const GPUAADNumber& x);
    friend GPUAADNumber max(const GPUAADNumber& a, const GPUAADNumber& b);
    friend GPUAADNumber min(const GPUAADNumber& a, const GPUAADNumber& b);
    
    // Safe math functions with numerical stability
    friend GPUAADNumber safe_log(const GPUAADNumber& x);
    friend GPUAADNumber safe_sqrt(const GPUAADNumber& x);
    friend GPUAADNumber safe_divide(const GPUAADNumber& numerator, const GPUAADNumber& denominator);
};
