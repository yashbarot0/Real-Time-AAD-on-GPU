// AADNumber.h
#pragma once

#include "AADNode.h"
#include <memory>
#include <cmath>

class AADNumber {
public:
    std::shared_ptr<AADNode> node;

    AADNumber(double val = 0.0) {
        node = std::make_shared<AADNode>();
        node->val = val;
    }

    double val() const { return node->val; }
    double adj() const { return node->adj; }

    void setAdj(double a) { node->adj = a; }
    void propagate() { node->propagate(); }

    // Operator overloads
    AADNumber operator+(const AADNumber& other) const {
        AADNumber res(node->val + other.node->val);
        res.node->dependencies = {{node.get(), 1.0}, {other.node.get(), 1.0}};
        return res;
    }

    friend AADNumber operator*(double lhs, const AADNumber& rhs) {
    AADNumber res(lhs * rhs.node->val);
    res.node->dependencies = {{rhs.node.get(), lhs}};
    return res;
}

AADNumber operator-() const {
    AADNumber res(-node->val);
    res.node->dependencies = {{node.get(), -1.0}};
    return res;
}


friend AADNumber operator+(double lhs, const AADNumber& rhs) {
    AADNumber res(lhs + rhs.node->val);
    res.node->dependencies = {{rhs.node.get(), 1.0}};
    return res;
}


    AADNumber operator-(const AADNumber& other) const {
        AADNumber res(node->val - other.node->val);
        res.node->dependencies = {{node.get(), 1.0}, {other.node.get(), -1.0}};
        return res;
    }

    AADNumber operator*(const AADNumber& other) const {
        AADNumber res(node->val * other.node->val);
        res.node->dependencies = {{node.get(), other.node->val}, {other.node.get(), node->val}};
        return res;
    }

    AADNumber operator/(const AADNumber& other) const {
        AADNumber res(node->val / other.node->val);
        res.node->dependencies = {
            {node.get(), 1.0 / other.node->val},
            {other.node.get(), -node->val / (other.node->val * other.node->val)}
        };
        return res;
    }

    friend AADNumber log(const AADNumber& x) {
        AADNumber res(std::log(x.node->val));
        res.node->dependencies = {{x.node.get(), 1.0 / x.node->val}};
        return res;
    }

    friend AADNumber exp(const AADNumber& x) {
        AADNumber res(std::exp(x.node->val));
        res.node->dependencies = {{x.node.get(), res.node->val}};
        return res;
    }

    friend AADNumber sqrt(const AADNumber& x) {
        AADNumber res(std::sqrt(x.node->val));
        res.node->dependencies = {{x.node.get(), 0.5 / res.node->val}};
        return res;
    }

    friend AADNumber erf(const AADNumber& x) {
        double erf_val = std::erf(x.node->val);
        double d_erf = 2.0 / std::sqrt(M_PI) * std::exp(-x.node->val * x.node->val);
        AADNumber res(erf_val);
        res.node->dependencies = {{x.node.get(), d_erf}};
        return res;
    }

    friend AADNumber norm_cdf(const AADNumber& x) {
        return 0.5 * (AADNumber(1.0) + erf(x / sqrt(AADNumber(2.0))));
    }
};