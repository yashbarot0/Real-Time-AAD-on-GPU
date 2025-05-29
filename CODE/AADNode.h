// AADNode.h
#pragma once

#include <vector>
#include <functional>

struct AADNode {
    double val = 0.0;                         // Primal value
    double adj = 0.0;                         // Adjoint value
    std::vector<std::pair<AADNode*, double>> dependencies;  // Dependencies and local partials

    void propagate() {
        for (auto& dep : dependencies) {
            dep.first->adj += dep.second * adj;
        }
    }
};
