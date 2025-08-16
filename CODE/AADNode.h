// AADNode.h
#pragma once

#include <vector>
#include <functional>

struct AADNode {
    double val = 0.0;                         // Primal value
    double adj = 0.0;                         // Adjoint value
    std::vector<std::pair<AADNode*, double>> dependencies;  // Dependencies and local partials
    bool visited = false;                     // For topological sort

    void propagate() {
        // Reset visited flags first
        reset_visited_recursive();
        
        // Now do the actual propagation
        propagate_internal();
    }
    
private:
    void reset_visited_recursive() {
        if (!visited) return;
        visited = false;
        for (auto& dep : dependencies) {
            dep.first->reset_visited_recursive();
        }
    }
    
    void propagate_internal() {
        if (visited) return;
        visited = true;
        
        for (auto& dep : dependencies) {
            dep.first->adj += dep.second * adj;
            dep.first->propagate_internal();
        }
    }
    
public:
    
    void reset_visited() {
        if (!visited) return;  // Already reset
        visited = false;
        for (auto& dep : dependencies) {
            dep.first->reset_visited();
        }
    }
    
    void reset_adjoints() {
        adj = 0.0;
        visited = false;
        for (auto& dep : dependencies) {
            dep.first->reset_adjoints();
        }
    }
};
