#include <iostream>
#include <iomanip>
#include "AADNumber.h"

// Add debug prints to trace propagation
struct DebugAADNode {
    double val = 0.0;
    double adj = 0.0;
    std::vector<std::pair<DebugAADNode*, double>> dependencies;
    bool visited = false;
    std::string name;  // For debugging
    
    DebugAADNode(double v = 0.0, const std::string& n = "") : val(v), name(n) {}
    
    void propagate() {
        std::cout << "Starting propagation from " << name << " (adj=" << adj << ")" << std::endl;
        reset_visited_recursive();
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
        if (visited) {
            std::cout << "  " << name << " already visited, skipping" << std::endl;
            return;
        }
        visited = true;
        
        std::cout << "  Propagating through " << name << " (adj=" << adj << ", deps=" << dependencies.size() << ")" << std::endl;
        
        for (auto& dep : dependencies) {
            double contribution = dep.second * adj;
            std::cout << "    Adding " << contribution << " to " << dep.first->name << " (was " << dep.first->adj << ")" << std::endl;
            dep.first->adj += contribution;
            dep.first->propagate_internal();
        }
    }
};

class DebugAADNumber {
public:
    std::shared_ptr<DebugAADNode> node;

    DebugAADNumber(double val = 0.0, const std::string& name = "") {
        node = std::make_shared<DebugAADNode>(val, name);
    }

    double val() const { return node->val; }
    double adj() const { return node->adj; }
    void setAdj(double a) { node->adj = a; }
    void propagate() { node->propagate(); }

    DebugAADNumber operator*(const DebugAADNumber& other) const {
        DebugAADNumber res(node->val * other.node->val, "(" + node->name + "*" + other.node->name + ")");
        res.node->dependencies = {{node.get(), other.node->val}, {other.node.get(), node->val}};
        return res;
    }
};

int main() {
    std::cout << "=== Tracing Simple Multiplication ===" << std::endl;
    
    DebugAADNumber x(3.0, "x");
    DebugAADNumber y(4.0, "y");
    
    std::cout << "Created x=" << x.val() << ", y=" << y.val() << std::endl;
    
    DebugAADNumber z = x * y;
    std::cout << "z = x * y = " << z.val() << std::endl;
    
    z.setAdj(1.0);
    std::cout << "Set z.adj() = 1.0" << std::endl;
    
    z.propagate();
    
    std::cout << "\nFinal results:" << std::endl;
    std::cout << "x.adj() = " << x.adj() << std::endl;
    std::cout << "y.adj() = " << y.adj() << std::endl;
    
    return 0;
}