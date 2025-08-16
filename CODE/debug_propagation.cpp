#include <iostream>
#include <iomanip>
#include "AADNumber.h"

// Add some debug prints to the AADNode class temporarily
void debug_node_info(const AADNumber& num, const std::string& name) {
    std::cout << name << ": val=" << num.val() << ", adj=" << num.adj() 
              << ", deps=" << num.node->dependencies.size() 
              << ", visited=" << num.node->visited << std::endl;
    
    for (size_t i = 0; i < num.node->dependencies.size(); ++i) {
        auto& dep = num.node->dependencies[i];
        std::cout << "  dep[" << i << "]: partial=" << dep.second 
                  << ", target_adj=" << dep.first->adj << std::endl;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Debug Propagation Mechanism ===" << std::endl;
    
    // Test 1: Very simple case
    std::cout << "\n--- Test 1: Simple x + y ---" << std::endl;
    AADNumber x(3.0);
    AADNumber y(4.0);
    AADNumber z = x + y;
    
    std::cout << "Before propagation:" << std::endl;
    debug_node_info(x, "x");
    debug_node_info(y, "y");
    debug_node_info(z, "z");
    
    z.setAdj(1.0);
    std::cout << "\nAfter setting z.adj=1.0:" << std::endl;
    debug_node_info(z, "z");
    
    z.propagate();
    std::cout << "\nAfter propagation:" << std::endl;
    debug_node_info(x, "x");
    debug_node_info(y, "y");
    debug_node_info(z, "z");
    
    // Test 2: Chain of operations
    std::cout << "\n--- Test 2: Chain a * b + c ---" << std::endl;
    AADNumber a(2.0);
    AADNumber b(3.0);
    AADNumber c(1.0);
    AADNumber temp = a * b;  // temp = 6
    AADNumber result = temp + c;  // result = 7
    
    std::cout << "Before propagation:" << std::endl;
    debug_node_info(a, "a");
    debug_node_info(b, "b");
    debug_node_info(c, "c");
    debug_node_info(temp, "temp");
    debug_node_info(result, "result");
    
    result.setAdj(1.0);
    result.propagate();
    
    std::cout << "\nAfter propagation:" << std::endl;
    debug_node_info(a, "a");
    debug_node_info(b, "b");
    debug_node_info(c, "c");
    debug_node_info(temp, "temp");
    debug_node_info(result, "result");
    
    return 0;
}