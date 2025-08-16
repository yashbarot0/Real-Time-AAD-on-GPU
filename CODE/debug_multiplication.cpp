#include <iostream>
#include <iomanip>
#include "AADNumber.h"

void debug_detailed(const AADNumber& num, const std::string& name) {
    std::cout << name << ": val=" << num.val() << ", adj=" << num.adj() 
              << ", deps=" << num.node->dependencies.size() 
              << ", ptr=" << num.node.get() << std::endl;
    
    for (size_t i = 0; i < num.node->dependencies.size(); ++i) {
        auto& dep = num.node->dependencies[i];
        std::cout << "  dep[" << i << "]: partial=" << dep.second 
                  << ", target_ptr=" << dep.first
                  << ", target_adj=" << dep.first->adj << std::endl;
    }
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Debugging Multiplication Dependencies ===" << std::endl;
    
    AADNumber a(3.0);
    AADNumber b(4.0);
    
    std::cout << "\nBefore multiplication:" << std::endl;
    debug_detailed(a, "a");
    debug_detailed(b, "b");
    
    AADNumber c = a * b;
    
    std::cout << "\nAfter c = a * b:" << std::endl;
    debug_detailed(a, "a");
    debug_detailed(b, "b");
    debug_detailed(c, "c");
    
    // Check if the pointers in c's dependencies match a and b
    std::cout << "\nPointer verification:" << std::endl;
    std::cout << "a.node.get() = " << a.node.get() << std::endl;
    std::cout << "b.node.get() = " << b.node.get() << std::endl;
    if (c.node->dependencies.size() >= 2) {
        std::cout << "c.dep[0].first = " << c.node->dependencies[0].first << std::endl;
        std::cout << "c.dep[1].first = " << c.node->dependencies[1].first << std::endl;
        std::cout << "Pointer match a: " << (c.node->dependencies[0].first == a.node.get()) << std::endl;
        std::cout << "Pointer match b: " << (c.node->dependencies[1].first == b.node.get()) << std::endl;
    }
    
    c.setAdj(1.0);
    std::cout << "\nAfter setting c.adj = 1.0:" << std::endl;
    debug_detailed(c, "c");
    
    c.propagate();
    std::cout << "\nAfter propagation:" << std::endl;
    debug_detailed(a, "a");
    debug_detailed(b, "b");
    debug_detailed(c, "c");
    
    return 0;
}