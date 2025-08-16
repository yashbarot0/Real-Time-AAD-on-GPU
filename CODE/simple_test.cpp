#include <iostream>
#include "AADNumber.h"

int main() {
    // Simple test: f(x) = x * 2
    AADNumber x(5.0);
    AADNumber result = x * 2.0;
    
    std::cout << "Before propagation:" << std::endl;
    std::cout << "x.val() = " << x.val() << ", x.adj() = " << x.adj() << std::endl;
    std::cout << "result.val() = " << result.val() << ", result.adj() = " << result.adj() << std::endl;
    
    result.setAdj(1.0);
    result.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "x.adj() = " << x.adj() << " (should be 2.0)" << std::endl;
    
    return 0;
}