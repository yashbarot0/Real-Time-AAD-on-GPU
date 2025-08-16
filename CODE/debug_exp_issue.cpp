#include <iostream>
#include <iomanip>
#include "AADNumber.h"

void debug_node_info(const AADNumber& num, const std::string& name) {
    std::cout << name << ": val=" << num.val() << ", adj=" << num.adj() 
              << ", deps=" << num.node->dependencies.size() << std::endl;
}

int main() {
    std::cout << std::fixed << std::setprecision(6);
    
    std::cout << "=== Debugging exp(-r*T) Issue ===" << std::endl;
    
    // Test 1: Simple exp(x)
    std::cout << "\n--- Test 1: Simple exp(x) ---" << std::endl;
    AADNumber x1(0.5);
    AADNumber result1 = exp(x1);
    
    debug_node_info(x1, "x1");
    debug_node_info(result1, "result1");
    
    result1.setAdj(1.0);
    result1.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "x1.adj() = " << x1.adj() << " (should be exp(0.5) = " << std::exp(0.5) << ")" << std::endl;
    
    // Test 2: exp(-x)
    std::cout << "\n--- Test 2: exp(-x) ---" << std::endl;
    AADNumber x2(0.5);
    AADNumber neg_x2 = -x2;
    AADNumber result2 = exp(neg_x2);
    
    debug_node_info(x2, "x2");
    debug_node_info(neg_x2, "neg_x2");
    debug_node_info(result2, "result2");
    
    result2.setAdj(1.0);
    result2.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "x2.adj() = " << x2.adj() << " (should be -exp(-0.5) = " << (-std::exp(-0.5)) << ")" << std::endl;
    
    // Test 3: r * T
    std::cout << "\n--- Test 3: r * T ---" << std::endl;
    AADNumber r3(0.05);
    AADNumber T3(0.25);
    AADNumber product = r3 * T3;
    
    debug_node_info(r3, "r3");
    debug_node_info(T3, "T3");
    debug_node_info(product, "product");
    
    product.setAdj(1.0);
    product.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "r3.adj() = " << r3.adj() << " (should be T = " << T3.val() << ")" << std::endl;
    std::cout << "T3.adj() = " << T3.adj() << " (should be r = " << r3.val() << ")" << std::endl;
    
    // Test 4: -r * T
    std::cout << "\n--- Test 4: -r * T ---" << std::endl;
    AADNumber r4(0.05);
    AADNumber T4(0.25);
    AADNumber neg_product = -(r4 * T4);
    
    debug_node_info(r4, "r4");
    debug_node_info(T4, "T4");
    debug_node_info(neg_product, "neg_product");
    
    neg_product.setAdj(1.0);
    neg_product.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "r4.adj() = " << r4.adj() << " (should be -T = " << (-T4.val()) << ")" << std::endl;
    std::cout << "T4.adj() = " << T4.adj() << " (should be -r = " << (-r4.val()) << ")" << std::endl;
    
    // Test 5: exp(-r * T) step by step
    std::cout << "\n--- Test 5: exp(-r * T) step by step ---" << std::endl;
    AADNumber r5(0.05);
    AADNumber T5(0.25);
    AADNumber product5 = r5 * T5;
    AADNumber neg_product5 = -product5;
    AADNumber result5 = exp(neg_product5);
    
    std::cout << "Before propagation:" << std::endl;
    debug_node_info(r5, "r5");
    debug_node_info(T5, "T5");
    debug_node_info(product5, "product5");
    debug_node_info(neg_product5, "neg_product5");
    debug_node_info(result5, "result5");
    
    result5.setAdj(1.0);
    result5.propagate();
    
    std::cout << "After propagation:" << std::endl;
    std::cout << "r5.adj() = " << r5.adj() << std::endl;
    std::cout << "T5.adj() = " << T5.adj() << std::endl;
    
    double expected_r_adj = -T5.val() * result5.val();
    double expected_T_adj = -r5.val() * result5.val();
    std::cout << "Expected r5.adj() = " << expected_r_adj << std::endl;
    std::cout << "Expected T5.adj() = " << expected_T_adj << std::endl;
    
    return 0;
}