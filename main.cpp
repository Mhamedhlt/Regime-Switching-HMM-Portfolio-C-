#include "HMM.h"
#include <iostream>

int main() {
    std::cout << "=== STUDENT-T HMM PORTFOLIO MANAGER ===" << std::endl;
    std::cout << "Engin initialisation C++..." << std::endl;

  
    int num_states = 3; 
    double degrees_of_freedom = 4.0;
    
    HMM model(num_states, degrees_of_freedom);


    model.loadData("input_returns.txt");

    
    std::cout << "Start of the EM algorithm (Baum-Welch)..." << std::endl;
    
  
    model.fit(1000, 1e-8); 


    model.printParameters();
    // ----------------------------------------------

  
    std::cout << "Decoding regimes (Viterbi) ans save..." << std::endl;
    model.saveResults("output_results.csv");

    std::cout << "successfully done." << std::endl;
    return 0;
}
