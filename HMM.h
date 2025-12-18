#ifndef HMM_H
#define HMM_H

#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <iostream> 

class HMM {
private:
    
    int K; 
    double nu; 

   
    std::vector<std::vector<double>> trans_prob;
    
    
    std::vector<double> means;  
    std::vector<double> vars;   
    std::vector<double> priors; 


    std::vector<double> observations; 
    int T; // Nombre d'observations

    double final_log_likelihood = -1e9;
public:
   
    HMM(int num_states, double degrees_of_freedom);

    
    void loadData(const std::string& filename);
    void saveResults(const std::string& filename);

   
   
    double getStudentProb(double observation, int state_idx);

   
    void fit(int max_iter = 100, double tol = 1e-6); 
    std::vector<int> predict(); 
    

    void printParameters() const { 
        std::cout << "\n==========================================" << std::endl;
        std::cout << "   Calibration Result (Student-t)" << std::endl;
        std::cout << "==========================================" << std::endl;

       
        std::cout << "\n[Emission parameters]" << std::endl;
        std::cout << std::left << std::setw(10) << "State" 
                  << std::setw(15) << "Mu (mean)" 
                  << std::setw(15) << "Sigma (Vol)" 
                  << std::setw(15) << "Nu (freed.Lib)" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;

        for (int k = 0; k < K; ++k) {
           
            
            std::cout << std::left << std::setw(10) << ("State " + std::to_string(k))
                      << std::setw(15) << std::fixed << std::setprecision(6) << means[k]
                      << std::setw(15) << std::sqrt(vars[k]) 
                      << std::setw(15) << nu << std::endl; 
        }

       
        std::cout << "\n[Transition Matrix (Probabilities)]" << std::endl;
        std::cout << "      ";
        for(int k=0; k<K; ++k) std::cout << "Towards S" << k << "   ";
        std::cout << std::endl;

        for (int i = 0; i < K; ++i) {
            std::cout << "from S" << i << " : ";
            for (int j = 0; j < K; ++j) {
                // Utilisation de trans_prob au lieu de A
                std::cout << std::fixed << std::setprecision(4) << trans_prob[i][j] << "   "; 
            }
            std::cout << std::endl;
        }
        std::cout << "==========================================\n" << std::endl;
    }
    double getLogLikelihood() const { return final_log_likelihood; }
private:
   
    void initializeParameters(); 

   
    void forwardBackward(std::vector<std::vector<double>>& alpha, 
                         std::vector<std::vector<double>>& beta,
                         std::vector<double>& scales);
};

#endif
