#ifndef HMM_H
#define HMM_H

#include <vector>
#include <string>
#include <cmath>
#include <iomanip>
#include <iostream> // <--- Indispensable pour std::cout

class HMM {
private:
    // --- Paramètres du Modèle ---
    int K; // Nombre de régimes (ex: 2 ou 3)
    double nu; // Degrés de liberté (fixé à 4.0 par défaut)

    // Matrice de Transition A [K][K]
    std::vector<std::vector<double>> trans_prob;
    
    // Paramètres d'émission (Student-t)
    std::vector<double> means;  // mu
    std::vector<double> vars;   // sigma^2 (Variance)
    std::vector<double> priors; // pi (probabilité initiale)

    // --- Données ---
    std::vector<double> observations; // Les rendements lus du fichier
    int T; // Nombre d'observations

    double final_log_likelihood = -1e9;
public:
    // Constructeur
    HMM(int num_states, double degrees_of_freedom);

    // 1. Gestion des I/O
    void loadData(const std::string& filename);
    void saveResults(const std::string& filename);

    // 2. Le Cœur Mathématique
    // Calcule la densité Student b_j(r_t)
    double getStudentProb(double observation, int state_idx);

    // 3. Algorithmes
    void fit(int max_iter = 100, double tol = 1e-6); // Baum-Welch (EM)
    std::vector<int> predict(); // Viterbi (Decode)
    
    // 4. Affichage (Adapté à tes variables)
    void printParameters() const { // 'const' ajouté car ça ne modifie pas les données
        std::cout << "\n==========================================" << std::endl;
        std::cout << "   Calibration Result (Student-t)" << std::endl;
        std::cout << "==========================================" << std::endl;

        // 1. Affichage des paramètres d'émission
        std::cout << "\n[Emission parameters]" << std::endl;
        std::cout << std::left << std::setw(10) << "State" 
                  << std::setw(15) << "Mu (mean)" 
                  << std::setw(15) << "Sigma (Vol)" 
                  << std::setw(15) << "Nu (freed.Lib)" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;

        for (int k = 0; k < K; ++k) {
            // Note: L'ordre dépend du résultat de l'entraînement.
            // On affiche std::sqrt(vars[k]) car 'vars' contient la variance (sigma^2)
            // et on veut afficher l'écart-type (sigma/volatilité).
            
            std::cout << std::left << std::setw(10) << ("State " + std::to_string(k))
                      << std::setw(15) << std::fixed << std::setprecision(6) << means[k]
                      << std::setw(15) << std::sqrt(vars[k]) 
                      << std::setw(15) << nu << std::endl; 
        }

        // 2. Affichage de la Matrice de Transition
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
    // Méthodes internes pour EM
    void initializeParameters(); 

    // Forward-Backward renvoie alpha et beta
    void forwardBackward(std::vector<std::vector<double>>& alpha, 
                         std::vector<std::vector<double>>& beta,
                         std::vector<double>& scales);
};

#endif