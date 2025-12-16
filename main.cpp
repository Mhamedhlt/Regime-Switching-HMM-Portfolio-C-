#include "HMM.h"
#include <iostream>

int main() {
    std::cout << "=== STUDENT-T HMM PORTFOLIO MANAGER ===" << std::endl;
    std::cout << "Engin initialisation C++..." << std::endl;

    // 1. Configuration : 3 Régimes (Bull, Bear, Crash), Nu = 4.0
    int num_states = 3; 
    double degrees_of_freedom = 4.0;
    
    HMM model(num_states, degrees_of_freedom);

    // 2. Chargement des données (Générées par ton script Python)
    model.loadData("input_returns.txt");

    // 3. Entraînement (Calibration)
    std::cout << "Start of the EM algorithm (Baum-Welch)..." << std::endl;
    
    // Modification ici : 1000 itérations et tolérance 1e-8 comme discuté
    model.fit(1000, 1e-8); 

    // --- NOUVEL AJOUT : AFFICHAGE DES RÉSULTATS ---
    // On affiche les résultats dans la console pour vérifier la cohérence
    model.printParameters();
    // ----------------------------------------------

    // 4. Décodage et Sauvegarde
    std::cout << "Decoding regimes (Viterbi) ans save..." << std::endl;
    model.saveResults("output_results.csv");

    std::cout << "successfully done." << std::endl;
    return 0;
}