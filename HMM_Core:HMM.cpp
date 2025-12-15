#include "HMM.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <random>
#include <iomanip>
#include <utility> // Pour std::pair

// --- Constructeur ---
HMM::HMM(int num_states, double degrees_of_freedom) : K(num_states), nu(degrees_of_freedom) {
    // Initialisation des vecteurs
    trans_prob.resize(K, std::vector<double>(K));
    means.resize(K);
    vars.resize(K);
    priors.resize(K);
}

// --- Chargement des données ---
void HMM::loadData(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;
    double value;

    if (!file.is_open()) {
        std::cerr << "Critical error : Impossible to open " << filename << std::endl;
        exit(1);
    }

    observations.clear();
    while (std::getline(file, line)) {
        try {
            if (!line.empty()) {
                value = std::stod(line);
                observations.push_back(value);
            }
        } catch (...) { continue; }
    }
    T = observations.size();
    std::cout << "[INFO] Loaded data T = " << T << " observations." << std::endl;
}

// --- Densité Student-t (Formule Eq 3.1) ---
double HMM::getStudentProb(double x, int j) {
    double mu = means[j];
    double sigma = std::sqrt(vars[j]);
    
    // Protection contre sigma trop petit (division par zero)
    if (sigma < 1e-8) sigma = 1e-8;

    double numerator = std::tgamma((nu + 1.0) / 2.0);
    double denominator = std::tgamma(nu / 2.0) * std::sqrt(M_PI * nu) * sigma;
    
    double d = (x - mu) / sigma;
    double core = 1.0 + (d * d) / nu;
    double power = - (nu + 1.0) / 2.0;

    return (numerator / denominator) * std::pow(core, power);
}

// --- Initialisation Intelligente (Quant Style) ---
// On trie les données pour donner des valeurs de départ cohérentes aux régimes
void HMM::initializeParameters() {
    std::vector<double> sorted_obs = observations;
    std::sort(sorted_obs.begin(), sorted_obs.end());
    int chunk_size = T / K;
    
    // Générateur aléatoire pour le Multi-start
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> noise(0.8, 1.2); // Variation de +/- 20%

    for (int i = 0; i < K; ++i) {
        // Transition avec bruit
        double row_sum = 0;
        for (int j = 0; j < K; ++j) {
            trans_prob[i][j] = ((i == j) ? 0.9 : 0.1 / (K - 1)) * noise(gen); 
            row_sum += trans_prob[i][j];
        }
        // Renormalisation
        for(int j=0; j<K; ++j) trans_prob[i][j] /= row_sum;

        priors[i] = 1.0 / K;

        // Moyennes et variances avec bruit
        double sum = 0, sq_sum = 0;
        for (int t = 0; t < chunk_size; ++t) {
            double val = sorted_obs[i * chunk_size + t];
            sum += val;
            sq_sum += val * val;
        }
        
        // On multiplie la moyenne et la variance initiales par un facteur aléatoire
        means[i] = (sum / chunk_size) * noise(gen);
        vars[i] = ((sq_sum / chunk_size) - (means[i] * means[i])) * noise(gen);
        
        if (vars[i] < 1e-6) vars[i] = 1e-6;
    }
}

// --- Algorithme Forward-Backward avec Scaling ---
void HMM::forwardBackward(std::vector<std::vector<double>>& alpha, 
                          std::vector<std::vector<double>>& beta,
                          std::vector<double>& scales) {
    // 1. FORWARD
    scales.assign(T, 0.0);
    
    // t = 0
    for (int i = 0; i < K; ++i) {
        alpha[0][i] = priors[i] * getStudentProb(observations[0], i);
        scales[0] += alpha[0][i];
    }
    // Scaling t=0
    for (int i = 0; i < K; ++i) alpha[0][i] /= scales[0];

    // t = 1 à T-1
    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < K; ++j) {
            double sum_trans = 0.0;
            for (int i = 0; i < K; ++i) {
                sum_trans += alpha[t-1][i] * trans_prob[i][j];
            }
            alpha[t][j] = sum_trans * getStudentProb(observations[t], j);
            scales[t] += alpha[t][j];
        }
        // Scaling
        for (int j = 0; j < K; ++j) alpha[t][j] /= scales[t];
    }

    // 2. BACKWARD
    // t = T-1
    for (int i = 0; i < K; ++i) beta[T-1][i] = 1.0 / scales[T-1]; // scaled by c_T-1

    // t = T-2 à 0
    for (int t = T - 2; t >= 0; --t) {
        for (int i = 0; i < K; ++i) {
            double sum = 0.0;
            for (int j = 0; j < K; ++j) {
                sum += trans_prob[i][j] * getStudentProb(observations[t+1], j) * beta[t+1][j];
            }
            beta[t][i] = sum / scales[t]; // Scaling
        }
    }
}

// --- Algorithme EM (Baum-Welch) ---
void HMM::fit(int max_iter, double tol) {
    initializeParameters();

    std::vector<std::vector<double>> alpha(T, std::vector<double>(K));
    std::vector<std::vector<double>> beta(T, std::vector<double>(K));
    std::vector<std::vector<double>> gamma(T, std::vector<double>(K));
    std::vector<std::vector<std::vector<double>>> xi(T, std::vector<std::vector<double>>(K, std::vector<double>(K)));
    std::vector<double> scales(T);

    double log_likelihood_old = -1e9;

    // --- Boucle EM ---
    for (int iter = 0; iter < max_iter; ++iter) {
        // --- E-STEP ---
        forwardBackward(alpha, beta, scales);

        // Calcul de Gamma (Probas lissées) et Xi (Probas de transition)
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < K; ++i) {
                gamma[t][i] = alpha[t][i] * beta[t][i] * scales[t]; 
            }
        }

        for (int t = 0; t < T - 1; ++t) {
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < K; ++j) {
                    xi[t][i][j] = alpha[t][i] * trans_prob[i][j] * getStudentProb(observations[t+1], j) * beta[t+1][j];
                }
            }
        }

        // --- CALCUL DES POIDS ROBUSTES (u_jt) ---
        // Poids spécifiques Student-t pour réduire l'impact des outliers
        std::vector<std::vector<double>> u(T, std::vector<double>(K));
        for (int t = 0; t < T; ++t) {
            for (int j = 0; j < K; ++j) {
                double diff = observations[t] - means[j];
                double mahalanobis = (diff * diff) / vars[j];
                u[t][j] = (nu + 1.0) / (nu + mahalanobis);
            }
        }

        // --- M-STEP (Mise à jour) ---
        
        // 1. Transition Matrix A
        for (int i = 0; i < K; ++i) {
            double denominator = 0.0;
            for (int t = 0; t < T - 1; ++t) denominator += gamma[t][i];

            for (int j = 0; j < K; ++j) {
                double numerator = 0.0;
                for (int t = 0; t < T - 1; ++t) numerator += xi[t][i][j];
                trans_prob[i][j] = numerator / denominator;
            }
        }

        // 2. Moyennes (mu) et Variances (sigma^2) avec Poids Robustes
        for (int j = 0; j < K; ++j) {
            double num_mu = 0.0, den_mu = 0.0;
            double num_var = 0.0, den_var = 0.0;

            for (int t = 0; t < T; ++t) {
                // Pour la moyenne : pondérée par gamma * u
                double w = gamma[t][j] * u[t][j];
                num_mu += w * observations[t];
                den_mu += w;
            }
            means[j] = num_mu / den_mu;

            for (int t = 0; t < T; ++t) {
                // Pour la variance : numérateur pondéré par gamma * u, dénominateur par gamma
                double diff = observations[t] - means[j];
                num_var += gamma[t][j] * u[t][j] * (diff * diff);
                den_var += gamma[t][j];
            }
            vars[j] = num_var / den_var;
        }

        // --- Convergence Check ---
        double log_likelihood = 0.0;
        for (int t = 0; t < T; ++t) log_likelihood += std::log(scales[t]);
        
        if (iter % 10 == 0 || iter == max_iter - 1) {
             std::cout << "Iteration " << iter + 1 << " | LogLikelihood: " << log_likelihood << std::endl;
        }

        if (std::abs(log_likelihood - log_likelihood_old) < tol) {
            std::cout << "[INFO] Convergence reached after " << iter + 1 << " Iterations " << std::endl;
            break;
        }
        log_likelihood_old = log_likelihood;
    } // Fin boucle EM
    final_log_likelihood = log_likelihood_old;
    // ============================================================
    // ÉTAPE FINALE : TRI DES ÉTATS (Low Vol -> High Vol)
    // ============================================================
    // Ceci garantit que l'Etat 0 = Bull, et l'Etat K-1 = Crash
    
    // 1. On crée des paires (Variance, AncienIndex)
    std::vector<std::pair<double, int>> order;
    for (int k = 0; k < K; ++k) {
        order.push_back({vars[k], k});
    }

    // 2. On trie par variance croissante (État 0 = Plus petite variance)
    std::sort(order.begin(), order.end());

    // 3. On crée des tampons pour stocker les valeurs réordonnées
    std::vector<double> new_means(K);
    std::vector<double> new_vars(K);
    std::vector<double> new_priors(K);
    std::vector<std::vector<double>> new_trans(K, std::vector<double>(K));

    // 4. On remplit les tampons
    for (int new_i = 0; new_i < K; ++new_i) {
        int old_i = order[new_i].second; // Quel était l'ancien index qui va ici ?

        new_means[new_i] = means[old_i];
        new_vars[new_i]  = vars[old_i];
        new_priors[new_i]= priors[old_i];

        for (int new_j = 0; new_j < K; ++new_j) {
            int old_j = order[new_j].second;
            // Attention : il faut permuter lignes ET colonnes de la matrice
            new_trans[new_i][new_j] = trans_prob[old_i][old_j];
        }
    }

    // 5. On remplace les variables membres par les versions triées
    means = new_means;
    vars = new_vars;
    priors = new_priors;
    trans_prob = new_trans;
    
    std::cout << "[INFO] States sorted by growing volatility (0=Low, " << K-1 << "=High)." << std::endl;
}

// --- Viterbi (Decoding) ---
std::vector<int> HMM::predict() {
    std::vector<std::vector<double>> delta(T, std::vector<double>(K));
    std::vector<std::vector<int>> psi(T, std::vector<int>(K));

    // Init (en Log pour éviter underflow)
    for (int i = 0; i < K; ++i) {
        // Protection log(0)
        double p = (priors[i] > 0) ? priors[i] : 1e-10;
        double prob_obs = getStudentProb(observations[0], i);
        if (prob_obs < 1e-300) prob_obs = 1e-300; // Protection underflow

        delta[0][i] = std::log(p) + std::log(prob_obs);
    }

    // Récursion
    for (int t = 1; t < T; ++t) {
        for (int j = 0; j < K; ++j) {
            double max_val = -1e300;
            int max_idx = 0;

            for (int i = 0; i < K; ++i) {
                double trans = (trans_prob[i][j] > 0) ? trans_prob[i][j] : 1e-10;
                double val = delta[t-1][i] + std::log(trans);
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }
            double prob_obs = getStudentProb(observations[t], j);
            if (prob_obs < 1e-300) prob_obs = 1e-300;

            delta[t][j] = max_val + std::log(prob_obs);
            psi[t][j] = max_idx;
        }
    }

    // Backtracking
    std::vector<int> states(T);
    double max_final = -1e300;
    for (int i = 0; i < K; ++i) {
        if (delta[T-1][i] > max_final) {
            max_final = delta[T-1][i];
            states[T-1] = i;
        }
    }

    for (int t = T - 2; t >= 0; --t) {
        states[t] = psi[t+1][states[t+1]];
    }

    return states;
}

// --- Sauvegarde des Résultats ---
void HMM::saveResults(const std::string& filename) {
    std::ofstream file(filename);
    file << "Observation,Regim_Viterbi,Mean_Regim,Vol_Regim\n";
    
    std::vector<int> states = predict();

    for (int t = 0; t < T; ++t) {
        int s = states[t];
        file << observations[t] << "," 
             << s << "," 
             << means[s] << "," 
             << std::sqrt(vars[s]) << "\n";
    }
    file.close();
    std::cout << "[INFO] Results saved in " << filename << std::endl;
}