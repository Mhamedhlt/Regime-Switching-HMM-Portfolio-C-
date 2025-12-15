# Regime-Switching-HMM-Portfolio-C-
Quantitative strategy for dynamic asset allocation using Student-t Hidden Markov Models (HMM) implemented in C++ and Python.
# 📉 Regime-Switching, Tail-Risk and Stress-Aware Portfolio

## Dynamic Asset Allocation using Student-t Hidden Markov Models (HMM)

This project develops an adaptive quantitative investment strategy designed to preserve capital and generate superior risk-adjusted returns by dynamically switching portfolio exposure based on detected market regimes (Bull, Bear, Crash).

The strategy is a hybrid implementation: the high-performance core is written in C++ for numerical stability and speed, while the backtesting and visualization are handled in Python.

---

## 🚀 Key Contributions and Results

### 1. Robust Regime Identification
* **Model:** Student-t Hidden Markov Model (HMM) with 3 states (Bull, Bear, Crash). The Student-t distribution is used to capture **heavy tails** and volatility clustering inherent in financial returns.
* **Core Engine:** The HMM estimation (Baum-Welch algorithm) is implemented in **C++** to handle large time series and ensure rigorous numerical stability.
* **Validation:** The model successfully reconstructs historical crises, identifying the 2008 collapse as a prolonged **Crash** regime and the 2020 shock as an abrupt, fast-recovering **Crash**.

### 2. Strategy and Implementation
* **Strategy:** "Bang-Bang" switching rule: **100% S&P 500** in Bull/Bear regimes, and **100% Risk-Free Asset (Cash)** in the Crash regime.
* **Backtesting Period:** 1985 to 2023 (S&P 500).
* **Architecture:** C++ for the HMM solver, Python for data pre-processing and dynamic backtesting.

### 3. Financial Performance Highlights (1985-2023)

| Metric | S&P 500 (Benchmark) | HMM Strategy | Improvement |
| :--- | :--- | :--- | :--- |
| **Sharpe Ratio** | 0.47 | **0.89** | **+89%** |
| Max Drawdown | -56.77% | **-35.60%** | **Capital Preservation** |
| Annualized Volatility | 18.40% | **14.16%** | **Smoother Returns** |
| Total Return | 2233% (22x) | **9227% (92x)** | |

---

## 🔍 Stress Test Validation (VaR and Crisis Zoom)
* **Tail-Risk:** Monte Carlo simulations confirmed that the strategy structurally **cuts the left tail** of the loss distribution, resulting in a significantly lower Value-at-Risk (VaR 95%) compared to the Buy & Hold benchmark.
* **2008 Crisis:** The strategy switched to Cash in late 2008, preserving capital while the market plunged (Rebased Value).

---

## 📚 Repository Structure

* `Finance_math.pdf`: The complete 42-page research report.
* `HMM_Core/`: C++ source files (`.cpp`, `.h`) for the Student-t HMM estimation engine.
* `Python_Backtester/`: Python scripts (`.py`) for data handling, strategy execution, and plotting performance.

---

## ⚙️ Technologies Used

* **C++:** Core HMM estimation (Baum-Welch, Viterbi, Forward-Backward algorithms).
* **Python:** Data analysis (Pandas, NumPy), Financial data fetching (yfinance), and Backtesting.

**See the full research report for mathematical formulations and economic analysis.**
