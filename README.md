# AI Portfolio Optimization with NSGA-II

This repository contains a complete, end-to-end implementation of a multi-objective portfolio optimization framework using the Non‑dominated Sorting Genetic Algorithm II (NSGA‑II). It focuses entirely on financial portfolio construction; there is currently no implemented project for green hydrogen or Differential Evolution in this repository.

---

## Overview

The goal of this project is to construct equity portfolios that balance two competing objectives:

- **Maximize expected portfolio return**
- **Minimize portfolio risk**, measured by variance

Using NSGA‑II, the project generates a Pareto front of optimal portfolios where improving one objective necessarily worsens the other, giving investors a spectrum of efficient trade‑offs between risk and return.

The dataset consists of daily price data (April 17, 2024 to April 17, 2025) for five large‑cap Indian equities from the Nifty 50 index.

---

## Data and Universe

The portfolio universe includes the following stocks, with price histories stored as CSV files in the repository:

| Ticker      | Sector          |
|------------|-----------------|
| TATASTEEL  | Metals          |
| TITAN      | Consumer Goods  |
| AXISBANK   | Banking         |
| HDFCBANK   | Banking         |
| BHARTIARTL | Telecom         |

From these series, the scripts compute daily returns, annualized expected returns, and the covariance matrix used in the optimization.

---

## NSGA‑II Optimization

The NSGA‑II implementation searches over portfolio weight vectors subject to realistic constraints (such as full investment and no short selling) and evaluates each candidate by:

- Computing the **expected portfolio return**
- Computing the **portfolio variance (risk)**

Key NSGA‑II steps implemented in the code:

1. **Initialization** – Randomly generate a population of feasible portfolios (weight vectors).
2. **Fitness evaluation** – Calculate return and risk for each portfolio.
3. **Non‑dominated sorting** – Rank portfolios into Pareto fronts based on dominance.
4. **Crowding distance calculation** – Maintain diversity along the front.
5. **Selection, crossover, and mutation** – Create offspring portfolios and evolve the population.
6. **Elitism** – Combine parent and offspring populations and retain the best individuals.

The end result is an approximation of the efficient frontier in risk‑return space.

---

## Results and Visualizations

### Pareto Front

The file `pareto_front.png` shows the final Pareto front obtained from the NSGA‑II run. Each point corresponds to a non‑dominated portfolio characterized by its risk (x‑axis) and expected return (y‑axis).

![Pareto Front](pareto_front.png)

### Balanced Portfolio Allocation

The chart `balanced_portfolio_pie.png` visualizes the asset weights of a selected “balanced” portfolio, typically chosen to maximize a risk‑adjusted performance metric such as the Sharpe ratio.

![Balanced Portfolio](balanced_portfolio_pie.png)

### Example Portfolio Types

From the Pareto set, the analysis highlights:

1. **Minimum Risk Portfolio** – Lowest volatility, conservative allocation.
2. **Maximum Return Portfolio** – Highest expected return, aggressive allocation.
3. **Balanced Portfolio** – Best compromise between risk and return (highest Sharpe ratio in the study).

These are explained in more detail in the accompanying report files.

---

## How to Run the Optimization

### 1. Install Dependencies

```bash
pip install numpy pandas matplotlib deap
```

### 2. Execute the Main Script

```bash
python nsga_ii_portfolio_final.py
```

This will:

- Load the historical price data from the CSV files.
- Run the NSGA‑II optimization loop.
- Generate and save plots for the Pareto front and the chosen portfolio allocation.

---

## Repository Contents

| File | Description |
|------|-------------|
| `nsga_ii_portfolio_final.py` | Main, fully featured NSGA‑II implementation for portfolio optimization. |
| `nsga_ii_portfolio_fix.py`   | Refined/cleaned version of an earlier implementation. |
| `nsga_ii_portfolio.py`       | Basic or experimental NSGA‑II version. |
| `Portfolio_Optimization_NSGA_II.md` | Theory and algorithm documentation for NSGA‑II in a portfolio context. |
| `NSGA_II_FinancialReport.md` | Detailed financial analysis and interpretation of results. |
| `pareto_front.png`           | Risk–return Pareto front visualization. |
| `balanced_portfolio_pie.png` | Pie chart of the selected balanced portfolio weights. |
| `Quote-Equity-*.csv`         | Historical price data for each stock. |

There is also a research paper PDF (`AIDA2_merged.pdf`) included as background reading on related optimization topics, but there is no implemented Differential Evolution/green hydrogen code in this repository.

---

## Project Structure

```text
AI_proj/
├── nsga_ii_portfolio_final.py
├── nsga_ii_portfolio_fix.py
├── nsga_ii_portfolio.py
├── NSGA_II_FinancialReport.md
├── Portfolio_Optimization_NSGA_II.md
├── pareto_front.png
├── balanced_portfolio_pie.png
├── Quote-Equity-AXISBANK-*.csv
├── Quote-Equity-BHARTIARTL-*.csv
├── Quote-Equity-HDFCBANK-*.csv
├── Quote-Equity-TATASTEEL-*.csv
├── Quote-Equity-TITAN-*.csv
└── README.md
```

---

## References

1. Markowitz, H. (1952). Portfolio Selection. The Journal of Finance.
2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A Fast and Elitist Multiobjective Genetic Algorithm: NSGA‑II. IEEE Transactions on Evolutionary Computation.

---

## License

This project is provided for educational and research purposes only.
