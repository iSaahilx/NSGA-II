# Pareto Optimization for Portfolio Management using NSGA-II

This repository contains a complete, end-to-end implementation of **multi-objective portfolio optimization** using the **Non-dominated Sorting Genetic Algorithm II (NSGA-II)**. The project demonstrates how evolutionary algorithms can construct optimal portfolios that balance **expected return** and **risk (variance)** for a set of large-cap Indian equities.

The implementation, analysis, and results are documented in detail in the report  
`NSGA_II_FinancialReport.md` and its compiled version `NSGA_II_FinancialReport.pdf`  
([PDF version](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf)).

---

## Contents of this Repository

- `nsga_ii_portfolio_final.py` – Main NSGA-II implementation (DEAP-based).
- `Portfolio_Optimization_NSGA_II.md` – Theory and algorithm write-up.
- `NSGA_II_FinancialReport.md` / `NSGA_II_FinancialReport.pdf` – Full academic-style report with methodology, experiments, and analysis.
- `pareto_front.png` – Final Pareto front (efficient frontier) of risk vs. return.
- `balanced_portfolio_pie.png` – Asset allocation of the balanced (highest Sharpe) portfolio.
- `Equity_data/` – Historical price data for each stock (CSV).

---

## Problem Overview

The portfolio consists of **five Nifty 50 stocks**:

- Tata Steel (TATASTEEL)  
- Titan Company (TITAN)  
- Axis Bank (AXISBANK)  
- HDFC Bank (HDFCBANK)  
- Bharti Airtel (BHARTIARTL)

Using **one year of daily closing prices** (17 April 2024 – 17 April 2025), the goal is to determine asset weights \( w_i \) that:

- **Maximize** expected portfolio return \( E(R_p) \)
- **Minimize** portfolio risk \( \sigma_p^2 \) (variance)

subject to:

\[
\sum_{i=1}^{n} w_i = 1, \quad w_i \ge 0 \ \forall i
\]

with \( n = 5 \) assets and **no short-selling** allowed.

---

## Mathematical Formulation

### Objectives

**1. Maximize expected portfolio return**

\[
E(R_p) = \sum_{i=1}^{n} w_i \, E(R_i)
\]

where:

- \( E(R_p) \) – expected portfolio return  
- \( w_i \) – portfolio weight of asset \( i \)  
- \( E(R_i) \) – expected return of asset \( i \) (mean of historical daily returns)

**2. Minimize portfolio variance (risk)**

\[
\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i \, w_j \, \sigma_{ij}
\]

where:

- \( \sigma_p^2 \) – variance of portfolio returns  
- \( \sigma_{ij} \) – covariance between returns of assets \( i \) and \( j \)

### Constraints

- **Full investment:**  
  \[
  \sum_{i=1}^{n} w_i = 1
  \]
- **Non-negativity (no short selling):**  
  \[
  w_i \ge 0, \quad i=1,\dots,n
  \]

The problem is treated as a **bi-objective optimization**:  
maximize \( f_1 = E(R_p) \) and minimize \( f_2 = \sigma_p^2 \).

---

## NSGA-II Approach

The project implements NSGA-II using the **DEAP** library, following the classic steps:

1. **Initialization** – Randomly generate a population of portfolios (vectors of raw weights), then normalize them to satisfy the constraints.
2. **Fitness Evaluation** – For each portfolio:
   - compute \( E(R_p) \) from historical returns  
   - compute \( \sigma_p^2 \) from the covariance matrix  
3. **Non-dominated Sorting** – Rank individuals into Pareto fronts (Front 1 = non-dominated, Front 2 dominated only by Front 1, etc.).
4. **Crowding Distance** – Within each front, estimate density in objective space to preserve diversity:

   \[
   d_i = \sum_{m=1}^{M} \frac{f_m^{(i+1)} - f_m^{(i-1)}}{f_m^{\max} - f_m^{\min}}
   \]

5. **Selection** – Use NSGA-II selection (rank + crowding distance) to choose parents.
6. **Crossover & Mutation** – Apply blend crossover and Gaussian mutation to create offspring portfolios.
7. **Elitism** – Merge parents and offspring, re-sort, and keep the best \( N \) individuals for the next generation.
8. **Termination** – After a fixed number of generations, the final population approximates the **Pareto front / efficient frontier**.

Key configuration (see the report and code for details):

- Population size: 100  
- Generations: 100  
- Crossover probability: 0.9 (BLX-\( \alpha \), \( \alpha = 0.5 \))  
- Mutation probability: 0.1 (Gaussian)  

Further methodological details and justification are in  
`NSGA_II_FinancialReport.md` and [`NSGA_II_FinancialReport.pdf`](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf).

---

## Results

### Pareto Front (Efficient Frontier)

The NSGA-II run produces a **Pareto front** of non-dominated portfolios, mapping risk (variance) against expected annual return. Each point corresponds to a feasible allocation where you cannot improve return without increasing risk, or reduce risk without sacrificing return.

![Pareto Front](pareto_front.png)

The front is generated directly by `nsga_ii_portfolio_final.py` and analyzed in  
[`NSGA_II_FinancialReport.md`](NSGA_II_FinancialReport.md).

### Key Portfolios

From the final Pareto front, three representative portfolios are highlighted in the report:

1. **Minimum Risk Portfolio**
   - Strongly diversified with high weights in **HDFC Bank** and **Bharti Airtel**, moderate in **Titan**, minimal in **Tata Steel** and **Axis Bank**.
   - Example metrics (from the report):
     - Expected annual return ≈ 22.19%  
     - Variance ≈ 0.000095  

2. **Maximum Return Portfolio**
   - Concentrated in the highest-return assets: **Bharti Airtel**, **HDFC Bank**, and **Axis Bank**, with negligible allocation to **Tata Steel** and **Titan**.
   - Example metrics:
     - Expected annual return ≈ 29.80%  
     - Variance ≈ 0.000112  

3. **Balanced Portfolio (Highest Sharpe Ratio)**
   - In this dataset, the highest Sharpe portfolio coincides with the maximum return portfolio, providing the best risk-adjusted performance given the historical returns.
   - Sharpe ratio (based on daily returns and zero risk-free rate) ≈ 10.51.

### Balanced Portfolio Allocation

The file `balanced_portfolio_pie.png` visualizes the asset weights of the balanced / highest Sharpe portfolio:

![Balanced Portfolio](balanced_portfolio_pie.png)

This visualization clearly shows the tilt towards **HDFC Bank**, **Bharti Airtel**, and **Axis Bank**, reflecting their superior historical performance and favorable risk characteristics in the study period.

---

## How to Run the Code

### 1. Set Up Environment

```bash
pip install numpy pandas matplotlib deap
```

You may also create and activate a virtual environment and install the above packages there.

### 2. Run the NSGA-II Optimization

```bash
python nsga_ii_portfolio_final.py
```

This will:

- Load preprocessed equity data from `Equity_data/`
- Compute returns and covariance matrix
- Run NSGA-II for the configured number of generations
- Output:
  - The final Pareto front
  - The selected key portfolios
  - Plots `pareto_front.png` and `balanced_portfolio_pie.png`

---

## Repository Structure

```text
AI_proj_git/
├── nsga_ii_portfolio_final.py        # Main NSGA-II implementation
├── Portfolio_Optimization_NSGA_II.md # Theory notes and algorithm description
├── NSGA_II_FinancialReport.md        # Full report (Markdown)
├── NSGA_II_FinancialReport.pdf       # Full report (PDF)
├── pareto_front.png                  # Risk–return Pareto front
├── balanced_portfolio_pie.png        # Balanced portfolio allocation
├── Equity_data/                      # Historical equity price data
└── nsga_venv/                        # Optional virtual environment (local use)
```

---

## References

The methodology and implementation details closely follow the literature cited in the report, including:

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II*. IEEE Transactions on Evolutionary Computation.  
- Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance.  
- Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M., & Gagné, C. (2012). *DEAP: Evolutionary Algorithms Made Easy*. Journal of Machine Learning Research.  

For a full discussion of results and additional references, see  
[`NSGA_II_FinancialReport.md`](NSGA_II_FinancialReport.md) and the PDF version  
[`NSGA_II_FinancialReport.pdf`](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf).


