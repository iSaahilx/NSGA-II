## AI Portfolio Optimization with NSGA-II

This repository implements a complete multi-objective portfolio optimization workflow using the Non‑dominated Sorting Genetic Algorithm II (NSGA‑II).  
It reproduces and extends the analysis described in the project report *“Pareto Optimization for Portfolio Management Using NSGA‑II”* [`NSGA_II_FinancialReport.pdf`](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf), turning the methodology into fully runnable, well-documented Python code.

---

## Overview

The project constructs equity portfolios that balance two conflicting objectives:

- **Maximize expected portfolio return** \(based on historical daily returns\)
- **Minimize portfolio risk** \(portfolio variance\)

Rather than outputting a single “best” portfolio, NSGA‑II generates a **Pareto front** of non‑dominated solutions.  
Each point on this front represents an efficient trade‑off where any improvement in return requires accepting higher risk, and vice versa.

The study uses one year of daily data (17 April 2024 – 17 April 2025) for five large‑cap Nifty 50 constituents, as detailed in the report [`NSGA_II_FinancialReport.pdf`](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf).

---

## Data and Investment Universe

The portfolio universe consists of the following stocks, with cleaned NSE price histories stored as CSV files:

| Ticker      | Company        | Sector          |
|------------|----------------|-----------------|
| TATASTEEL  | Tata Steel     | Metals          |
| TITAN      | Titan Company  | Consumer Goods  |
| AXISBANK   | Axis Bank      | Banking         |
| HDFCBANK   | HDFC Bank      | Banking         |
| BHARTIARTL | Bharti Airtel  | Telecom         |

From these series the code computes:

- Daily log returns  
- Expected annual returns (assuming 252 trading days)  
- The \(5 \times 5\) covariance matrix of returns, used in risk calculations  

Key statistical findings from the report include:

- **Highest expected returns**: BHARTIARTL and HDFCBANK  
- **Negative expected returns in this window**: TATASTEEL and TITAN  
- **Highest individual variance**: TATASTEEL  

These properties strongly influence optimal weights in the final portfolios [`NSGA_II_FinancialReport.pdf`](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf).

---

## Optimization Formulation

Let \(w_i\) denote the weight of asset \(i\) in a portfolio of \(n = 5\) assets.

- **Objective 1 – Maximize expected return**

\\[
E(R_p) = \sum_{i=1}^{n} w_i \, E(R_i)
\\]

- **Objective 2 – Minimize portfolio variance**

\\[
\sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \sigma_{ij}
\\]

Subject to:

- **Full investment**: \\(\sum_{i=1}^{n} w_i = 1\\)  
- **No short selling**: \\(w_i \ge 0\\) for all \(i\)

This leads to a classic bi‑objective portfolio problem where NSGA‑II searches directly in weight space under these constraints.

---

## NSGA‑II Implementation

The implementation uses the DEAP evolutionary computation library and closely follows the configuration described in the report:

- **Population size**: 100 portfolios  
- **Generations**: 100  
- **Crossover**: Blend crossover (BLX‑α with α = 0.5), probability 0.9  
- **Mutation**: Gaussian mutation (mean 0, std 0.1), probability 0.1  
- **Selection**: `tools.selNSGA2` with elitism and crowding distance  

Chromosomes are length‑5 real‑valued vectors representing raw weights.  
Constraint handling is enforced after genetic operations:

1. Clip negative weights to 0.  
2. Renormalize so that weights sum to 1.  

Fitness evaluation returns a two‑element tuple:

- \(f_1 = -E(R_p)\) (minimized to maximize return)  
- \(f_2 = \sigma_p^2\) (portfolio variance)  

This design exactly matches the mathematical formulation and ensures compatibility with NSGA‑II’s dominance comparisons, as explained in [`NSGA_II_FinancialReport.pdf`](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf).

---

## Results and Visualizations

### Pareto Front

The file `pareto_front.png` shows the final Pareto front in risk–return space.  
Each point corresponds to a non‑dominated portfolio with its own allocation across the five assets.

![Pareto Front](pareto_front.png)

The front exhibits the expected upward‑sloping shape: portfolios with higher expected returns carry higher variance, confirming the risk–return trade‑off observed in the report.

### Balanced Portfolio Allocation

The chart `balanced_portfolio_pie.png` visualizes the allocation of the selected **balanced portfolio**, defined as the portfolio with the highest Sharpe ratio (assuming risk‑free rate 0).

![Balanced Portfolio](balanced_portfolio_pie.png)

The report shows that this portfolio heavily weights HDFCBANK and BHARTIARTL, with modest exposure to AXISBANK and minimal or zero allocation to the lower‑return stocks TATASTEEL and TITAN [`NSGA_II_FinancialReport.pdf`](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf).

### Highlighted Portfolios

The analysis identifies three representative portfolios on the Pareto front:

1. **Minimum Risk Portfolio**  
   - Strong diversification across HDFCBANK, BHARTIARTL, and TITAN  
   - Very small allocations to TATASTEEL and AXISBANK  
   - Lowest variance, suitable for highly risk‑averse investors  

2. **Maximum Return Portfolio**  
   - Dominated by BHARTIARTL and HDFCBANK  
   - Negligible or zero allocation to the negative‑return assets  
   - Highest expected annual return but also higher variance  

3. **Balanced (Maximum Sharpe) Portfolio**  
   - Same composition as the maximum‑return portfolio in this dataset  
   - Achieves the best risk‑adjusted performance (Sharpe ratio ≈ 10.5)  

Detailed numerical results and tables for these portfolios are provided in [`NSGA_II_FinancialReport.pdf`](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf).

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

- Load the historical price data from the `Quote-Equity-*.csv` files  
- Construct return and covariance matrices  
- Run the NSGA‑II optimization loop  
- Save visualizations for the Pareto front and the chosen balanced portfolio  

You can modify parameters such as population size, number of generations, and risk‑return preferences directly in `nsga_ii_portfolio_final.py`.

---

## Repository Contents

| File | Description |
|------|-------------|
| `nsga_ii_portfolio_final.py` | Main NSGA‑II implementation used for experiments and figures. |
| `nsga_ii_portfolio_fix.py`   | Refined/cleaned version of an earlier prototype. |
| `nsga_ii_portfolio.py`       | Initial/basic NSGA‑II implementation. |
| `Portfolio_Optimization_NSGA_II.md` | Theory and algorithm overview for NSGA‑II in portfolio management. |
| `NSGA_II_FinancialReport.pdf` | Full academic-style report with methodology, statistics, and detailed results. |
| `pareto_front.png`           | Visualization of the final Pareto front. |
| `balanced_portfolio_pie.png` | Allocation chart for the balanced (maximum Sharpe) portfolio. |
| `Quote-Equity-*.csv`         | Cleaned price data for the five Nifty 50 stocks. |

---

## Project Structure

```text
AI_proj/
├── nsga_ii_portfolio_final.py
├── nsga_ii_portfolio_fix.py
├── nsga_ii_portfolio.py
├── NSGA_II_FinancialReport.pdf
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

1. Markowitz, H. (1952). *Portfolio Selection*. The Journal of Finance.  
2. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA‑II*. IEEE Transactions on Evolutionary Computation.  
3. Saahil Ahmad, Md Rameez Haider, Ramjan Khandelwal. *Pareto Optimization for Portfolio Management Using NSGA‑II* [`NSGA_II_FinancialReport.pdf`](file:///e%3A/Projects/AI_proj_git/NSGA_II_FinancialReport.pdf).

---

## License

This project is provided for educational and research purposes only.
