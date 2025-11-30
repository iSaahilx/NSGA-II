import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import os

# Define the problem: maximize return, minimize risk
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))  # Maximize return, minimize risk
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Load and preprocess stock data
def load_stock_data():
    stocks = ['TATASTEEL', 'TITAN', 'AXISBANK', 'HDFCBANK', 'BHARTIARTL']
    data = {}
    
    for stock in stocks:
        file_path = f"Quote-Equity-{stock}-EQ-17-04-2024-to-17-04-2025.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Calculate daily returns (percentage change)
            if 'Close Price' in df.columns:
                df['Return'] = df['Close Price'].pct_change()
                data[stock] = df['Return'].dropna()
    
    # Combine all returns into a single DataFrame
    returns_df = pd.DataFrame(data)
    return returns_df

# Calculate expected returns and covariance matrix
def calculate_portfolio_stats(returns_df):
    # Expected returns (mean of daily returns)
    expected_returns = returns_df.mean()
    
    # Covariance matrix
    cov_matrix = returns_df.cov()
    
    return expected_returns, cov_matrix

# Evaluate portfolio
def evaluate_portfolio(individual, expected_returns, cov_matrix):
    # Normalize weights to sum to 1
    weights = np.array(individual)
    weights = weights / np.sum(weights)
    
    # Calculate expected portfolio return
    portfolio_return = np.sum(weights * expected_returns)
    
    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    return portfolio_return, portfolio_variance

# NSGA-II Setup and Execution
def run_nsga_ii(expected_returns, cov_matrix, num_assets=5, pop_size=100, num_gen=100):
    # Initialize DEAP toolbox
    toolbox = base.Toolbox()
    
    # Register gene, individual, and population initialization
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function
    toolbox.register("evaluate", evaluate_portfolio, expected_returns=expected_returns, cov_matrix=cov_matrix)
    
    # Register genetic operators
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)
    
    # Initialize population
    pop = toolbox.population(n=pop_size)
    
    # Run NSGA-II algorithm
    algorithms.eaMuPlusLambda(
        pop, toolbox, 
        mu=pop_size, 
        lambda_=pop_size, 
        cxpb=0.9, 
        mutpb=0.1, 
        ngen=num_gen, 
        stats=None, 
        halloffame=None, 
        verbose=True
    )
    
    return pop

# Visualize Pareto front
def plot_pareto_front(population, stocks):
    # Extract returns and risks
    returns = [ind.fitness.values[0] for ind in population]
    risks = [ind.fitness.values[1] for ind in population]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    plt.scatter(risks, returns, c='blue', alpha=0.5)
    plt.title('Portfolio Optimization Pareto Front')
    plt.xlabel('Risk (Variance)')
    plt.ylabel('Expected Return')
    plt.grid(True)
    
    # Highlight extreme portfolios: min risk and max return
    min_risk_idx = np.argmin(risks)
    max_return_idx = np.argmax(returns)
    
    plt.scatter(risks[min_risk_idx], returns[min_risk_idx], c='green', s=100, label='Min Risk')
    plt.scatter(risks[max_return_idx], returns[max_return_idx], c='red', s=100, label='Max Return')
    
    plt.legend()
    plt.savefig('pareto_front.png')
    plt.close()
    
    # Print details of notable portfolios
    print("\nPortfolio with Minimum Risk:")
    weights = np.array(population[min_risk_idx]) / sum(population[min_risk_idx])
    for i, stock in enumerate(stocks):
        print(f"{stock}: {weights[i]*100:.2f}%")
    print(f"Expected Return: {returns[min_risk_idx]*100:.4f}%")
    print(f"Risk (Variance): {risks[min_risk_idx]:.6f}")
    
    print("\nPortfolio with Maximum Return:")
    weights = np.array(population[max_return_idx]) / sum(population[max_return_idx])
    for i, stock in enumerate(stocks):
        print(f"{stock}: {weights[i]*100:.2f}%")
    print(f"Expected Return: {returns[max_return_idx]*100:.4f}%")
    print(f"Risk (Variance): {risks[max_return_idx]:.6f}")
    
    # Find and print a balanced portfolio
    # Use Sharpe ratio as a metric (assuming risk-free rate = 0)
    sharpe_ratios = [ret/risk if risk > 0 else 0 for ret, risk in zip(returns, risks)]
    balanced_idx = np.argmax(sharpe_ratios)
    
    print("\nBalanced Portfolio (Highest Sharpe Ratio):")
    weights = np.array(population[balanced_idx]) / sum(population[balanced_idx])
    for i, stock in enumerate(stocks):
        print(f"{stock}: {weights[i]*100:.2f}%")
    print(f"Expected Return: {returns[balanced_idx]*100:.4f}%")
    print(f"Risk (Variance): {risks[balanced_idx]:.6f}")
    print(f"Sharpe Ratio: {sharpe_ratios[balanced_idx]:.4f}")

# Main function
def main():
    print("Loading stock data...")
    returns_df = load_stock_data()
    stocks = returns_df.columns.tolist()
    
    print(f"Calculating statistics for stocks: {stocks}")
    expected_returns, cov_matrix = calculate_portfolio_stats(returns_df)
    
    print("Expected Annual Returns:")
    for stock, ret in expected_returns.items():
        print(f"{stock}: {ret*252*100:.2f}%")  # Annualized returns (252 trading days)
    
    print("\nRunning NSGA-II optimization...")
    population = run_nsga_ii(expected_returns, cov_matrix, num_assets=len(stocks))
    
    print("\nOptimization complete. Plotting Pareto front...")
    plot_pareto_front(population, stocks)
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main() 