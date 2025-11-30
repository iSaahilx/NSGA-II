import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import os
import re

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
            print(f"Processing {file_path}...")
            
            # The files have a non-standard format. Read as text first
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Find the header line (contains "Date", "series", etc.)
            header_line = None
            for i, line in enumerate(lines):
                if "Date" in line and "OPEN" in line and "HIGH" in line and "LOW" in line:
                    header_line = i
                    break
            
            if header_line is not None:
                # Extract all the data lines in reverse (they're in reverse chronological order)
                data_lines = lines[header_line+1:]
                data_lines.reverse()  # Put in chronological order
                
                # Process data
                dates = []
                close_prices = []
                
                for line in data_lines:
                    # Extract values from quoted fields
                    values = re.findall(r'"([^"]+)"', line)
                    
                    if not values:  # If no quotes, split by whitespace
                        values = [val.strip() for val in line.split() if val.strip()]
                    
                    if len(values) >= 7:  # Ensure we have enough values
                        date_str = values[0]
                        close_price_str = values[6]  # Assuming 'close' is the 7th column
                        
                        try:
                            # Remove commas and convert to float
                            close_price = float(close_price_str.replace(',', ''))
                            dates.append(date_str)
                            close_prices.append(close_price)
                        except ValueError:
                            continue
                
                # Create a DataFrame
                df = pd.DataFrame({'Date': dates, 'Close Price': close_prices})
                
                # Calculate daily returns
                df['Return'] = df['Close Price'].pct_change()
                
                # Store returns for this stock
                data[stock] = df['Return'].dropna().values
    
    # Ensure all arrays have the same length
    min_length = min(len(arr) for arr in data.values()) if data else 0
    for stock in data:
        data[stock] = data[stock][-min_length:]
    
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
    # Ensure weights are non-negative
    weights = np.array([max(0, w) for w in individual])
    
    # Normalize weights to sum to 1
    if np.sum(weights) > 0:
        weights = weights / np.sum(weights)
    else:
        # If all weights are 0, create uniform weights
        weights = np.ones(len(individual)) / len(individual)
    
    # Calculate expected portfolio return
    portfolio_return = np.sum(weights * expected_returns)
    
    # Calculate portfolio variance
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
    
    return portfolio_return, portfolio_variance

# Generate a random portfolio weight vector with non-negative weights summing to 1
def generate_weights(num_assets):
    # Generate random weights
    weights = np.random.random(num_assets)
    # Normalize to sum to 1
    weights = weights / np.sum(weights)
    return weights

# NSGA-II Setup and Execution
def run_nsga_ii(expected_returns, cov_matrix, num_assets=5, pop_size=100, num_gen=100):
    # Initialize DEAP toolbox
    toolbox = base.Toolbox()
    
    # Register custom initialization function for positive weights summing to 1
    toolbox.register("attr_float", random.random)  # generate random number between 0 and 1
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=num_assets)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Register evaluation function
    toolbox.register("evaluate", evaluate_portfolio, expected_returns=expected_returns, cov_matrix=cov_matrix)
    
    # Register genetic operators
    # Blend crossover keeps genes in range [min, max] but does not ensure sum=1
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    
    # Mutation (small Gaussian changes, clamped to positive values later)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    
    # Selection
    toolbox.register("select", tools.selNSGA2)
    
    # Initialize population
    pop = toolbox.population(n=pop_size)
    
    # Run NSGA-II algorithm
    for gen in range(num_gen):
        # Select parents
        offspring = toolbox.select(pop, len(pop))
        
        # Clone selected individuals
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        # Apply crossover and mutation
        for i in range(1, len(offspring), 2):
            if random.random() < 0.9:  # crossover probability
                toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values
                del offspring[i].fitness.values
        
        for i in range(len(offspring)):
            if random.random() < 0.1:  # mutation probability
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
                
            # Ensure weights are non-negative
            for j in range(len(offspring[i])):
                offspring[i][j] = max(0, offspring[i][j])
            
            # Normalize to sum to 1
            weight_sum = sum(offspring[i])
            if weight_sum > 0:
                for j in range(len(offspring[i])):
                    offspring[i][j] /= weight_sum
        
        # Evaluate fitness of modified individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Replace population with offspring
        pop[:] = offspring
        
        # Print progress
        print(f"Generation {gen+1}/{num_gen}")
    
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
    weights = np.array(population[min_risk_idx])
    for i, stock in enumerate(stocks):
        print(f"{stock}: {weights[i]*100:.2f}%")
    print(f"Expected Daily Return: {returns[min_risk_idx]*100:.4f}%")
    print(f"Expected Annual Return: {returns[min_risk_idx]*252*100:.2f}%")
    print(f"Risk (Variance): {risks[min_risk_idx]:.6f}")
    
    print("\nPortfolio with Maximum Return:")
    weights = np.array(population[max_return_idx])
    for i, stock in enumerate(stocks):
        print(f"{stock}: {weights[i]*100:.2f}%")
    print(f"Expected Daily Return: {returns[max_return_idx]*100:.4f}%")
    print(f"Expected Annual Return: {returns[max_return_idx]*252*100:.2f}%")
    print(f"Risk (Variance): {risks[max_return_idx]:.6f}")
    
    # Find and print a balanced portfolio
    # Use Sharpe ratio as a metric (assuming risk-free rate = 0)
    sharpe_ratios = [ret/risk if risk > 0 else 0 for ret, risk in zip(returns, risks)]
    balanced_idx = np.argmax(sharpe_ratios)
    
    print("\nBalanced Portfolio (Highest Sharpe Ratio):")
    weights = np.array(population[balanced_idx])
    for i, stock in enumerate(stocks):
        print(f"{stock}: {weights[i]*100:.2f}%")
    print(f"Expected Daily Return: {returns[balanced_idx]*100:.4f}%")
    print(f"Expected Annual Return: {returns[balanced_idx]*252*100:.2f}%")
    print(f"Risk (Variance): {risks[balanced_idx]:.6f}")
    print(f"Sharpe Ratio: {sharpe_ratios[balanced_idx]:.4f}")
    
    # Create a pie chart of the balanced portfolio
    plt.figure(figsize=(10, 6))
    plt.pie(weights, labels=stocks, autopct='%1.1f%%', startangle=90)
    plt.title('Balanced Portfolio Allocation (Highest Sharpe Ratio)')
    plt.savefig('balanced_portfolio_pie.png')
    plt.close()

# Main function
def main():
    print("Loading stock data...")
    returns_df = load_stock_data()
    
    if returns_df.empty:
        print("ERROR: Could not load stock data. Exiting.")
        return
    
    stocks = returns_df.columns.tolist()
    
    print(f"Calculating statistics for stocks: {stocks}")
    expected_returns, cov_matrix = calculate_portfolio_stats(returns_df)
    
    print("Expected Annual Returns:")
    for stock, ret in expected_returns.items():
        print(f"{stock}: {ret*252*100:.2f}%")  # Annualized returns (252 trading days)
    
    print("\nCovariance Matrix:")
    print(cov_matrix)
    
    print("\nRunning NSGA-II optimization...")
    population = run_nsga_ii(expected_returns, cov_matrix, num_assets=len(stocks))
    
    print("\nOptimization complete. Plotting Pareto front...")
    plot_pareto_front(population, stocks)
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main() 