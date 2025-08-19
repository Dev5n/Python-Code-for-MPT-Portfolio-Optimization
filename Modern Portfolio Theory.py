import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# List of asset tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Download historical data
data = yf.download(tickers, start="2020-01-01", end="2025-01-01")['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Annualized average returns and covariance
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252
risk_free_rate = 0.015  # Assume 1.5% risk-free rate

# Number of assets
num_assets = len(tickers)

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.dot(weights, mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, std_dev

# Negative Sharpe Ratio (for minimization)
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_return, p_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(p_return - risk_free_rate) / p_std_dev

# Constraints and bounds
constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Sum of weights = 1
bounds = tuple((0, 1) for _ in range(num_assets))  # No short-selling

# Initial guess (equal distribution)
init_guess = num_assets * [1. / num_assets]

# Optimization
optimized = minimize(negative_sharpe_ratio, init_guess,
                     args=(mean_returns, cov_matrix, risk_free_rate),
                     method='SLSQP', bounds=bounds, constraints=constraints)

# Extract results
optimal_weights = optimized.x
opt_return, opt_std_dev = portfolio_performance(optimal_weights, mean_returns, cov_matrix)
opt_sharpe = (opt_return - risk_free_rate) / opt_std_dev

# Print results
print("Optimal Portfolio Allocation:")
for i, ticker in enumerate(tickers):
    print(f"{ticker}: {optimal_weights[i]:.2%}")
print(f"\nExpected Annual Return: {opt_return:.2%}")
print(f"Annual Volatility (Risk): {opt_std_dev:.2%}")
print(f"Sharpe Ratio: {opt_sharpe:.2f}")

# Plotting Efficient Frontier
def simulate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3, num_portfolios))
    weights_record = []

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weights_record.append(weights)
        
        portfolio_return, portfolio_std_dev = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_std_dev
        results[1, i] = portfolio_return
        results[2, i] = (portfolio_return - risk_free_rate) / portfolio_std_dev

    return results, weights_record

num_portfolios = 10000
results, _ = simulate_random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate)

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', alpha=0.3)
plt.colorbar(label='Sharpe Ratio')
plt.scatter(opt_std_dev, opt_return, marker='*', color='r', s=300, label='Optimal Portfolio')
plt.title('Efficient Frontier with Optimal Portfolio')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.legend(labelspacing=0.8)
plt.grid(True)
plt.show()
