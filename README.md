👉 https://enzo-dash.streamlit.app/

# 📊 Portfolio Backtesting Dashboard

An interactive **Streamlit dashboard** to backtest investment portfolios composed of **ETFs, crypto assets, and commodities**.

The app allows users to build portfolios with custom weights, analyze historical performance, compare benchmarks, and visualize risk metrics.

---

## 🚀 Features

- 📦 Portfolio builder with custom asset weights  
- 📈 NAV evolution (base 10,000 USD)
- 📊 Key metrics: CAGR, Volatility, Sharpe Ratio 
- 🆚 Benchmark comparison (up to 3 assets)
- 📉 Returns distribution (monthly / annual)
- 🎲 Monte Carlo simulation
- 🌐 Efficient frontier
- 🔗 Correlation matrix
- ⚠️ Risk & stress analysis

---

## 🧱 Project Structure

```text
.
├── app.py                      # Main Streamlit entry point
├── requirements.txt            # Python dependencies
├── README.md
├── .gitignore
├── lib/                         # Core logic & calculations
│   ├── data.py
│   ├── portfolio.py
│   └── ui.py
└── sections/                    # Streamlit UI sections
    ├── portfolio_builder.py
    ├── portfolio_view.py
    ├── benchmark_chart.py
    ├── returns_distribution.py
    ├── monte_carlo_simulation.py
    ├── efficient_frontier.py
    ├── correlation_matrix.py
    ├── risk_stress.py
    └── factor_exposure.py