# AI-Driven Quantamental Portfolio Terminal 

An institutional-grade portfolio management system that merges **Deep Learning**, **Modern Portfolio Theory (PMPT)**, and **Fundamental Analysis**. This terminal automates the process of stock picking, risk allocation, and portfolio recalibration across global markets.

## Core Features

### 1. Predictive Engine (6D LSTM)
Unlike traditional models that only look at price, this terminal uses a **6-Dimensional Long Short-Term Memory (LSTM)** neural network to forecast cumulative returns. The model analyzes:
*   **Price Action:** Historical returns.
*   **Liquidity:** Trading volume.
*   **Technical Momentum:** RSI and MACD.
*   **Macro Environment:** 10-Year Treasury Yields (^TNX).
*   **Valuation:** P/E Ratios.

### 2. Hierarchical Risk Parity (HRP)
Implementation of **Marcos López de Prado's** HRP algorithm. It uses machine learning clustering (Euclidean distance) to build a hierarchical tree of assets, distributing risk more robustly than the classic Markowitz Mean-Variance optimization by avoiding the instability of covariance matrix inversion.

### 3. "Quantamental" Screening
The terminal acts as a digital Value Investor. Every candidate must pass a strict fundamental scan:
*   **Profitability:** Positive EPS and ROE.
*   **Growth:** Revenue and Earnings expansion.
*   **Safety:** Debt-to-Equity caps and FCF yield analysis.
*   **MOAT Evaluation:** Categorizes competitive advantages based on gross margins and capital allocation efficiency.

### 4. Risk & Diversification Controls
*   **The "Triple Guillotine":** Filters assets by Volatility, Beta (regional benchmarks), and projected Sharpe Ratio.
*   **Geographic Correlation:** Differentiates between domestic and international correlations to ensure true global diversification.
*   **Sector Caps:** Hard limits to prevent over-concentration (Max 25% per GICS sector).

---

## Dashboard Preview

#### *Institutional Allocation View*
![Portfolio Allocation]<img width="1437" height="721" alt="Captura de pantalla 2026-05-21 133842" src="https://github.com/user-attachments/assets/1dc53028-bdbd-4417-9e58-6439d65796af" />


#### *Risk Analysis & Correlation Heatmap*
![Heatmap]<img width="1414" height="655" alt="Captura de pantalla 2026-05-21 134001" src="https://github.com/user-attachments/assets/5e22968f-0a55-4366-a5ac-840b64bb7931" />

#### *AI Performance vs Benchmarks*
![Backtest Chart]<img width="1436" height="514" alt="Captura de pantalla 2026-05-21 133445" src="https://github.com/user-attachments/assets/ad1609db-b545-4004-8dd3-3ee534bea5e7" />

---

##  Tech Stack

*   **Language:** Python 3.11
*   **Machine Learning:** TensorFlow / Keras (LSTM)
*   **Optimization:** SciPy / Scikit-Learn
*   **Financial Data:** Yahoo Finance API
*   **Visualization:** Streamlit & Plotly
*   **Database:** SQLite3 (Portfolio persistence)

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/SebasAct24/AI-PORTFOLIO-MANAGER.git
