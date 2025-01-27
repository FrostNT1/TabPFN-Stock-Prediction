# **Stock Price Prediction with TabPFN**

## 1. Overview
This repository applies the [TabPFN (Tabular Prior-data Fitted Network)](https://github.com/automl/TabPFN) model to **next-day stock price prediction**. TabPFN uses two-way 1D attention (across features and samples) and is pre-trained on millions of synthetic tabular datasets. We compare its performance with:
- **LightGBM** (a strong tree-based baseline)
- **Simple RNN/LSTM** (a common deep-learning baseline for time-series)

We aim to show how TabPFN can be directly used on stock data retrieved from [Yahoo Finance](https://pypi.org/project/yfinance/) and how it performs in both **forecast accuracy** and a **basic trading strategy**.

---

## 2. Statement of Problem
**Goal**: Predict the next-day closing price (or price return) of a chosen stock or ETF.  
**Why**:  
- Accurate short-term predictions can aid in decision-making for traders, analysts, or automated systems.  
- Traditional methods (like tree-based models) can work well on tabular data, but we want to see if TabPFN's attention-based approach can match or outperform them in terms of speed and accuracy on small-to-medium datasets.

---

## 3. Data Sources & Description
- **Yahoo Finance** via the `yfinance` library
- Typical features: Date, Open, High, Low, Close, Volume
- Engineered features: Moving Averages, RSI, MACD, or any additional technical indicators
- Each row corresponds to one trading day. We create a shifted "next-day" target column to predict future prices.

---

## 4. Approach
1. **Data Collection**: Pull historical data (e.g., 2010–2023) for a specific ticker like `SPY`, clean and store in a Pandas DataFrame.  
2. **Feature Engineering**: Compute rolling averages, RSI, or other indicators.  
3. **Model Training**  
   - **LightGBM**: Classic gradient boosting baseline  
   - **RNN**: Simple Keras/TensorFlow model for time-series  
   - **TabPFN**: Directly consumes tabular rows and outputs next-day predictions  
4. **Evaluation**  
   - **RMSE, MAE, or R²** on a time-based split (train/validation/test)  
   - **Trading Strategy**: If predicted return is positive, simulate a "long" position; otherwise remain in cash (or short). Measure total returns.  
5. **Comparison**: Which model yields the best numeric forecasts and/or trading performance?

---

## **Diagram: End-to-End Flow**
```mermaid
flowchart LR
    A[Yahoo Finance Data] -->|Download| B[Raw OHLCV Data]
    B -->|Compute| C[Technical Indicators]
    C -->|Feature Engineering| D[Training Data]
    
    subgraph Models
        E[TabPFN]
        F[LightGBM]
        G[RNN/LSTM]
    end
    
    D --> E
    D --> F
    D --> G
    
    E --> H[Predictions]
    F --> H
    G --> H
    
    H -->|Evaluate| I[RMSE/MAE]
    H -->|Simulate| J[Trading Returns]
    
    style A fill:#f9f,stroke:#333
    style Models fill:#f5f5f5,stroke:#333
    style H fill:#9cf,stroke:#333
    style I fill:#9f9,stroke:#333
    style J fill:#9f9,stroke:#333
```

## Diagram: TabPFN's Two-Way Attention Mechanism

![alt text](images/TabPFN-Paper.png)
---

## 5. Timeline & Deliverables
### Timeline
- Data Gathering & Feature Engineering
- Initial Model Training
- Comparison & Validation
- Backtesting & Strategy Analysis
- Final Report

### Deliverables
- **Data**: Cleaned CSV or Parquet files with indicators
- **Notebooks**:
  - Data exploration
  - Model training scripts
  - Backtesting
- **Results**: Plots, metrics (RMSE, directional accuracy, cumulative returns)

---


## 6. Resources
### Software
- Python 3
- yfinance
- tabpfn
- lightgbm
- tensorflow/torch for RNNs

### Hardware
- Google Colab or a local machine with a GPU for faster training

### Documentation
- [TabPFN GitHub](https://github.com/automl/TabPFN)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [Keras Documentation](https://keras.io/api/)

---

## 7. Stakeholders
- **Data Scientists/Quants**: Interested in leveraging new ML approaches for market forecasts
- **Traders/Portfolio Managers**: May integrate these predictions into strategies
- **Academics/Researchers**: Can extend the approach to other tabular/time-series data

---

## 8. Contact Information
- **Maintainer**: Shivam Tyagi (st.shivamtyagi.01@gmail.com)
- **Contributors**: Feel free to submit PRs/issues on GitHub or email the maintainer

---

## 9. License
Distributed under the Apache 2.0 License. See `LICENSE` file for more information.

---

## 10. How to Contribute
1. Fork this repository
2. Create a new branch for your feature/fix
3. Commit your changes with clear messages
4. Open a Pull Request describing your changes

We welcome improvements to:
- Data pipelines
- Feature engineering steps  
- Model training scripts
- Documentation and tutorials

---

## 11. Results and Findings

### TabPFN for Time Series Analysis: Key Insights

#### Challenges

1. **Structural Limitations**
   - TabPFN's architecture, while powerful for tabular data, isn't optimized for sequential data
   - Time window flattening (lookback period of 20) results in high-dimensional input (640 dimensions)
   - Loss of explicit temporal structure in the data representation

2. **Data Distribution**
   - Slight class imbalance observed in market direction:
     - Training: 3386 up vs 2966 down movements
     - Validation: 701 up vs 558 down movements
     - Test: 388 up vs 365 down movements
   - Model sensitivity to market regime changes needs consideration

3. **Temporal Aspects**
   - Limited ability to capture long-term dependencies
   - Independent sample treatment may miss important sequential patterns
   - No built-in mechanism for temporal dependency handling

#### Advantages

1. **Zero-Shot Capabilities**
   - No traditional training required
   - Potentially more robust to market regime changes
   - Reduced risk of overfitting to historical patterns

2. **Uncertainty Quantification**
   - Probability estimates enable confidence-based trading
   - Enhanced risk management through probability thresholds
   - More nuanced decision-making capability

3. **Feature Interaction Analysis**
   - Effective capture of non-linear relationships
   - Strong handling of complex technical indicator patterns
   - Attention mechanism provides insight into feature importance

### Recommendations for Future Work

1. **Model Integration**
   - Consider ensemble approaches combining TabPFN with traditional time series models
   - Use TabPFN's probability outputs as features for other models
   - Implement dynamic confidence thresholds based on market conditions

2. **Feature Engineering Improvements**
   - Optimize lookback period to balance information and dimensionality
   - Incorporate more domain-specific features
   - Add explicit temporal features for better time awareness

3. **Evaluation Framework**
   - Implement walk-forward testing for realistic performance assessment
   - Focus on high-confidence predictions
   - Compare against simple baseline models

### Final Assessment
While TabPFN shows promise in handling complex tabular data, its application to stock prediction faces notable challenges. The model might be more valuable as part of a hybrid approach rather than a standalone solution. Success depends heavily on proper feature engineering and careful handling of temporal aspects. The project demonstrates both the potential and limitations of applying advanced tabular models to time series prediction tasks.
