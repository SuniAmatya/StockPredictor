# Predicting Next-Day Stock Movement of NIFRA

Project: Can historical price and trading data predict whether Nepal Infrastructure Bank Limited (NIFRA) shares go **UP** or **DOWN** on the next trading day?

> **Disclaimer:** Educational, experimental project. Not investment advice.

## Problem Statement
- **Target = 1 (UP):** next day's close > today's close
- **Target = 0 (DOWN):** next day's close <= today's close

The goal is to test whether the features carry **useful signal**, not to maximise accuracy.

## Why NIFRA?
NIFRA is Nepal's first and only "E" class bank, created to finance infrastructure projects, so its shares are tied to an economically important sector.

## Repository Structure
```
DSML-Project/
├── Stock Trading of Nepal Infrastructure Bank Limited.csv
├── NIFRAStockAnalysis.ipynb     # exploratory and risk-return analysis
├── NIFRAPrediction.ipynb        # features + Logistic Regression, tuning, Random Forest
└── README.md
```

## Dataset
- **Source:** NEPSE data via [NepseAlpha](https://nepsealpha.com/nepse-data)
- **Period:** Oct 2021 to Oct 2026 (~1,170 trading days)
- **Columns used:** Date, Open, High, Low, Close, Volume, Turn Over


## Methodology
```
Raw Data -> Cleaning -> Feature Engineering -> Target Creation -> Chronological Train/Test Split -> Models -> Evaluation
```

**Engineered features (12), all computed from data up to the current day to avoid leakage:**

| Feature | Meaning |
|---|---|
| `Return` | Today's % change vs previous close |
| `Ret_lag1d`, `Ret_lag2d` | Return 1 and 2 days ago |
| `Ret_5d` | % change vs 5 trading days ago |
| `Gap` | Open vs previous close (%) |
| `Range` | (High - Low) / Close (%) |
| `Close_Position` | Where the close sits within the day's range (0 = low, 100 = high) |
| `AvgPrice_vs_Close` | Turnover/Volume average price vs close (%) |
| `Close_vs_MA5`, `Close_vs_MA20` | Close relative to 5- and 20-day moving averages |
| `Volatility10` | 10-day standard deviation of returns |
| `Vol_ratio10` | Today's volume vs 10-day average volume |

**Split:** oldest 80% train, newest 20% test (230 days), no shuffling.

**Models**
1. Logistic Regression (`StandardScaler` + `LogisticRegression` in one pipeline)
2. Logistic Regression tuned with `GridSearchCV` + `TimeSeriesSplit` (scored by ROC-AUC)
3. Random Forest (300 trees, depth 5, min leaf 20) as a non-linear comparison

**Evaluation:** Accuracy, Precision/Recall, Confusion Matrix, ROC-AUC, Majority-class baseline, and Walk-forward precision on the top 15% highest-confidence days.

---

## Notebook 1: NIFRAStockAnalysis (Exploratory Analysis)

| Measure | Result | Interpretation |
|---|---|---|
| Average close | Rs. 253.53 (std 45.05) | Moderate dispersion |
| Average daily range | ~3.37% of close | Typical intraday swing |
| Annualized volatility | ~34.5% | High-volatility stock |
| Annual return | 2021: -13.4%, 2022: -28.2%, 2023: -1.6%, 2024: +22.7%, 2025: +6.7%, 2026 (to Oct): -4.7% | Post-listing decline, 2024 rally, mild fade |
| Variance of daily returns | 6.55 (2021) to 2.05 (2026) | Daily swings calmed over time |
| 50/200-day moving averages | Price mostly below both in 2026 | Long-term trend bearish (lagging indicator) |
| Z-score outliers (\|Z\| > 3) | 26 days, clustered mid-2024 | Likely news/liquidity-driven spikes |

Annual return (start to end of year) and mean daily return can disagree: a year can end lower while its average daily return is slightly positive.

### Limitations
- 2021 and 2026 are partial years, so their annual returns are not comparable to full years.
- Coefficient of variation is not meaningful because mean daily return is near zero.
- Z-scores assume roughly normal returns; stock returns have fat tails, so counts are indicative.
- Moving-average signals lag and are not trade recommendations.
- Circuit-limit days (~10% moves) and market closures create outlier returns.

---

## Notebook 2: NIFRAPrediction 

### Results (test set: 230 days, Nov 2025 to Oct 2026)

| Model | Accuracy | ROC-AUC | UP precision / recall |
|---|---|---|---|
| Majority baseline (always DOWN) | 0.574 | 0.500 | n/a |
| Logistic Regression | 0.565 | 0.539 | 0.43 / 0.06 |
| Tuned Logistic (C = 0.1) | 0.580 | 0.539 | 0.56 / 0.05 |
| Random Forest | 0.580 | 0.524 | 0.60 / 0.03 |

Walk-forward top-15% precision vs UP base rate: 
| Model | Walk-Forward Top-15% Precision | Performance Assessment |
| :--- | :--- | :--- |
| 
| **Random Forest** | **44.83%** | **Ineffective** —  performs slightly below a 50% coin flip despite having the highest relative score |
| **Logistic Regression** | **43.45%** | **Ineffective** — non-linear splits fail to generalize out-of-sample across changing market regimes|
| **GridSearch-Tuned Model** | **42.76%** | **Ineffective** — hyperparameter tuning overfits historical noise and degrades forward precision |

### Interpretation
- **No meaningful predictive skill.** All models score at or near the always-DOWN baseline (0.574), and ROC-AUC is only marginally above 0.5.
- All models predict DOWN on 94-98% of days. With UP as the minority class, the 0.5 threshold rarely produces UP predictions, so accuracy mostly reflects class balance.
- Hyperparameter tuning and a non-linear model did not help, which points to the **information in the features**, not model choice, as the limit.
- Even the most confident predictions are not clearly better than the UP base rate.
- The next-day UP probability was close to 50% for every model. A single correct prediction (e.g. a DOWN day matching a DOWN forecast) is not evidence of skill.

**Conclusion:** 
Features derived from past price and volume carry little to no reliable signal for NIFRA's next-day direction. This answers the problem statement and fits the project's aim of testing for signal.

### Limitations
1. **Small data:** ~1,150 usable rows, 230 test days; AUC of 0.54 cannot be confidently separated from 0.5.
2. **Thin single-stock market** with circuit limits (~10%) distorting returns.
3. **Limited information:** No market-wide (NEPSE index, sector), dividend/bonus/right-share, news or macro data.
4. **Class imbalance and threshold:** The 0.5 cutoff suppresses UP predictions; a training-based threshold or probability-based metrics are fairer.
5. **Baseline and uncertainty:** no confidence interval reported for ROC-AUC.
6. **Walk-forward evaluation** uses the fold's own top-15% cutoff and folds are small; treat as indicative.
7. **No trading costs or backtest:** measures classification, not profitability.
8. **Non-stationarity:** patterns from 2021-2025 may not hold in 2026.
9. Logistic Regression was chosen for interpretability; more complex models are unlikely to help with this sample size (Random Forest confirmed this).

### Possible future work
Market-wide features (NEPSE index, sector returns), calendar and announcement effects, bootstrap CIs for ROC-AUC, a persistence baseline, anomaly detection around public events.
