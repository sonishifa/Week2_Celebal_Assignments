# Tesla Global Deliveries & Pricing: End-to-End ML Pipeline

An end-to-end Machine Learning and Time Series forecasting pipeline built for Tesla global vehicle delivery records and pricing datasets (spanning 2015 to 2025). This project demonstrates data engineering, comprehensive Exploratory Data Analysis (EDA), regression models, hyperparameter tuning, recursive forecasting, explainability (SHAP & Permutation Importance), and customer/product segmentation using unsupervised learning.

---

## Project Overview

This repository hosts a robust Jupyter notebook (`tesla_ml_pipeline.ipynb`) that runs a complete data science workflow on Tesla's historical deliveries dataset. The primary objective is to analyze delivery dynamics, engineer high-value temporal and ratios features, predict quarterly/monthly vehicle deliveries using regression and tree-based models, forecast future demand recursively, and cluster products to extract business insights.

- **Target Variable**: `Estimated_Deliveries`
- **Temporal Horizon**: 2015 – 2025
- **Dimensional Scope**: Global regions (North America, Europe, Asia, Middle East), models (Model S, Model X, Model 3, Model Y, Cybertruck), pricing, ranges, and charging infrastructure variables.

---

## Technical Stack & Dependencies

The pipeline is implemented using Python and standard data science libraries:
*   **Data Manipulation**: `pandas`, `numpy`
*   **Visualization**: `matplotlib`, `seaborn`
*   **Machine Learning**: `scikit-learn` (preprocessing, linear models, metrics, clustering, decomposition)
*   **Tree-based Modeling**: `xgboost`
*   **Model Explainability**: `shap`

---

## Detailed Pipeline Workflow

The notebook is divided into 12 structured sections:

### 0. Imports & Setup
*   Initializes libraries and sets global configuration for visualizations and warnings.
*   Loads the dataset. *Note: The notebook references `/mnt/user-data/uploads/tesla_deliveries_dataset_2015_2025.csv` for run environment compatibility, which can be modified to `./tesla_deliveries_dataset_2015_2025.csv` for local execution.*

### 1. Preprocessing
*   Checks for missing values and duplicates to ensure data sanity.
*   Extracts datetime features from raw date formats (`Date`, `Quarter`, `Month`).
*   Creates a monotonic `Time_Index` to capture linear trend progression.
*   Performs categorical encoding for `Region`, `Model`, and `Source_Type` using `LabelEncoder`.

### 2. Exploratory Data Analysis (EDA)
Includes 11 distinct, high-quality visualizations:
1.  **Target Distribution**: Histogram and KDE of `Estimated_Deliveries`.
2.  **Yearly Trends**: Line plot showing deliveries growth over years.
3.  **Seasonality Heatmap**: Monthly seasonal distribution patterns.
4.  **Region Share**: Pie and bar plots showing market share per geographic region.
5.  **Model Performance**: Deliveries breakdown by Tesla vehicle model.
6.  **Model Outliers**: Boxplot highlighting delivery variance across model lines.
7.  **Supply vs. Demand**: Production counts vs. Estimated Deliveries.
8.  **Price Elasticity**: Scatter plot showing price vs. delivery volume.
9.  **Infrastructure Correlation**: Charging station density vs. deliveries.
10. **Correlation Heatmap**: Linear relationship matrix between numeric features.

### 3. Feature Engineering
Constructs high-value domain features:
*   **Cyclical Temporal Encodings**: `Month_sin` and `Month_cos` to preserve seasonal closeness (e.g., Dec-Jan).
*   **Operational & Financial Ratios**:
    *   `Delivery_Ratio` (Deliveries / Production)
    *   `Price_per_km` (Base Price / Range in km)
    *   `CO2_per_delivery` (CO2 Savings / Estimated Deliveries)
    *   `Charging_density` (Charging Stations / Base Price)
*   **Sequential Features**:
    *   **Lag features**: `Lag_1`, `Lag_3`, and `Lag_6` of deliveries to capture momentum.
    *   **Rolling statistics**: `Roll_3` and `Roll_6` moving averages to smooth short-term variance.

### 4. Baseline Regression
*   Implements a standard `LinearRegression` model.
*   Enforces strict temporal splitting (80/20 train/test split based on `Time_Index`) to prevent future look-ahead bias.
*   Normalizes features using `StandardScaler`.
*   Evaluates performance using **RMSE**, **MAE**, and **$R^2$**.
*   Plots **Residual Distributions** to analyze model bias.

### 5. Tree-Based Models
*   Trains a baseline `RandomForestRegressor` (200 estimators) to capture non-linear feature interactions.
*   Trains an `XGBoost` regressor (tuned with learning rate `0.1`, `max_depth=6`, `200` estimators).
*   Compares tree baseline scores directly with the Linear Regression baseline.

### 6. Regularization & Hyperparameter Tuning
*   Applies L1 and L2 regularization: **Ridge**, **Lasso**, and **ElasticNet**.
*   Uses `GridSearchCV` with 5-fold cross-validation to search over:
    *   `alpha` (regularization strength): 30 values log-spaced between $10^{-3}$ and $10^3$.
    *   `l1_ratio` (ElasticNet mix parameter): `[0.1, 0.25, 0.5, 0.75, 0.9]`.
*   Generates **Regularization Path plots** showing coefficient shrinkage as `alpha` increases.
*   Compares the regression coefficients of each regularized model.

### 7. Time Series Forecasting
*   Aggregates data into a strict monthly time series.
*   Engineers autoregressive variables: `Lag_1`, `Lag_2`, `Lag_3`, `RollingMean_3`, and `RollingMean_6`.
*   Trains a time-series specialized Linear Regression model.
*   Implements a **recursive multi-step forecasting loop** to project deliveries 12 months into the future without label leakage.
*   Plots historical values alongside the projected forecast.

### 8. Model Explainability
*   **Linear Model Analysis**: Directly examines and plots normalized coefficients to establish parameter influence.
*   **Permutation Importance**: Evaluates feature importance by measuring the score drop when individual columns are shuffled.
*   **SHAP (SHapley Additive exPlanations)**: Interprets the non-linear XGBoost model:
    *   `SHAP Bar Plot` for global feature importance.
    *   `SHAP Beeswarm Plot` showing directionality of feature impacts.
    *   `SHAP Dependence Plot` detailing interactions between pricing, range, and delivery volume.

### 9. Error Analysis
*   Extracts residuals from test predictions and segments them.
*   Analyzes average error metrics by **Region** and **Model**.
*   Visualizes error concentration using a **Region × Model Error Heatmap** to detect localized model underperformance.

### 10. Temporal Trend Analysis
*   Splits the timeline into two distinct eras: **2015–2020 (Growth Phase)** vs. **2021–2025 (Mature Phase)**.
*   Computes descriptive metrics comparing average deliveries, pricing, and variability across eras.
*   Plots comparative timelines and boxplots to visualize how market dynamics evolved.

### 11. Customer/Product Segmentation (Clustering)
*   Applies Unsupervised Learning using `KMeans` to group delivery contexts.
*   Uses the **Elbow Method** to select the optimal number of clusters ($k=3$).
*   Projects features into 2D space using **PCA (Principal Component Analysis)** for visualization.
*   Profiles the 3 clusters:
    *   **Cluster 0 (Premium / Enthusiast)**: High Price, High Range, lower delivery volume (e.g., Model S/X, Cybertruck).
    *   **Cluster 1 (Mass Market / Budget)**: Lower Price, Moderate Range, High Deliveries (e.g., Model 3/Y).
    *   **Cluster 2 (Mid-Range / Transitional)**: Mid pricing, hybrid metrics.

---

## Key Findings & Insights
*   **Temporal Splits matter**: Evaluating models using chronological splitting reveals realistic generalization, as temporal autocorrelation in vehicle deliveries is strong.
*   **Lasso Regularization**: Effectively performs feature selection by zeroing out low-value ratio combinations under high alpha penalties.
*   **Segmentation**: The KMeans clusters clearly outline Tesla's transition from a high-end luxury vehicle manufacturer (Model S/X) to a high-volume mass-market producer (Model 3/Y).
*   **Autoregressive Features**: Adding lagged metrics (`Lag_1` and `Roll_3`) significantly reduces predictive error, highlighting the sequential nature of supply chain and seasonal delivery waves.
