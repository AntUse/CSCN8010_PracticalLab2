# Practical Lab 2: Multivariate Linear Regression, Non-Parametric Models, and Cross-Validation

This project uses the **Scikit-Learn Diabetes dataset** to build predictive models for **disease progression one year after baseline**. Models are evaluated using **R²**, **MAE**, and **MAPE** with a train/validation/test split. All plots are shown in separate figures for clarity.

## Dataset

- **Source:** [Scikit-Learn Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset)
- **Samples:** 442 patients
- **Features:** 10 — `age`, `sex`, `bmi`, `bp`, `s1`–`s6` (blood serum measurements); real-valued, scaled
- **Target:** Continuous measure of disease progression one year after baseline (e.g. 25–346)

Data is loaded with `sklearn.datasets.load_diabetes(as_frame=True)` so features and target are in a single pandas DataFrame.

## Objective

Build and compare regression models to predict diabetes progression:

1. **Univariate polynomial regression** — BMI only, degrees 0–5; best model selected by validation R²
2. **Multivariate polynomial regression** — degree 2 and 3, all 10 features (with standardization)
3. **Decision trees** — `max_depth=3` and `max_depth=6`
4. **k-Nearest Neighbors (kNN)** — k=3 and k=7 (features standardized)
5. **Ridge regression** — L2-regularized linear regression
6. **Lasso regression** — L1-regularized linear regression

Evaluation uses **R²**, **Mean Absolute Error (MAE)**, and **Mean Absolute Percentage Error (MAPE)**. Data is split into **train (75%)**, **validation (10%)**, and **test (15%)**.

## Project Structure

| File | Description |
|------|-------------|
| `Multivariate_Linear_Regression.ipynb` | Main notebook: data load, EDA, univariate/multivariate models, comparison, and conclusions |
| `README.md` | This file |

## Notebook Outline

### Part 1 — Data Acquisition and EDA
1. **Get the data** — Load diabetes dataset; document features and sample count
2. **Frame the problem** — Supervised regression; target = disease progression; workshop talking points (Performance Metrics, K-NN, Logistic Regression)
3. **EDA** — Descriptive statistics, histograms per feature, scatter plots (feature vs target), correlation matrix, insights (e.g. BMI association with target)
4. **Data cleaning** — No missing values; numeric, scaled; standardization applied where needed (e.g. for kNN and multivariate polynomial); no further cleaning
5. **Split** — 75% train, 10% validation, 15% test (two-step split)

### Part 2 — Univariate Polynomial Regression (BMI)
6. **Models** — Six polynomial models (degree 0–5) on BMI vs. disease progression
7. **Comparison table** — Train/validation R², MAE, MAPE per model
8. **Best model** — Selected by highest validation R² (e.g. degree 5)
9. **Test evaluation** — R², MAE, MAPE on held-out test set
10. **Plots** — Train, validation, and test points with fitted curve (separate figures)
11. **Equation** — Intercept and coefficients with two-decimal precision
12. **Prediction** — `model.predict()` for a chosen BMI value (e.g. median)
13. **Trainable parameters** — Count per degree via `get_feature_names_out()`
14. **Conclusions** — Performance, overfitting/underfitting, limitations, workshop points (kNN, regularization)

### Part 3 — Multivariate Models (All Features)
- **Preprocessing:** StandardScaler on features for models that need it (polynomial expansions, kNN, Ridge, Lasso)
- **Models:** Polynomial (degree 2, 3), Decision trees (max_depth 3, 6), kNN (k=3, 7), Ridge, Lasso
- **Workflow:** Train/validation metrics table → best model by validation R² → test metrics (R², MAE, MAPE)
- Best multivariate model (e.g. Poly d=2) is evaluated on the test set
- **Final conclusions** — Best models, bias–variance trade-off, where models fail, workshop points, next steps (e.g. cross-validation, feature engineering)

## Requirements

- Python 3.8+
- **scikit-learn** — datasets, models, metrics, train_test_split, StandardScaler, PolynomialFeatures
- **pandas** — DataFrame handling
- **numpy** — arrays and numerics
- **matplotlib** — scatter plots, histograms, fit curves
- **seaborn** — correlation heatmap, boxplots


## create a venv and install requirements.txt file

## How to Run

1. Clone or download this folder.
2. Install dependencies (see above).
3. Open `Multivariate_Linear_Regression.ipynb` in Jupyter Notebook or JupyterLab.
4. Run all cells in order (Kernel → Restart & Run All).

The notebook is structured for readability: markdown sections explain each step; code cells include inline comments.

## Metrics

- **R² (R-squared):** Fraction of variance in the target explained by the model.
- **MAE (Mean Absolute Error):** Average absolute prediction error (same units as target).
- **MAPE (Mean Absolute Percentage Error):** Average absolute percentage error; can be unstable when true values are near zero.

## License

This project is for educational use (e.g. course lab). The Scikit-Learn diabetes dataset is from Bradley Efron et al. and is included in scikit-learn under the BSD license.
