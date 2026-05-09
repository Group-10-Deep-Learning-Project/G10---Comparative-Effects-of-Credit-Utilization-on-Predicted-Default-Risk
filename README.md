```
 ██████╗  ██╗  ██████╗
██╔════╝ ███║ ██╔═████╗
██║  ███╗╚██║ ██║██╔██║
██║   ██║ ██║ ████╔╝██║
╚██████╔╝ ██║ ╚██████╔╝
 ╚═════╝  ╚═╝  ╚═════╝
```

# Comparative Effects of Credit Utilization and Credit Limit Adjustments on Predicted Default Risk

> *How do balance reductions, credit-limit increases, and utilization-controlled limit changes affect modeled credit-card default risk?*  
> We trained five machine learning models, interpreted their predictions with SHAP, and tested structured counterfactual interventions.

---

```
┌────────────────────────────────────────────────────────────────────┐
│                                                                    │
│   DATASET      →   UCI Credit Card Default   (30,000 clients)      │
│   FEATURES     →   29 raw engineered features → 97 encoded inputs  │
│   MODELS       →   LR · SVM · RF · XGBoost · MLP                   │
│   METHODS      →   Multi-seed evaluation · SHAP · Counterfactuals  │
│   COURSE       →   DS6050 · Advanced Machine Learning · UVA        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## Project Question

Financial institutions use credit risk models to predict who will default, but those same models can also be used to ask how predictions respond to structured financial changes. This project studies how predicted default probability changes under three counterfactual interventions:

```
  Intervention A                    Intervention B                    Intervention C
  ──────────────                    ──────────────                    ──────────────
  Reduce bill amounts               Increase credit limit             Increase credit limit
  Hold credit limit fixed           Hold bill amounts fixed           and bill amounts together
                                                                        Hold utilization constant

        ↓                                  ↓                                  ↓

  Model response to                 Model response to                 Model response to
  balance paydown                   limit increases                   pure limit/raw-balance change
```

These simulations describe **model behavior**, not causal effects. They show how trained models respond to feature perturbations, not how real borrowers would necessarily respond to financial interventions.

---

## Dataset and Preprocessing

The project uses the [UCI Default of Credit Card Clients](https://doi.org/10.24432/C55S3H) dataset, which contains 30,000 credit card clients in Taiwan with demographic, repayment-history, bill-statement, payment-amount, and credit-limit variables.

Key preprocessing steps:

- Verified a default rate of approximately **22.12%**.
- Verified no missing values across the original variables.
- Engineered six monthly utilization ratios:

```math
\text{Utilization}_t = \frac{\text{BILL\_AMT}_t}{\text{LIMIT\_BAL}}, \quad t = 1,2,\dots,6
```

- Treated repayment status variables as categorical because undocumented values `-2` and `0` are common and do not follow a simple ordinal risk pattern.
- One-hot encoded repayment status, education, marital status, and sex variables.
- Final feature representation: **29 raw engineered features** expanded to **97 encoded model input features**.

---

## Models

Five classification models were trained and evaluated using a stratified **70/20/10 train-validation-test split** across three random seeds: **42, 123, and 456**.

| Model | Role |
|---|---|
| Logistic Regression | Linear baseline |
| SVM with RBF kernel | Nonlinear baseline |
| Random Forest | Main interpretable model for SHAP and counterfactual analysis |
| XGBoost | High-performing boosted-tree benchmark |
| MLP | Neural-network benchmark |

---

## Classification Performance

Results are reported as mean ± standard deviation across three random seeds.

| Model | AUC-ROC | F1 | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.763 ± 0.012 | 0.521 ± 0.007 | 0.787 ± 0.015 | 0.522 ± 0.035 | 0.525 ± 0.047 |
| SVM | 0.726 ± 0.008 | 0.489 ± 0.020 | 0.793 ± 0.009 | 0.542 ± 0.030 | 0.450 ± 0.056 |
| XGBoost | 0.778 ± 0.009 | 0.536 ± 0.012 | 0.783 ± 0.008 | 0.510 ± 0.018 | 0.566 ± 0.032 |
| Random Forest | 0.777 ± 0.012 | 0.537 ± 0.015 | 0.783 ± 0.010 | 0.510 ± 0.020 | 0.568 ± 0.034 |
| MLP | 0.777 ± 0.012 | 0.528 ± 0.023 | 0.769 ± 0.018 | 0.482 ± 0.030 | 0.584 ± 0.013 |

**Summary:** Random Forest, XGBoost, and MLP performed comparably across seeds. XGBoost had the highest average AUC-ROC, while Random Forest had the highest average F1 score. Random Forest was selected for the main counterfactual analysis because it performed comparably to XGBoost and MLP while supporting interpretable SHAP-based analysis.

---

## SHAP Feature Importance

Average top feature rankings across models showed that repayment behavior and utilization-related variables were consistently influential.

| Rank | SVM | Random Forest | XGBoost | MLP |
|---:|---|---|---|---|
| 1 | Feature_35 | PAY_0_0 | PAY_0_2 | Feature_35 |
| 2 | Feature_46 | PAY_0_2 | PAY_2_2 | Feature_37 |
| 3 | Feature_0 | PAY_2_2 | PAY_3_2 | Feature_34 |
| 4 | Feature_37 | PAY_3_2 | PAY_4_2 | Feature_36 |
| 5 | Feature_89 | PAY_AMT1 | PAY_AMT1 | Feature_45 |

Feature labels such as `Feature_X` correspond to encoded repayment-status, payment-history, and engineered utilization variables generated during preprocessing.

---

## Counterfactual Intervention Results

The main counterfactual analysis used the Random Forest model and measured mean absolute change in predicted default probability across the 3,000-client test set.

| Intervention | 10% | 25% | 50% |
|---|---:|---:|---:|
| A: Reduce Bills | 0.078 ± 0.018 | 0.077 ± 0.018 | 0.073 ± 0.017 |
| B: Increase Limit | 0.076 ± 0.016 | 0.075 ± 0.016 | 0.073 ± 0.016 |
| C: Limit Increase, Utilization Constant | 0.078 ± 0.017 | 0.080 ± 0.018 | 0.082 ± 0.019 |

**Interpretation:** Interventions A and B produced similar model-response magnitudes. Intervention C also produced comparable changes, indicating that the Random Forest model remains sensitive to raw credit-limit and bill-amount changes even when utilization is held approximately constant. Because these are absolute changes, larger values mean larger model response, not necessarily larger reductions in predicted default probability.

---

## Client Segmentation

Directional changes were evaluated at the 25% intervention level for high-risk and low-risk clients under the Random Forest model.

| Group | n | Baseline P | Intervention A | Intervention B |
|---|---:|---|---:|---:|
| High-Risk | 640 | > 0.5 | -0.0636 | -0.0530 |
| Low-Risk | 1894 | < 0.4 | 0.0401 | 0.0510 |

**Interpretation:**

- For high-risk clients, both interventions reduced predicted default probability on average, with balance reduction producing the larger decrease.
- For low-risk clients, both interventions increased predicted default probability on average, with credit-limit increases producing the larger increase.
- This suggests that the model responds differently to the same intervention depending on baseline risk group.

---

## Key Takeaways

1. **No single nonlinear model dominated.** Random Forest, XGBoost, and MLP achieved similar multi-seed performance.
2. **Repayment history was the strongest driver of default prediction.** SHAP rankings consistently emphasized repayment-status and payment-history features.
3. **Credit utilization and credit-limit features still mattered.** Counterfactual simulations showed measurable model responses to balance and limit changes.
4. **Intervention effects were heterogeneous.** High-risk and low-risk clients responded differently in the Random Forest segmentation analysis.
5. **Results are model-behavior findings, not causal claims.** The counterfactuals show how trained models respond to feature changes, not how real-world default risk would necessarily change.

---

## Repository

```
 📁 G10
 ├── 📁 Images                                      Figures used in paper/README
 ├── 📁 models                                      Saved trained model files
 │   ├── rf_model_seed42.joblib                     Random Forest model, seed 42
 │   ├── rf_model_seed123.joblib                    Random Forest model, seed 123
 │   ├── rf_model_seed456.joblib                    Random Forest model, seed 456
 │   ├── xgb_model_seed42.joblib                    XGBoost model, seed 42
 │   ├── xgb_model_seed123.joblib                   XGBoost model, seed 123
 │   └── xgb_model_seed456.joblib                   XGBoost model, seed 456
 ├── 📄 README.md                                   Project overview and results summary
 ├── 📄 LICENSE                                     Project license
 ├── 🐍 Preprocessing.py                            Data cleaning, feature engineering, and encoding
 ├── 🐍 LR.py                                       Logistic Regression model
 ├── 🐍 SVM.py                                      Support Vector Machine model
 ├── 🐍 RF.py                                       Random Forest model
 ├── 🐍 XGBoost.py                                  XGBoost model
 ├── 🐍 MLP.py                                      Multilayer Perceptron model
 ├── 🐍 counterfactual.py                           Counterfactual intervention analysis
 ├── 🐍 runner.py                                   Main script to run preprocessing, models, and outputs
 ├── 📊 counterfactual_summary.csv                  Counterfactual intervention outputs
 ├── 📊 segmentation_summary.csv                    Risk-group segmentation outputs
 ├── 📊 Working Version - default of credit card clients.xls
 └── 📦 default+of+credit+card+clients.zip          Original/raw dataset archive
```
---

## Team

```
Sree Prabhav Bandakavi  ·  Logistic Regression + SVM
Humaid Billoo           ·  XGBoost + Dataset Analysis
Claudio Cela            ·  Random Forest + Methodology
Jack Thompson Hays      ·  MLP + Data Processing

University of Virginia · School of Data Science · 2026
```

---

<p align="center">
  <sub>Built for DS6050 · University of Virginia · School of Data Science</sub>
</p>
