```
 ██████╗  ██╗  ██████╗
██╔════╝ ███║ ██╔═████╗
██║  ███╗╚██║ ██║██╔██║
██║   ██║ ██║ ████╔╝██║
╚██████╔╝ ██║ ╚██████╔╝
 ╚═════╝  ╚═╝  ╚═════╝
```

# Comparative Effects of Credit Utilization and Credit Limit Adjustments on Predicted Default Risk

> *Does paying down debt or raising a credit limit do more to reduce predicted default probability?*  
> We built five machine learning models to find out.

---

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   DATASET     →   UCI Credit Card Default   (30,000 clients)   │
│   MODELS      →   LR · SVM · RF · XGBoost · MLP                │
│   METHOD      →   Counterfactual Simulation + SHAP Analysis     │
│   COURSE      →   DS6050 · Advanced Machine Learning · UVA      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Question

Financial institutions use credit risk models to predict who will default — but rarely use those same models to ask *what they could do about it*. This project bridges that gap.

We train multiple machine learning classifiers on the [UCI Default of Credit Card Clients](https://doi.org/10.24432/C55S3H) dataset, then simulate two real-world interventions:

```
  Intervention A                    Intervention B
  ──────────────                    ──────────────
  Reduce bill amounts               Increase credit limit
  (client pays down debt)           (lender raises the limit)

        ↓                                  ↓

  Which one moves the predicted default probability more?
```

---

## The Team

```
  Sree Prabhav Bandakavi  ·  Logistic Regression + SVM
  Humaid Billoo           ·  XGBoost + Dataset Analysis  
  Claudio Cela            ·  Random Forest + Methodology
  Jack Thompson Hays      ·  MLP + Data Processing
  
  University of Virginia · School of Data Science · 2026
```

---

## Repository

```
📁 G10
├── 📓 Project_LR.ipynb                               Logistic Regression
├── 📓 Project_SVM.ipynb                              Support Vector Machine  
├── 📓 Random Forest - Counterfactual Analysis.ipynb  Random Forest + SHAP
├── 📓 Xgboost.ipynb                                  XGBoost
├── 📓 MLP.ipynb                                      Neural Network
└── 📊 Working Version - default of credit card clients.xls
```

---

<p align="center">
  <sub>Built for DS6050 · University of Virginia · School of Data Science</sub>
</p>
