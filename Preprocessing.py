# %%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import random


def valuesWithSeed(seed):

    # Seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Read in DataFrame
    df = pd.read_excel('Working Version - default of credit card clients.xls', header=1)
    df = pd.DataFrame(df)

    X = df.loc[:, df.columns != 'default payment next month'].copy()
    Y = df.loc[:, 'default payment next month'].values.copy()

    # Separate into numeric and categorical blocks BEFORE splitting
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    num_cols = [c for c in X.columns if c not in cat_cols + ['Average of Util Over 6 Months', 'ID']]

    X_num = X[num_cols].values   # numeric block — to be scaled
    X_cat = X[cat_cols].values   # categorical block — to be one-hot encoded

    # Split first (on raw unscaled data)
    # Scaler and encoder are fit on training data only, then applied to val/test to avoid data leakage

    # First split: 90% train+val, 10% test
    Xn_trainv, Xn_test, Xc_trainv, Xc_test, y_trainv, y_test = train_test_split(
        X_num, X_cat, Y,
        test_size=0.1,
        stratify=Y,
        random_state=seed
    )

    # Second split: ~70% train, ~20% val (0.222 of 90% ≈ 20% of total)
    Xn_train, Xn_v, Xc_train, Xc_v, y_train, y_v = train_test_split(
        Xn_trainv, Xc_trainv, y_trainv,
        test_size=0.222,
        stratify=y_trainv,
        random_state=seed
    )

    # Fit scaler and encoder on training data ONLY
    scaler = StandardScaler()
    cate   = OneHotEncoder(sparse_output=False)

    # Fit on train, transform train/val/test
    Xn_train = scaler.fit_transform(Xn_train)
    Xn_v     = scaler.transform(Xn_v)
    Xn_test  = scaler.transform(Xn_test)

    Xc_train = cate.fit_transform(Xc_train)
    Xc_v     = cate.transform(Xc_v)
    Xc_test  = cate.transform(Xc_test)

    # Recombine numeric and categorical blocks 
    X_train_combined = np.hstack([Xn_train, Xc_train])
    X_v_combined     = np.hstack([Xn_v,     Xc_v])
    X_test_combined  = np.hstack([Xn_test,  Xc_test])

    # Convert to tensors 
    x_train = torch.tensor(X_train_combined, dtype=torch.float32)
    y_train = torch.tensor(y_train,          dtype=torch.float32).view(-1, 1)

    x_v     = torch.tensor(X_v_combined,     dtype=torch.float32)
    y_v     = torch.tensor(y_v,              dtype=torch.float32).view(-1, 1)

    x_test  = torch.tensor(X_test_combined,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,           dtype=torch.float32).view(-1, 1)

    return x_v, y_v, x_train, y_train, x_test, y_test
