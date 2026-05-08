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

    # Separate into numeric and categorical blocks
    cat_cols = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    num_cols = [c for c in X.columns if c not in cat_cols + ['Average of Util Over 6 Months', 'ID']]

    X_num = X[num_cols].values   # numeric block — to be scaled
    X_cat = X[cat_cols].values   # categorical block — to be one-hot encoded

    # Fit OHE on full dataset, transform vocab only, so no target leakage
    cate = OneHotEncoder(sparse_output=False)
    X_cat_encoded = cate.fit_transform(X_cat)

    # Combine numeric (unscaled) and encoded categorical into one matrix
    X_combined = np.hstack([X_num, X_cat_encoded])

    # Split (on raw unscaled numeric + encoded categorical)
    # First split: 90% train+val, 10% test
    x_trainv, x_test, y_trainv, y_test = train_test_split(
        X_combined, Y,
        test_size=0.1,
        stratify=Y,
        random_state=seed
    )

    # Second split: ~70% train, ~20% val (0.222 of 90% ≈ 20% of total)
    x_train, x_v, y_train, y_v = train_test_split(
        x_trainv, y_trainv,
        test_size=0.222,
        stratify=y_trainv,
        random_state=seed
    )

    # Fit scaler on training data ONLY, transform all splits
    # The scaler learns mean/std from training rows only to prevent leakage.
    n_num = X_num.shape[1]  

    scaler = StandardScaler()
    x_train[:, :n_num] = scaler.fit_transform(x_train[:, :n_num])
    x_v[:, :n_num]     = scaler.transform(x_v[:, :n_num])
    x_test[:, :n_num]  = scaler.transform(x_test[:, :n_num])

    # OHE columns are already encoded and need no further scaling

    # Convert to tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train,          dtype=torch.float32).view(-1, 1)

    x_v     = torch.tensor(x_v,     dtype=torch.float32)
    y_v     = torch.tensor(y_v,              dtype=torch.float32).view(-1, 1)

    x_test  = torch.tensor(x_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,           dtype=torch.float32).view(-1, 1)

    return x_v, y_v, x_train, y_train, x_test, y_test
