# %%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import pandas as pd 
import numpy as np
import random


def valuesWithSeed(seed):
    
    #Seed setting
    random.seed(seed)
    np.random.seed(seed)

    ##Read in DataFrame
    df = pd.read_excel('Working Version - default of credit card clients.xls',header=1)
    df = pd.DataFrame(df)

    X = df.loc[:,df.columns != 'default payment next month'].copy()
    X = pd.DataFrame(X)
    Y = df.loc[:,'default payment next month'].values.copy()


    #To be one hot encoded
    XC = X.loc[:,['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].values
    XC

    #To be scaled normallly -> N(0,1)
    X = X.loc[:, ~X.columns.isin(['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6', 
    'Average of Util Over 6 Months', 
    'ID'])].values

    #Scaling + One Hot Encoding 
    scaler = StandardScaler()
    cate = OneHotEncoder()
    X = scaler.fit_transform(X)
    XC = cate.fit_transform(XC)

    #Recombination
    XC = XC.toarray()
    X_combined = np.hstack([X, XC])

    #Test Train Split
    x_trainv, x_test, y_trainv, y_test = train_test_split(X_combined, Y, test_size=0.1,stratify=Y, random_state = 28)

    x_train, x_v, y_train, y_v = train_test_split(x_trainv, y_trainv, test_size=0.222, stratify=y_trainv, random_state = 28)

    x_v = torch.tensor(x_v,dtype=torch.float32)
    y_v = torch.tensor(y_v,dtype=torch.float32).view(-1, 1)

    x_train = torch.tensor(x_train,dtype=torch.float32)
    y_train = torch.tensor(y_train,dtype=torch.float32).view(-1, 1)

    x_test = torch.tensor(x_test,dtype=torch.float32)
    y_test = torch.tensor(y_test,dtype=torch.float32).view(-1, 1)

    return x_v, y_v, x_train, y_train, x_test, y_test
