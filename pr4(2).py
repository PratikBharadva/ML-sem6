# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 12:24:38 2024

@author: student
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
#import matplotlib.pyplot as plt

df = pd.read_csv("../3CP10/50_Startups.csv")

X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
Y = df['Profit']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, Y, test_size=1/3, random_state=0)

regr = LinearRegression()
regr.fit(Xtrain, Ytrain)

Ypred = regr.predict(Xtrain)

print(Ypred)