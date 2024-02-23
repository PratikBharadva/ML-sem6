# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 11:38:46 2024

@author: student
"""

import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt

#importing dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

#Splitting the dataset into the trainingset and testset
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size=1/3,random_state=0)

#Fitting Simple linear regression to the training set
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(Xtrain,Ytrain)

#Predicting the testset results
Ypred = reg.predict(Xtrain)

#visualing the training set results
plt.scatter(Xtrain,Ytrain,color="green")
plt.plot(Xtrain,Ypred)
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualing the test set results
plt.scatter(Xtest,Ytest,color="green")
plt.plot(Xtrain,Ypred,color="red")
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()