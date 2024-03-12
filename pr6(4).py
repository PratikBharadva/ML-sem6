# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 11:56:31 2024

@author: student
"""

from sklearn.cluster import KMeans 
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split 

df = pd.read_csv('IRIS.csv') 
df.columns = ['X1', 'X2', 'X3', 'X4', 'Y'] 
df = df.drop(columns=['X4', 'X3'], axis=1) 
df.head() 
kmeans = KMeans(n_clusters=3) 
X = df.values[:, 0:2] 
kmeans.fit(X) 
df['Pred'] = kmeans.predict(X) 
df.head() 
sns.set_context('notebook', font_scale=1.1) 
sns.set_style('ticks') 
sns.lmplot(x='X1',y='X2', scatter=True, fit_reg=False, data=df, hue = 'Pred') 