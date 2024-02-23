# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 10:56:57 2024

@author: student
"""

from pandas import read_csv
# sample points  
#X = [0, 6, 11, 14, 22] 
#Y = [1, 7, 12, 15, 21]
filename = '.\Salary_Data.csv'
dataframe =  read_csv(filename)
array = dataframe.values
X = array[:,0]
Y = array[:,1]


# solve for a and b 
def best_fit(X, Y): 
 
    xbar = sum(X)/len(X) 
    ybar = sum(Y)/len(Y) 
    n = len(X) # or len(Y) 
 
    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar 
    denum = sum([xi**2 for xi in X]) - n * xbar**2 
 
    beta1 = numer / denum 
    beta0 = ybar - beta1 * xbar 
 
    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(beta0, beta1)) 
 
    return beta0, beta1 

# solution 
a, b = best_fit(X, Y) 
#best fit line: 
#y = 0.80 + 0.92x 
 
# plot points and fit line 
import matplotlib.pyplot as plt 
plt.scatter(X, Y) 
yfit = [a + b * xi for xi in X] 
plt.plot(X, yfit) 
plt.show() 
 
#best fit line: 
#y = 1.48 + 0.92x 
