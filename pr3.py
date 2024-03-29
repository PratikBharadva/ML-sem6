# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 11:08:04 2024

@author: student
"""

#Preprocessing Techniques
from pandas import read_csv
from numpy import set_printoptions
filename = 'diabetes.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = read_csv(filename, names=names)
array = dataframe.values
 
# separate array into input and output components
X = array[1:,0:8]
Y = array[:,8]
"""
#1.Rescaling
print("Rescaling:")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 2))
rescaledX = scaler.fit_transform(X)
 
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:6,:])

#2.Standardization
print("StandardScale:")
from sklearn.preprocessing import StandardScale
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X);
 
# summarize transformed data
set_printoptions(precision=3)
print(rescaledX[0:6,:])

#3.Normalization
print("Normalizer:")
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X)
normalX = scaler.transform(X)

set_printoptions(precision=3)
print(normalX[0:6,:])

#4.Binarization
print("binarizer:")
from sklearn.preprocessing import Binarizer
binarizer = Binarizer(threshold=0.0).fit(X)
binaryX = binarizer.transform(X)
# summarize transformed data
set_printoptions(precision=3)
print(binaryX[0:5,:])
"""

#5.One Hot Encoding
print("One Hot Encoding:")
from sklearn import preprocessing
encoder = preprocessing.OneHotEncoder() 
encoder.fit([[0, 2, 1, 12], [1, 3, 5, 3], [2, 3, 2, 12], [1, 2, 4, 3]]) 
encoded_vector = encoder.transform([[2, 3, 2, 12]]).toarray() 
print ("\nEncoded vector =", encoded_vector)

#6.Label Encoding
print("Label encoding")
label_encoder = preprocessing.LabelEncoder() 
input_classes = ['suzuki', 'ford', 'suzuki', 'toyota', 'ford', 'bmw','tata'] 
label_encoder.fit(input_classes) 
print ("\nClass mapping:") 
for i, item in enumerate(label_encoder.classes_): 
     print (item, '-->', i) 