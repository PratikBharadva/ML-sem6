# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:42:34 2024

@author: student
"""

#iterate through list using numeric indexing
li = [1,2,3,4]
for i in range(len(li)):
    print(li[i])
    
#find repeated items in a tuple
tup = (1,2,3,1,5,2)

for i in range(1, len(tup)):
    
    for j in range(0,i):
        
        if tup[i] == tup[j]:
            print(tup[i], "is repeated in tuple")
            
#calculate product using function,multiplying all the numbers of a given tuple
def multiply(tuple1):
    m = 1
    for i in tuple1:
        m *= i
    return m

ans=multiply((2,3,4))
print("Multiplication:",ans)        