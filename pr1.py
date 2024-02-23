# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 10:27:59 2024

@author: student
"""
#1.1
num = int(input("Enter any int number:"))
if num>0:
    print("Number is positive")
elif num<0:
    print("Number is negative")
else:
    print("Number is 0")
    
#1.2: sum of list values
li=[1,2,3]
num = int(input("Enter number to append:"))
li.append(num)
sum=0
for i in li:
    sum+=i
print(sum, "is the sum of list items")

