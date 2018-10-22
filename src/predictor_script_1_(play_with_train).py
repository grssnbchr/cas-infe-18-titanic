# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 19:54:41 2014

@author: stdm
"""

import csv as csv 
import numpy as np

# Open up the CSV file in to a Python object
with open('../data/titanic3_train.csv', 'r') as f:
    csv_file_object = csv.reader(f, delimiter=';') 
    header = next(csv_file_object) #next() just skips the first line holding the column headers
    data=[]
    for row in csv_file_object:  #Run through each row in the CSV file, adding each row to the data variable
        data.append(row)

# Then convert from a list to an array 
# (Be aware that each item is currently a string in this format)
data = np.array(data) 
print("Header: ", header)
print("Data: ", data)

# The size() function counts how many elements are in
# the array and sum() (as you would expect) sums up
# the elements in the array.
number_passengers = np.size(data[0::,2].astype(np.float))
number_survived = np.sum(data[0::,2].astype(np.float))
proportion_survivors = number_survived / number_passengers
print("Proportions survived: ", proportion_survivors)

# This finds where all the elements in the gender column equal “female”
women_only_stats = data[0::,5] == "female"
# This finds where all the elements do not equal female (i.e. male)
men_only_stats = data[0::,5] != "female"   

# Using the index from above we select the females and males separately
women_onboard = data[women_only_stats,2].astype(np.float)     
men_onboard = data[men_only_stats,2].astype(np.float)
# Then we finds the proportions of them that survived
proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard) 
# and then print it out
print('Proportion of women who survived is %s' % proportion_women_survived)
print('Proportion of men who survived is %s' % proportion_men_survived)
            
