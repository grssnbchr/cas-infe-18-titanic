# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 20:30:17 2014

@author: stdm
"""

import csv as csv 

# Open up the CSV file into a Python object
test_file = open('../../data/titanic3_test.csv', 'r')
test_file_object = csv.reader(test_file, delimiter=';') 
header = next(test_file_object)

prediction_file = open("submission1_genderbased.csv", "w")
prediction_file_object = csv.writer(prediction_file, delimiter=';')
    
prediction_file_object.writerow(["key", "value"])
for row in test_file_object:
    if row[4] == 'female':                                     
        prediction_file_object.writerow([row[0],'1']) # predict 1
    else: #it is male
        prediction_file_object.writerow([row[0],'0']) # predict 0
test_file.close()
prediction_file.close()

print("Done.")
