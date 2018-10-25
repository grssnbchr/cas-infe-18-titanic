# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 20:54:42 2014
Refactored Mon Oct 16 15:43:11 2017

@author: stdm / spio
"""

import csv as csv 
import numpy as np

# Open up the CSV file in to a Python object
with open('../data/titanic3_train.csv', 'r') as f:
    csv_file_object = csv.reader(f, delimiter=';') 
    header = next(csv_file_object) #next() just skips the first line holding the column headers
    data=[]
    for row in csv_file_object:  #Run through each row in the csv file, adding each row to the data variable
        data.append(row)

# Then convert from a list to an array 
# (Be aware that each item is currently a string in this format)
data = np.array(data) 
print("Header: ", header)


#fill up empty fare values with the mean of the corresponding pclass
#1. calculate mean fare of each pclass
fare_mean_per_pclass = []
for i in range(len(np.unique(data[0::,1]))): #loop over all possible pclass's (i == pclass-1)
    fare_mean_per_pclass.append(0) #initialize ith pclass fare mean
    cnt = 0 #number of tickets with a value in pclass i
    for j in range(len(data[0::,10])):
        if data[j, 1].astype(np.int) == i+1:        
            try: #try to cast the jth fare value as float
               fare_mean_per_pclass[i] += data[j, 10].astype(np.float)
               cnt += 1
            except: #if it can't be casted to float: ignore it
                fare_mean_per_pclass[i] += 0 #just ignore empty values
    if cnt > 0: fare_mean_per_pclass[i] /= float(cnt) #calculate mean
print("Mean fare per pclass: ", list(enumerate(fare_mean_per_pclass, 1)))

#2. replace the empty values with the pclass mean
for i in range(len(data[0::,10])):
    try:
        test = float(data[i, 10]) #if this works, all is well with row i
    except ValueError:
        current_pclass = data[i, 1].astype(np.int)
        data[i, 10] = fare_mean_per_pclass[current_pclass-1] 


# So we add a ceiling... all ticket prices of 40 and above count as category 30..39
fare_ceiling = 40
# then modify the data in the Fare column to =39, if it is greater or equal to the ceiling
data[ data[0::,10].astype(np.float) >= fare_ceiling, 10 ] = fare_ceiling - 1.0

fare_bracket_size = 10
number_of_price_brackets = fare_ceiling // fare_bracket_size

# I know there were 1st, 2nd and 3rd classes on board
number_of_classes = 3
# But it's better practice to calculate this from the data directly
# Take the length of an array of unique values in column index 1
number_of_classes = len(np.unique(data[0::,1])) 

# Initialize the survival table [dimensions 3x4] with all zeros
survival_table = np.zeros((2, number_of_classes, number_of_price_brackets))


for i in range(number_of_classes): #loop through each class
    for j in range(number_of_price_brackets): #loop through each price bin
        #Which element is a female, 
        #and was ith class, 
        #was greater than this bin, 
        #and less than the next bin 
        #-> give the the 3rd col (survived)
        women_only_stats = data[ \
                                (data[0::,5] == "female") \
                                &(data[0::,1].astype(np.float) == i+1) \
                                &(data[0:,10].astype(np.float) >= j*fare_bracket_size) \
                                &(data[0:,10].astype(np.float) < (j+1)*fare_bracket_size) \
                                , 2]

        #Which element is a male,      
        #and was ith class,
        #was greater than this bin,
        #and less than the next bin
        men_only_stats = data[ \
                              (data[0::,5] != "female") \
                              &(data[0::,1].astype(np.float) == i+1) \
                              &(data[0:,10].astype(np.float) >= j*fare_bracket_size) \
                              &(data[0:,10].astype(np.float) < (j+1)*fare_bracket_size) \
                              , 2] 
        
        survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float)) 
        survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))
        survival_table[ survival_table != survival_table ] = 0.
                              
print("Survival table:\n", survival_table)

survival_table[ survival_table < 0.5 ] = 0
survival_table[ survival_table >= 0.5 ] = 1 
print("Survival table, rectified:\n", survival_table)


#open the test set and a new submission file
test_file = open('../../data/titanic3_test.csv', 'r')
test_file_object = csv.reader(test_file, delimiter=';')
header = next(test_file_object)
predictions_file = open("submission2_genderclassbased.csv", "w")
p = csv.writer(predictions_file, delimiter=';')
p.writerow(["key", "value"])


# We are going to loop through each passenger in the test set         
for row in test_file_object:        
    # For each passenger we loop through each price bin  
    for j in range(number_of_price_brackets):  
        try: # Some passengers have no fare data so try to make...
            row[9] = float(row[9]) # a float
        except: # If fails: no data, so... 
            bin_fare = 3 - float(row[1]) # bin the fare according to pclass
            break 
        if row[9] > fare_ceiling: # If there is data see if it is greater than fare ceiling we set earlier
            bin_fare = number_of_price_brackets-1 # If so set to highest bin
            break
        # If passed these tests then loop through each bin  
        if row[9] >= j * fare_bracket_size \
           and row[9] < (j+1) * fare_bracket_size: 
              bin_fare = j # If passed these tests then assign index
              break                   
    if row[4] == 'female': #If the passenger is female
        p.writerow([row[0], "%d" % int(survival_table[0, int(row[1])-1, bin_fare])])
    else: #passenger is male
        p.writerow([row[0], "%d" % int(survival_table[1, int(row[1])-1, bin_fare])])
     
# Close out the files.
test_file.close() 
predictions_file.close()
