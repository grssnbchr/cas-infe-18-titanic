# -*- coding: utf-8 -*-
'''
Created on Thu Dec 18 15:30:26 2014

@author: stdm
'''

import csv as csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# PART 1: PREPARING THE TRAINING DATA

# Open up the CSV file into a Python object
with open('../../data/titanic3_train.csv', 'r') as f:     # Load the training file
    csv_file_object = csv.reader(f, delimiter=';')
    next(csv_file_object)      # next() skips the first line holding the column headers
    orig_train_data = []
    for row in csv_file_object:  # Run through each row in the csv, add it to the data variable
        orig_train_data.append(row)

# Then convert from a list to an array
# (Be aware that each item is currently a string in this format)
orig_train_data = np.array(orig_train_data)

# Overview of the training data
# DATA      ORIG.INDEX
# id        0
# pclass    1
# survived  2
# name      3
# surname   4
# sex       5
# age       6
# sibsp     7
# parch     8
# ticket    9
# fare      10
# cabin     11
# embarked  12
# boat      13
# body      14
# home.dest 15

# Prepare the data structure used for training
# DATA USED FOR TRAINING    ORIG.INDEX      TRAIN.INDEX
# survived                  2               0
# pclass                    1               1
# sibsp                     7               2
# parch                     8               3
# sex                       5               4
# age                       6               5
# fare                      10              6
# embarked                  12              7

rows = len(orig_train_data[0::, 0])     # Number of rows in the training data
cols = 8                                # Number of columns in the training data
train_data = np.zeros((rows, cols))  # Array to store the data used for training
# SURVIVED: Store data 'survived' in train_data
train_data[0::, 0] = orig_train_data[0::, 2].astype(np.float)
# PCLASS: Store data 'pclass' in train_data
train_data[0::, 1] = orig_train_data[0::, 1].astype(np.float)
# SIBSP: Store data 'sibsp' in train_data
train_data[0::, 2] = orig_train_data[0::, 7].astype(np.float)
# PARCH: Store data 'parch' in train_data
train_data[0::, 3] = orig_train_data[0::, 8].astype(np.float)

# SEX: Prepare data 'sex' and store it in train_data
# First: get the most frequent gender
gender_data = orig_train_data[0::, 5]
num_female = sum(gender_data == 'female')
num_male = sum(gender_data == 'male')
# Set the most frequent gender (female = 0, male = 1)
most_freq_gender = 0 if num_female >= num_male else 1
# Second: store gender data in train_data
for i, sex in enumerate(gender_data):
    if sex == '':
        train_data[i, 4] = most_freq_gender  # Most freq. gender is used if 'sex' is undefined
    if sex == 'female':
        train_data[i, 4] = 0
    if sex == 'male':
        train_data[i, 4] = 1


# AGE: Prepare data 'age' and store it in train_data
# First: get the median age
# Convert 'age' to float, empty values to 0
age_data = [0 if age == '' else float(age) for age in orig_train_data[0::, 6]]
median_age = np.nanmedian(age_data)
# Second: store age data in train_data
# Alternative: train_data[0::, 5] = [median_age if age == 0 else age for age in age_data]
for i, age in enumerate(age_data):
    if age == 0:
        train_data[i, 5] = median_age    # Most freq. age is used if 'age' is undefined
    else:
        train_data[i, 5] = age

# FARE: Prepare data 'fare' and store it in train_data
# First: get the 'fare' and 'pclass' data
# Convert 'fare' to float, empty values to 0
fare_data = [0 if fare == '' else float(fare) for fare in orig_train_data[0::, 10]]
fare_data = np.array(fare_data)     # Convert from a list to an array
pclass_data = train_data[0::, 1]    # Get the 'pclass' values from train_data
pclass_data_unique = list(enumerate(np.unique(pclass_data)))  # Get the unique 'pclass' values
# Second: replace fares with value 0 with the median fare of the corresponding 'pclass'
for i, unique_pclass in pclass_data_unique:
    # Get array of fares corresponding to the current pclass
    pclass_fare = fare_data[pclass_data == unique_pclass]
    # Calculate the median of the previously received fares
    median_fare = np.nanmedian(pclass_fare)
    pclass_fare[pclass_fare == 0] = median_fare  # Replace fares with value 0 with median fare
    fare_data[pclass_data == unique_pclass] = pclass_fare
# Third: store fare data in train_data
train_data[0::, 6] = fare_data[0::]

# EMBARKED: Prepare data 'embarked' and store it in train_data
# First: get the most common 'embarked' value
embarked_data = list(orig_train_data[0::, 12])
mc_embarked = max(set(embarked_data), key=embarked_data.count)
# Second: replace empty entries with the most common 'embarked' value
embarked_data = [mc_embarked if embarked == '' else embarked for embarked in embarked_data]
embarked_data = np.array(embarked_data)
# Third: convert all 'embarked' values to int
# Get the unique 'embarked' values
embarked_data_unique = list(enumerate(np.unique(embarked_data)))
for i, unique_embarked in embarked_data_unique:
    embarked_data[embarked_data == unique_embarked] = i
# Fourth: store embarked data in train_data
train_data[0::, 7] = embarked_data[0::].astype(np.float)


# PART 2: PREPARING THE TEST DATA

# Now we have to do the same for the test data as we did for the training data

# Open up the CSV file into a Python object
with open('../../data/titanic3_test.csv', 'r') as f:     # Load the test file
    csv_file_object = csv.reader(f, delimiter=';')
    next(csv_file_object)   # next() skips the first line holding the column headers
    orig_test_data = []
    for row in csv_file_object:  # Run through each row in the csv, add it to the data variable
        orig_test_data.append(row)

# Then convert from a list to an array
# (Be aware that each item is currently a string in this format)
orig_test_data = np.array(orig_test_data)

# Overview of the test data
# DATA      ORIG.INDEX
# id        0
# pclass    1
# name      2
# surname   3
# sex       4
# age       5
# sibsp     6
# parch     7
# ticket    8
# fare      9
# cabin     10
# embarked  11
# boat      12
# body      13
# home.dest 14

# Prepare the data structure used for testing
# DATA USED FOR TESTING    ORIG.INDEX      TEST.INDEX
# pclass                    1               0
# sibsp                     6               1
# parch                     7               2
# sex                       4               3
# age                       5               4
# fare                      9               5
# embarked                  11              6

rows = len(orig_test_data[0::, 0])      # Number of rows in the test data
cols = 7                                # Number of columns in the test data
test_data = np.zeros((rows, cols))  # Array to store the data used for testing
# PCLASS: Store data 'pclass' in test_data
test_data[0::, 0] = orig_test_data[0::, 1].astype(np.float)
# SIBSP: Store data 'sibsp' in test_data
test_data[0::, 1] = orig_test_data[0::, 6].astype(np.float)
# PARCH: Store data 'parch' in test_data
test_data[0::, 2] = orig_test_data[0::, 7].astype(np.float)

# SEX: Prepare data 'sex' and store it in test_data
# First: get the most frequent gender
gender_data = orig_test_data[0::, 4]
num_female = sum(gender_data == 'female')
num_male = sum(gender_data == 'male')
# Set the most frequent gender (female = 0, male = 1)
most_freq_gender = 0 if num_female >= num_male else 1
# Second: store gender data in test_data
for i, sex in enumerate(gender_data):
    if sex == '':
        test_data[i, 3] = most_freq_gender  # Most freq. gender is used if 'sex' is unde-fined
    if sex == 'female':
        test_data[i, 3] = 0
    if sex == 'male':
        test_data[i, 3] = 1

# AGE: Prepare data 'age' and store it in test_data
# First: get the median age
# Convert 'age' to float, empty values to 0
age_data = [0 if age == '' else float(age) for age in orig_test_data[0::, 5]]
median_age = np.nanmedian(age_data)
# Second: store age data in test_data
# Alternative: test_data[0::, 5] = [median_age if age == 0 else age for age in age_data]
for i, age in enumerate(age_data):
    if age == 0:
        test_data[i, 4] = median_age    # Most freq. age is used if 'age' is undefined
    else:
        test_data[i, 4] = age

# FARE: Prepare data 'fare' and store it in test_data
# First: get the 'fare' and 'pclass' data
# Convert 'fare' to float, empty values to 0
fare_data = [0 if fare == '' else float(fare) for fare in orig_test_data[0::, 9]]
fare_data = np.array(fare_data)    # Convert from a list to an array
pclass_data = test_data[0::, 0]    # Get the 'pclass' values from test_data
pclass_data_unique = list(enumerate(np.unique(pclass_data)))  # Get the unique 'pclass' values
# Second: replace fares with value 0 with the median fare of the corresponding 'pclass'
for i, unique_pclass in pclass_data_unique:
    # Get array of fares corresponding to the current pclass
    pclass_fare = fare_data[pclass_data == unique_pclass]
    median_fare = np.nanmedian(pclass_fare)
    pclass_fare[pclass_fare == 0] = median_fare  # Replace fares with value 0 with median fare
    fare_data[pclass_data == unique_pclass] = pclass_fare
# Third: store fare data in test_data
test_data[0::, 5] = fare_data[0::]

# EMBARKED: Prepare data 'embarked' and store it in test_data
# First: get the most common 'embarked' value
embarked_data = list(orig_test_data[0::, 11])
mc_embarked = max(set(embarked_data), key=embarked_data.count)
# Second: replace empty entries with the most common 'embarked' value
embarked_data = [mc_embarked if embarked == '' else embarked for embarked in embarked_data]
embarked_data = np.array(embarked_data)
# Third: convert all 'embarked' values to int
# Get the unique 'embarked' values
embarked_data_unique = list(enumerate(np.unique(embarked_data)))
for i, unique_embarked in embarked_data_unique:
    embarked_data[embarked_data == unique_embarked] = i
# Fourth: store embarked data in test_data
test_data[0::, 6] = embarked_data[0::].astype(np.float)

# After the training and test data is created, collect the test data's ids
test_ids = orig_test_data[0::, 0]


# PART 3: TRAINING AND PREDICTION

# The data is now ready to go. So lets fit to the train, then predict to the test!
print('Training...')
forest = RandomForestClassifier(n_estimators=100)
# Build a forest of trees from the training set (X, y)
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

print('Predicting...')
output = forest.predict(test_data).astype(np.float)

# Write the data into a file
predictions_file = open('submission3_randomforest.csv', 'w')
open_file_object = csv.writer(predictions_file, delimiter=';')
open_file_object.writerow(['key', 'value'])
open_file_object.writerows(list(zip(test_ids, output)))
predictions_file.close()
print('Done.')
