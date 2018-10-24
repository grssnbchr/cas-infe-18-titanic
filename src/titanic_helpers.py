

import pandas as pd
import numpy as np
import datetime


def write_csv(df):
    '''
    Function to generate output file for upload
    :param df with id and survived columns:
    :return:
    '''
    assert isinstance(df, pd.DataFrame)
    filename = "titanic_"+datetime.datetime.now().isoformat().replace(".", "_").replace(":","_") + ".csv"
    df[["id", "survived"]].to_csv(filename, sep=";", header=["key", "value"], index = False)


def prepare_data(train_data):
    #
    # SEX: Prepare data 'sex' and store it in train_data
    # First: get the most frequent gender

    num_female = len(train_data[train_data.sex == 'female'])
    num_male = len(train_data[train_data.sex == 'male'])
    # Set the most frequent gender (female = 0, male = 1)
    most_freq_gender = 0 if num_female >= num_male else 1
    # Second: store gender data in train_data

    def sex_transformation(row):
        if row["sex"] == '':
            return most_freq_gender  # Most freq. gender is used if 'sex' is undefined
        if row["sex"] == 'female':
            return 0
        if row["sex"] == 'male':
            return 1

    train_data["sex"] = train_data.apply(lambda row: sex_transformation(row), axis = 1)

    # AGE: Prepare data 'age' and store it in train_data
    # First: get the median age
    # Convert 'age' to float, empty values to 0

    train_data["age"] = train_data.age.apply(lambda age: 0 if age == '' else float(age))

    median_age = train_data["age"].median(skipna = True )

    #store median age if age is undefined
    train_data["age"] = train_data.age.apply(lambda age: 0 if age == 0 else median_age)


    # FARE: Prepare data 'fare' and store it in train_data
    # First: get the 'fare' and 'pclass' data
    # Convert 'fare' to float, empty values to 0
    train_data["fare"] = train_data.fare.apply(lambda fare: 0 if fare == '' else float(fare))
    median_fare = train_data.groupby(["pclass"])["fare"].median(skipna=True)
    train_data["fare"] = train_data.apply(lambda row: 0 if row["fare"] == 0 else median_fare[row["pclass"]],axis = 1)



    # EMBARKED: Prepare data 'embarked' and store it in train_data
    # First: get the most common 'embarked' value
    embarked_data = train_data.groupby("embarked").size().reset_index(name='N')
    mc_embarked = embarked_data.embarked[embarked_data.N == max(embarked_data.N)]

    # Second: replace empty entries with the most common 'embarked' value
    train_data["embarked"] = train_data.embarked.apply(lambda embarked: mc_embarked if embarked == '' else embarked)

    # Third: convert all 'embarked' values to int

    train_data["embarked"] = pd.Categorical(train_data.embarked)
    train_data["embarked"] = train_data.embarked.cat.codes
    return train_data

