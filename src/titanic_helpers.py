

import pandas as pd
import numpy as np
import datetime
# package name is requests-html
from requests_html import HTMLSession
import re


def get_datetime():
    '''
    Returns formatted datetime string for submission
    :return datetime as string:
    '''
    return (
            datetime
            .datetime
            .now()
            .isoformat()
            .replace(".", "_")
            .replace(":", "_")
            )


def write_csv(df):
    '''
    Function to generate output file for upload
    :param df: df with id and survived columns:
    :return:
    '''
    assert isinstance(df, pd.DataFrame)
    filename = "titanic_" + get_datetime() + ".csv"
    df[["id", "survived"]].to_csv(filename,
                                  sep=";",
                                  header=["key", "value"],
                                  index=False)


def submit_answer(df, custom_name='submission'):
    '''
    Function to automagically submit solution df to submission website
    :param df: df with id and survived columns:
    :param custom_name: a name to append after team name, e.g. random_forest
        (should not contain whitespace)
    :return:
    '''
    assert isinstance(df, pd.DataFrame)
    df_text = df[['id', 'survived']].to_csv(sep=';',
                                            header=['key', 'value'],
                                            index=False)
    url = ('https://openwhisk.eu-de.bluemix.net/api/v1/web/SPLab_Scripting/'
           'default/titanic.html')
    payload = {
        'submission': f'team_8_{custom_name}_{get_datetime()}',
        'csv': df_text
    }
    try:
        session = HTMLSession()
        print('submitting solution to server...')
        # submit
        res = session.post(url, data=payload)
        print('submission successful')
        # get current score and position
        text = res.text
        regex = (r'<br>(\d+):\s' +
                 re.escape(payload['submission']) +
                 r'\s→\s(\d+\.\d+)<br>')
        groups = re.search(regex, text)
        # get current high score
        highscore_regex = r'<br>1:\s(.+?)→\s(\d+\.\d+)<br>'
        highscore_groups = re.search(highscore_regex, text)
        print(f'current score: {round(float(groups[2]), 3)}'
              f' (position: {groups[1]})')
        print(f'current highscore: {round(float(highscore_groups[2]), 3)}'
              f' by "{highscore_groups[1]}"')

    except Exception as e:
        print(f'error: {e}')
        quit(1)


def prepare_data(df):
    df = prepare_sex(df)
    df = prepare_age(df)
    df = prepare_fare(df)
    df = prepare_embarked(df)
    df = prepare_boat(df)
    return df


def prepare_sex(df):
    # SEX: Prepare data 'sex' and store it in train_data
    # First: get the most frequent gender
    num_female = len(df[df.sex == 'female'])
    num_male = len(df[df.sex == 'male'])
    # Set the most frequent gender (female = 0, male = 1)
    most_freq_gender = 0 if num_female >= num_male else 1
    # Second: store gender data in train_data

    def sex_transformation(row):
        if row['sex'] == '' or pd.isna(row['sex']):
            # Most freq. gender is used if 'sex' is undefined
            return most_freq_gender
        if row['sex'] == 'female':
            return 0
        if row["sex"] == 'male':
            return 1
    df["sex"] = df.apply(lambda row: sex_transformation(row), axis=1)
    return df

def prepare_age(df):
    # AGE: Prepare data 'age' and store it in train_data
    # First: get the median age
    # Convert 'age' to float, empty values to 0
    df["age"] = df.age.apply(lambda age:
                                             np.nan if (age == '' or pd.isna(age))
                                             else float(age))
    median_age = df["age"].median(skipna = True )
    #store median age if age is undefined
    df["age"] = df.age.apply(lambda age:
                                             median_age if (age == '' or pd.isna(age))
                                             else age)
    return df


def prepare_fare(df):
    # FARE: Prepare data 'fare' and store it in train_data
    # First: get the 'fare' and 'pclass' data
    # Convert 'fare' to float, empty values to 0
    df["fare"] = df.fare.apply(lambda fare: 0 if fare == '' else float(fare))
    median_fare = df.groupby(["pclass"])["fare"].median(skipna=True)
    df["fare"] = df.apply(lambda row: 0 if row["fare"] == 0 else median_fare[row["pclass"]],axis = 1)
    return df


def prepare_embarked(df):
    # EMBARKED: Prepare data 'embarked' and store it in train_data
    # First: get the most common 'embarked' value
    embarked_data = df.groupby("embarked").size().reset_index(name='N')
    mc_embarked = embarked_data.embarked[embarked_data.N == max(embarked_data.N)]

    # Second: replace empty entries with the most common 'embarked' value

    df["embarked"] = df.embarked.apply(lambda embarked: mc_embarked if embarked == '' else embarked)

    # Third: convert all 'embarked' values to int

    df["embarked"] = pd.Categorical(df.embarked)
    df["embarked"] = df.embarked.cat.codes
    return df

def prepare_boat(df):
    # boat: Prepare data 'boat' and store it in df
    df['boat'] = df.boat.astype(str)
    return df
