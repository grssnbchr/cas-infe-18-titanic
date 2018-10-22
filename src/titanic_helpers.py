

import pandas as pd
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

