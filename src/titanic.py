# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:49:47 2018

@author: patrik, team8, 8team
"""


import pandas as pd
import random
import titanic_helpers as th
import datetime

df = pd.read_csv("data/titanic3_train.csv", delimiter = ";")


df_women = df[df.sex == "female"]
df_men = df[df.sex == "male"]
df_men.survived.mean()
df_women.survived.mean()

df_test = pd.read_csv("data/titanic3_train.csv", delimiter = ";")


#def f(row):
#    if row["sex"] == "female":
#       return 1
 #   else:
#      return 0


#df_test["survived"] = df_test.apply(lambda row: f(row), axis = 1)


df_test[df_test["sex"] == "female"]["survived"] = 1
df_test[df_test["sex"]== "male"]["survived"] = 0


df_test["survived"] = 0


th.write_csv(df=df_test)


filename = r"titanic_"+str(datetime.datetime.now().isoformat()).replace(".", "_").replace(":","_") + ".csv"

df_test[["id", "survived"]].to_csv(filename, sep=";", header=["key", "value"], index = False)
