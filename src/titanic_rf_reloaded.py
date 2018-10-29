
# coding: utf-8

# ## Random Forest "Reloaded"

# In[ ]:


"""
A more advanced version of the random forest predictor, with
- advanced feature engineering
- grid search
- cross validation

@author: timo, team8, 8team
"""

# utilities

import numpy as np
import pandas as pd
import titanic_helpers as th

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


# ## Part 1: Data Preparation

# In[2]:


train_df = pd.read_csv('../data/titanic3_train.csv',  delimiter=';')
test_df = pd.read_csv('../data/titanic3_test.csv',  delimiter=';')


# Have a look at the data:

# In[3]:


test_df.columns.values


# In[4]:


missing_val_count_by_column = (train_df.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > -1])


# In[5]:


print(train_df.columns.values)

print(train_df.sample(3))

train_df.info()


# ### Variable preparation

# Drop some columns.

# In[129]:


# save passenger ids for submission
test_ids = test_df['id']

def drop_columns(df):
    return df.drop(['id', 'ticket', 'embarked', 'home.dest', 'boat', 'body'], axis=1)

train_df = drop_columns(train_df)
test_df = drop_columns(test_df)


# In[130]:


train_df.info()
missing_val_count_by_column = (train_df.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > -1])


# #### Age
# Add age buckets and fill up unknown ages with most common age.

# In[131]:


def simplify_ages(df):
    df.age = df.age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(df.age, bins, labels=group_names)
    df.age = categories
    return df
train_df = simplify_ages(train_df)
train_df.info()

test_df = simplify_ages(test_df)


# In[132]:


train_df.head()


# In[133]:


# Replace 'Unknown' with most common age group
ages = train_df.groupby("age").size().reset_index(name='N')
print(ages)
most_common_age = ages.age[ages.N == max(ages.N)]
# extract single value, else the below lambda function does not work
most_common_age = most_common_age.values[0]
# most common age is Adult
train_df.age = train_df.age.apply(lambda age: most_common_age if age == 'Unknown' else age)

test_df.age = test_df.age.apply(lambda age: most_common_age if age == 'Unknown' else age)


# In[134]:


sns.barplot(x='sex', y='survived',data=train_df)


# In[135]:


sns.pointplot(x='pclass', y='survived', hue='sex', data=train_df,
              palette={'male': 'blue', 'female': 'pink'},
              markers=['*', 'o'], linestyles=['-', '--']);


# ### Feature Engineering

# In[136]:


# for the moment, drop everything except sex, age and pclass
train_df = train_df.loc[:, ['survived', 'sex', 'age', 'pclass']]
train_df.info()
missing_val_count_by_column = (train_df.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > -1])

test_df = test_df.loc[:, ['survived', 'sex', 'age', 'pclass']]


# In[137]:


# enable one-hot-encoding
train_df = pd.get_dummies(train_df)
train_df.info()
train_df.head()

test_df = pd.get_dummies(test_df)


# ### Training

# In[144]:


X_train = train_df.drop(['survived'], axis=1)
y_train = train_df['survived']

X_test = test_df.drop(['survived'], axis=1)
y_test = test_df['survived']

forest = RandomForestClassifier(n_estimators=100, random_state=12334344)

forest = forest.fit(X_train, y_train)

train_df.info()
test_df.info()
# predict output of test_df
output = forest.predict(X_test).astype(np.float)



# ### Submission

# In[145]:



# Create DataFrame for outputfile
df = pd.DataFrame(columns=['id', 'survived'])
df['id'] = test_ids
df['survived'] = output.astype(int)

# Write the data into a file
th.write_csv(df=df)

# automagically submit to server
th.submit_answer(df, custom_name='random_forest_reloaded')

