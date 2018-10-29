
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import titanic_helpers as th

# PART 1: PREPARING THE TRAINING DATA
print("prepare train data...")


# Open up the CSV file into a Python object
trainFile = "../data/titanic3_train.csv"
orig_train_data = pd.read_csv(trainFile, delimiter=";")
train_data = orig_train_data.copy()
train_data['dataset'] = 'train'

# PART 2: PREPARING THE TEST DATA

# Now we have to do the same for the test data as we did for the training data
print("prepare test data...")

testFile = "../data/titanic3_test.csv"
orig_test_data = pd.read_csv(testFile, delimiter=";")
test_data = orig_test_data.copy()
test_data['dataset'] = 'test'
test_data['survived'] = None

df_data = test_data.append(train_data, ignore_index=False, sort=False)
df_data = th.prepare_data(df_data)

# After the training and test data is created, collect the test data's ids

# PART 3: TRAINING AND PREDICTION

# The data is now ready to go. So lets fit to the train, then predict to
# the test!
print('Training...')
forest = RandomForestClassifier(n_estimators=100)
# Build a forest of trees from the training set (X, y)
predictors = ["pclass", "sibsp", "parch", "sex", "age", "fare", "embarked"]
forest = (forest
          .fit(df_data[df_data['dataset'] == 'train'][predictors],
               df_data[df_data['dataset'] == 'train']["survived"].astype(int))
          )

print('Predicting...')
output = forest.predict(df_data[df_data['dataset'] == 'test'][predictors])

# Create DataFrame for outputfile
df = pd.DataFrame(columns=["id", "survived"])
df["id"] = df_data[df_data['dataset'] == 'test']['id']
df["survived"] = output

# Write the data into a file
th.write_csv(df=df)

# automagically submit to server
th.submit_answer(df, custom_name='random_forest_2')

print('Done.')
