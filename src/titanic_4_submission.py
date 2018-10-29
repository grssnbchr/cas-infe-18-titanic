
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
print("prepare test data...")

testFile = "../data/titanic3_test.csv"
orig_test_data = pd.read_csv(testFile, delimiter=";")
test_data = orig_test_data.copy()
test_data['dataset'] = 'test'
test_data['survived'] = None

df_data = test_data.append(train_data, ignore_index=False, sort=False)
df_data = th.prepare_data(df_data)

# PART 3: TRAINING AND PREDICTION

# The data is now ready to go. So lets fit to the train, then predict to
# the test!
print('Training...')

predictors = ["pclass", "sibsp", "parch", "sex", "age", "fare", "embarked"]

# fit a random forest
forest = RandomForestClassifier(n_estimators=100)
forest = (forest
          .fit(df_data[df_data['dataset'] == 'train'][predictors],
               df_data[df_data['dataset'] == 'train']["survived"].astype(int))
          )

# fit a logistic regression
logreg = LogisticRegression()
logreg.fit(df_data[df_data['dataset'] == 'train'][predictors],
           df_data[df_data['dataset'] == 'train']["survived"].astype(int))

print('Predicting...')
rf_out = forest.predict(df_data[df_data['dataset'] == 'test'][predictors])
lr_out = logreg.predict(df_data[df_data['dataset'] == 'test'][predictors])

# Create DataFrame for outputfile
df_rf = pd.DataFrame(columns=["id", "survived"])
df_rf["id"] = df_data[df_data['dataset'] == 'test']['id']
df_rf["survived"] = rf_out

# Write the data into a file
th.write_csv(df=df_rf)

# automagically submit to server
th.submit_answer(df_rf, custom_name='random_forest_2')


# Create DataFrame for outputfile
df_lr = pd.DataFrame(columns=["id", "survived"])
df_lr["id"] = df_data[df_data['dataset'] == 'test']['id']
df_lr["survived"] = lr_out

# Write the data into a file
th.write_csv(df=df_lr)

# automagically submit to server
th.submit_answer(df_lr, custom_name='logistic_regression_2')

print('Done.')
