import pandas as pd 
import numpy as np 
#import os
from DecisionTree import DecisionTree
from pathlib import Path

def error_rate(predy,y):
    if np.reshape(predy,(-1)).shape != np.reshape(y,(-1)).shape:
        raise ValueError("The sample size are not equal.")
    return np.mean(np.reshape(predy,(-1,1))!=np.reshape(y,(-1,1)))

print("CS5350/6350, HW1, Cen-Jhih Li.")

print("\n\n-------------Problem 2: car data")

data_path = Path('./data/car')
colnames = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
training_path = data_path/'train.csv'
test_path = data_path/'test.csv'

train_data = pd.read_csv(training_path, header=None, names=colnames) 
test_data = pd.read_csv(test_path, header=None, names=colnames) 
print("\nThe original data:")
print(train_data.head()) #viewing some row of the dataset

print("now training the tree, it may take a few seconds or minutes.")
column = train_data.columns.to_numpy()[:-1]
x = train_data[column].to_numpy()
y = train_data.label.to_numpy()
error_table = np.zeros((6,6))
for i, criterion in enumerate(["entropy", "gini", "me"]):
    for max_depth in range(1,7): #should be not larger than column.shape[0], DecisionTree will take minimum of {column.shape[0], max_depth}.
        mytree = DecisionTree(trainx = x, trainy = y, column = column, criterion = criterion, max_depth = max_depth, entropy_base = 6)
        mytree.fit()
        predy = mytree.predict(train_data.to_numpy())
        error_table[max_depth-1, 2*i] = error_rate(predy,train_data.label.to_numpy())
        #print("training error for max_depth = {} is {}".format(max_depth, error_table[max_depth-1, 2*i]))
        predy = mytree.predict(test_data.to_numpy())
        error_table[max_depth-1, 2*i+1] = error_rate(predy,test_data.label.to_numpy())
        #print("testing error for max_depth = {} is {}".format(max_depth, error_table[max_depth-1, 2*i+1]))

report = pd.DataFrame(error_table, columns = ["entropy_train", "entropy_test", "gini_train", "gini_test", "me_train", "me_test"])
report.insert(loc=0, column="depth", value=np.arange(1,7))
print("error rate:")
print(report.to_string(index=False))

print("averages  : {}".format(error_table.mean(axis = 0)))

print("\n\n-------------Problem 3: bank data (consider unknown as a catagory)")
data_path = Path('./data/bank')
colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 
          'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
training_path = data_path/'train.csv'
test_path = data_path/'test.csv'

train_data = pd.read_csv(training_path, header=None, names=colnames) 
test_data = pd.read_csv(test_path, header=None, names=colnames) 
print("\nThe original data:")
print(train_data.head()) #viewing some row of the dataset

thresholds = train_data[["age", "balance", "day", "duration", "campaign", "pdays", "previous"]].median()
#thain and test should use the same thresholds.
#consider unknown as a catagory
def bank_preprocessing(df):
    #for col in ["default", "housing", "loan", "y"]:
        #df.loc[df[col] == "yes", col] = 1
        #df.loc[df[col] == "no", col] = 0

    month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    df.month = df.month.map(month_map)
    #numeric: age balance day duration campaign pdays(-1 means client was not previously contacted) previous
    for col in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
        df.loc[df[col] <= thresholds[col], col] = 0
        df.loc[df[col] > thresholds[col], col] = 1
        df[col] = df[col].map({0: "low", 1: "high"})

    return df

train_data = bank_preprocessing(train_data)
test_data = bank_preprocessing(test_data)
print("\ntransform numerical data to categorical and transform month to integer:")
print(train_data.head()) #viewing some row of the dataset

print("now training the tree, it may take a few seconds or minutes.")
column = train_data.columns.to_numpy()[:-1]
x = train_data[column].to_numpy()
y = train_data.y.to_numpy()
error_table = np.zeros((16,6))
for i, criterion in enumerate(["entropy", "gini", "me"]):
    for max_depth in range(1,17): #should be not larger than column.shape[0], DecisionTree will take minimum of {column.shape[0], max_depth}.
        mytree = DecisionTree(trainx = x, trainy = y, column = column, criterion = criterion, max_depth = max_depth, entropy_base = 16)
        mytree.fit()
        predy = mytree.predict(train_data.to_numpy())
        error_table[max_depth-1, 2*i] = error_rate(predy,train_data.y.to_numpy())
        #print("training error for max_depth = {} is {}".format(max_depth, error_table[max_depth-1, 2*i]))
        predy = mytree.predict(test_data.to_numpy())
        error_table[max_depth-1, 2*i+1] = error_rate(predy,test_data.y.to_numpy())
        #print("testing error for max_depth = {} is {}".format(max_depth, error_table[max_depth-1, 2*i+1]))

report = pd.DataFrame(error_table, columns = ["entropy_train", "entropy_test", "gini_train", "gini_test", "me_train", "me_test"])
report.insert(loc=0, column="depth", value=np.arange(1,17))
print("error rate:")
print(report.to_string(index=False))

print("averages  : {}".format(error_table.mean(axis = 0)))

train_data[["job", "education", "contact", "poutcome"]].mode()


#replace unknown by most frequent value (mode)
print("\n\n-------------Problem 3: bank data (replace unknown by most frequent value (mode))")
def fill_unknown(df):
    fill_value = ["blue-collar", "secondary", "cellular", "failure"]
    for i, col in enumerate(["job", "education", "contact", "poutcome"]):
        df.loc[df[col] == "unknown", col] = fill_value[i]
    return df

train_data = fill_unknown(train_data)
test_data = fill_unknown(test_data)
print("\nreplace unknown by most frequent value (job: blue-collar; education: secondary; contact: cellular; poutcome: failure):")
print(train_data.head()) #viewing some row of the dataset

print("now training the tree, it may take a few seconds or minutes.")
column = train_data.columns.to_numpy()[:-1]
x = train_data[column].to_numpy()
y = train_data.y.to_numpy()
error_table = np.zeros((16,6))
for i, criterion in enumerate(["entropy", "gini", "me"]):
    for max_depth in range(1,17): #should be not larger than column.shape[0], DecisionTree will take minimum of {column.shape[0], max_depth}.
        mytree = DecisionTree(trainx = x, trainy = y, column = column, criterion = criterion, max_depth = max_depth, entropy_base = 16)
        mytree.fit()
        predy = mytree.predict(train_data.to_numpy())
        error_table[max_depth-1, 2*i] = error_rate(predy,train_data.y.to_numpy())
        #print("training error for max_depth = {} is {}".format(max_depth, error_table[max_depth-1, 2*i]))
        predy = mytree.predict(test_data.to_numpy())
        error_table[max_depth-1, 2*i+1] = error_rate(predy,test_data.y.to_numpy())
        #print("testing error for max_depth = {} is {}".format(max_depth, error_table[max_depth-1, 2*i+1]))
        
report = pd.DataFrame(error_table, columns = ["entropy_train", "entropy_test", "gini_train", "gini_test", "me_train", "me_test"])
report.insert(loc=0, column="depth", value=np.arange(1,17))
print("error rate:")
print(report.to_string(index=False))

print("averages  : {}".format(error_table.mean(axis = 0)))
print("\n")

mytree = DecisionTree(trainx = x, trainy = y, column = column, criterion = 'entropy', max_depth = 16, entropy_base = 16)
print("information gain from using poutcome as root is {}".format(mytree._IG(train_data.poutcome.to_numpy(), train_data.y.to_numpy())))
print("information gain from using month as root is {}".format(mytree._IG(train_data.month.to_numpy(), train_data.y.to_numpy())))
print("-------------THE END-------------")