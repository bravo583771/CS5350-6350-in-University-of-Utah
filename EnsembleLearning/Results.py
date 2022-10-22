import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#import os
from ensemble import AdaBoost, DecisionStump, DecisionTree, Bagging, RandomForest
from pathlib import Path
np.random.seed(2022)

def error_rate(predy,y):
    if np.reshape(predy,(-1)).shape != np.reshape(y,(-1)).shape:
        raise ValueError("The sample size are not equal.")
    return np.mean(np.reshape(predy,(-1,1))!=np.reshape(y,(-1,1)))

print("CS5350/6350, HW2, Cen-Jhih Li.")

print("\n\n-------------Problem 2: bank data")

print("\n\n-------------2a: Adaboost")

data_path = Path('./data/bank')
colnames = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 
          'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
training_path = data_path/'train.csv'
test_path = data_path/'test.csv'

train_data = pd.read_csv(training_path, header=None, names=colnames) 
test_data = pd.read_csv(test_path, header=None, names=colnames) 

thresholds = train_data[["age", "balance", "day", "duration", "campaign", "pdays", "previous"]].median()
def bank_preprocessing(df):
    #for col in ["default", "housing", "loan", "y"]:
        #df.loc[df[col] == "yes", col] = 1
        #df.loc[df[col] == "no", col] = 0

    #month_map = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
    #df.month = df.month.map(month_map)
    #numeric: age balance day duration campaign pdays(-1 means client was not previously contacted) previous
    for col in ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]:
        df.loc[df[col] <= thresholds[col], col] = 0
        df.loc[df[col] > thresholds[col], col] = 1
        df[col] = df[col].map({0: "low", 1: "high"})

    return df

train_data = bank_preprocessing(train_data)
test_data = bank_preprocessing(test_data)
print(train_data.head()) #viewing some row of the dataset

column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
#mytree = DecisionStump(trainx = x, trainy = y, column = column, criterion = "entropy", entropy_base = 16)
Boost_tree = AdaBoost(trainx = x, trainy = y, column = column, entropy_base = 16)
Boost_tree.fit()
train_error = np.zeros(500)
test_error = np.zeros(500)
train_predy, train_stump_predy = Boost_tree.predict(train_data.to_numpy(copy=True))
test_predy, test_stump_predy = Boost_tree.predict(test_data.to_numpy(copy=True))

for i in range(len(train_predy)):
    train_error[i] = error_rate( train_predy[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = error_rate( test_predy[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of iterations')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2a1.png')
print("image save 2a1.png") 
plt.show()

for i in range(len(train_predy)):
    train_error[i] = error_rate( train_stump_predy[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = error_rate( test_stump_predy[i], test_data.y.to_numpy(copy=True) )
plt.plot(train_error)
plt.plot(test_error)
plt.title('stump error')
plt.ylabel('error')
plt.xlabel('stump')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2a2.png') 
print("image save 2a2.png") 
plt.show()


print("\n\n-------------2b: bagging")
column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
Bagging_tree = Bagging(trainx = x, trainy = y, column = column, max_depth = 16)
Bagging_tree.fit()
train_error = np.zeros(500)
test_error = np.zeros(500)
train_predy = Bagging_tree.predict(train_data.to_numpy(copy=True))
test_predy = Bagging_tree.predict(test_data.to_numpy(copy=True))

for i in range(len(train_predy)):
    train_error[i] = error_rate( train_predy[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = error_rate( test_predy[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2b.png') 
print("image save 2b.png") 
plt.show()


print("\n\n-------------2c: bias and variance")

def bias(y_pred, y_true):
    """
    y_pred: (max_ite, p)
    y_true: (p,)
    """
    return np.square(np.subtract(np.mean(y_pred, axis = 0),y_true)).mean()
bias.__name__='bias^2'

def variance(y_pred, y_true):
    """
    y_pred: (max_ite, p)
    y_true: (p,)
    """
    return np.mean([np.square( np.subtract( y_hat, np.mean(y_pred, axis = 0) ) )  for y_hat in y_pred], axis = 0).mean()
variance.__name__="variance"

column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
n_sample = x.shape[0]
y_pred_s = []
y_pred_b = []
for i in range(100):
    print("repeat: ", i+1)
    index = np.random.choice(np.arange(n_sample), size=1000, replace=False)
    Bagging_tree = Bagging(trainx = x[index,:], trainy = y[index], column = column, max_depth = 16)
    Bagging_tree.fit()
    single_tree = Bagging_tree.tree[0]
    tempy = single_tree.predict(test_data.to_numpy(copy=True))
    y_pred_s.append(tempy)

    tempy = Bagging_tree.predict(test_data.to_numpy(copy=True))
    y_pred_b.append(tempy[499])

def helper(x):
    temp = np.ones(x.shape)
    temp[x=='no']=0
    return temp

y_pred_s_a = helper(np.array(y_pred_s))
y_pred_b_a = helper(np.array(y_pred_b))
y_true = helper(test_data.y.to_numpy(copy=True))

bias_s = bias(y_pred_s_a, y_true)
bias_b = bias(y_pred_b_a, y_true)
variance_s = variance(y_pred_s_a, y_true)
variance_b = variance(y_pred_b_a, y_true)
print("bias^2 (single, bagging): ", bias_s, bias_b)
print("variance (single, bagging): ", variance_s, variance_b)
print("total (single, bagging): ", bias_s + variance_s, bias_b + variance_b)

print("\n\n-------------2d: random forest")
print("number of features: 2")

np.random.seed(2022)
column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
tree_RF = RandomForest(trainx = x, trainy = y, column = column, max_depth = 16, select_features = 2)
tree_RF.fit()

train_error = np.zeros(500)
test_error = np.zeros(500)
train_predy = tree_RF.predict(train_data.to_numpy(copy=True))
test_predy = tree_RF.predict(test_data.to_numpy(copy=True))

for i in range(len(train_predy)):
    train_error[i] = error_rate( train_predy[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = error_rate( test_predy[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('RF error (features = 2)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2d1.png') 
print("image save 2d1.png") 
plt.show()

print("number of features: 4")
column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
tree_RF = RandomForest(trainx = x, trainy = y, column = column, max_depth = 16, select_features = 4)
tree_RF.fit()

train_error = np.zeros(500)
test_error = np.zeros(500)
train_predy = tree_RF.predict(train_data.to_numpy(copy=True))
test_predy = tree_RF.predict(test_data.to_numpy(copy=True))

for i in range(len(train_predy)):
    train_error[i] = error_rate( train_predy[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = error_rate( test_predy[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('RF error (features = 4)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2d2.png') 
print("image save 2d2.png") 
plt.show()

print("number of features: 6")
column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
tree_RF = RandomForest(trainx = x, trainy = y, column = column, max_depth = 16, select_features = 6)
tree_RF.fit()

train_error = np.zeros(500)
test_error = np.zeros(500)
train_predy = tree_RF.predict(train_data.to_numpy(copy=True))
test_predy = tree_RF.predict(test_data.to_numpy(copy=True))

for i in range(len(train_predy)):
    train_error[i] = error_rate( train_predy[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = error_rate( test_predy[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('RF error (features = 6)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('2d3.png') 
print("image save 2d3.png") 
plt.show()

print("\n\n-------------2c: bias and variance")

def bias(y_pred, y_true):
    """
    y_pred: (max_ite, p)
    y_true: (p,)
    """
    return np.square(np.subtract(np.mean(y_pred, axis = 0),y_true)).mean()
bias.__name__='bias^2'

def variance(y_pred, y_true):
    """
    y_pred: (max_ite, p)
    y_true: (p,)
    """
    return np.mean([np.square( np.subtract( y_hat, np.mean(y_pred, axis = 0) ) )  for y_hat in y_pred], axis = 0).mean()
variance.__name__="variance"

column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
n_sample = x.shape[0]
y_pred_s = []
y_pred_b = []
for i in range(100):
    print("repeat: ", i+1)
    index = np.random.choice(np.arange(n_sample), size=1000, replace=False)
    Bagging_tree = RandomForest(trainx = x[index,:], trainy = y[index], column = column, max_depth = 16, select_features = 6)
    Bagging_tree.fit()
    single_tree = Bagging_tree.tree[0]
    tempy = single_tree.predict(test_data.to_numpy(copy=True))
    y_pred_s.append(tempy)

    tempy = Bagging_tree.predict(test_data.to_numpy(copy=True))
    y_pred_b.append(tempy[499])

def helper(x):
    temp = np.ones(x.shape)
    temp[x=='no']=0
    return temp

y_pred_s_a = helper(np.array(y_pred_s))
y_pred_b_a = helper(np.array(y_pred_b))
y_true = helper(test_data.y.to_numpy(copy=True))

bias_s = bias(y_pred_s_a, y_true)
bias_b = bias(y_pred_b_a, y_true)
variance_s = variance(y_pred_s_a, y_true)
variance_b = variance(y_pred_b_a, y_true)
print("bias^2 (single, bagging): ", bias_s, bias_b)
print("variance (single, bagging): ", variance_s, variance_b)
print("total (single, bagging): ", bias_s + variance_s, bias_b + variance_b)

print("\n\n-------------3: credit card data")
data_path = Path('./data/credit_card/data.csv')
colnames = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 
          'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 
          'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'y'] #y: default payment next month
          																							

data = pd.read_csv(data_path, header=None, names=colnames) 
print("Original data")
print(data.head())

cts_f = ["LIMIT_BAL", "AGE", 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 
          'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
thresholds = data[cts_f].median()
def preprocessing(df):
    #numeric: age balance day duration campaign pdays(-1 means client was not previously contacted) previous
    for col in cts_f:
        df.loc[df[col] <= thresholds[col], col] = 0
        df.loc[df[col] > thresholds[col], col] = 1
        df[col] = df[col].map({0: "low", 1: "high"})

    return df
data = preprocessing(data)
shuffle = np.random.choice(np.arange(30000), size = 30000, replace=False)
data = data.iloc[shuffle]

print("Transform continuous variable to categorical variable")
print(data.head())
train_data = data.iloc[:24000,:]
test_data = data.iloc[24000:,:]
print(train_data.head())

print("\nAdaboost")
column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
Boost_tree = AdaBoost(trainx = x, trainy = y, column = column, entropy_base = 16)
Boost_tree.fit()
train_error = np.zeros(500)
test_error = np.zeros(500)
train_predy, _ = Boost_tree.predict(train_data.to_numpy(copy=True))
test_predy, _ = Boost_tree.predict(test_data.to_numpy(copy=True))

for i in range(len(train_predy)):
    train_error[i] = error_rate( train_predy[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = error_rate( test_predy[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of iterations')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('3ADA.png') 
print("image save 3ADA.png") 
plt.show()

print("\nbagging")
column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
Bagging_tree = Bagging(trainx = x, trainy = y, column = column, max_depth = 16)
Bagging_tree.fit()
train_error = np.zeros(500)
test_error = np.zeros(500)
train_predy = Bagging_tree.predict(train_data.to_numpy(copy=True))
test_predy = Bagging_tree.predict(test_data.to_numpy(copy=True))

for i in range(len(train_predy)):
    train_error[i] = error_rate( train_predy[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = error_rate( test_predy[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('model error')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('3Bagging.png') 
print("image save 3Bagging.png") 
plt.show()

print("number of features: 6")
column = train_data.columns.to_numpy(copy=True)[:-1]
x = train_data[column].to_numpy(copy=True)
y = train_data.y.to_numpy(copy=True)
tree_RF = RandomForest(trainx = x, trainy = y, column = column, max_depth = 23, select_features = 6)
tree_RF.fit()

train_error = np.zeros(500)
test_error = np.zeros(500)
train_predy = tree_RF.predict(train_data.to_numpy(copy=True))
test_predy = tree_RF.predict(test_data.to_numpy(copy=True))

for i in range(len(train_predy)):
    train_error[i] = error_rate( train_predy[i], train_data.y.to_numpy(copy=True) )
    test_error[i] = error_rate( test_predy[i], test_data.y.to_numpy(copy=True) )

plt.plot(train_error)
plt.plot(test_error)
plt.title('RF error (features = 6)')
plt.ylabel('error')
plt.xlabel('# of trees')
plt.legend(['training', 'testing'], loc='upper right')
plt.savefig('3RF.png') 
print("image save 3RF.png") 
plt.show()


print("-------------THE END-------------")