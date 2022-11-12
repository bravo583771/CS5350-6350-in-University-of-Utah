import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#import os
from SVM import SVM
from pathlib import Path
from itertools import product
np.random.seed(2022)

def error_rate(predy,y):
    if np.reshape(predy,(-1)).shape != np.reshape(y,(-1)).shape:
        raise ValueError("The sample size are not equal.")
    return np.mean(np.reshape(predy,(-1,1))!=np.reshape(y,(-1,1)))

def scheduler1(initial_learning_rate: float, iter: int):
    a = 20
    return initial_learning_rate/(1+initial_learning_rate*iter/a)

def scheduler2(initial_learning_rate: float, iter: int):
    return initial_learning_rate/(1+iter)

print("CS5350/6350, HW4, Cen-Jhih Li.")

print("\n\n-------------Problem 2: bank note, primal form optimization")

print("\n\n-------------2a: schedule of learning rate is gamma_0 / ( 1 + (gamma_0/t) * a )")

data_path = Path('./data/bank_note')
colnames = ['variance', 'skewness', 'curtosis', 'entropy', 'y']
training_path = data_path/'train.csv'
test_path = data_path/'test.csv'

train_data = pd.read_csv(training_path, header=None, names=colnames) 
test_data = pd.read_csv(test_path, header=None, names=colnames) 
train_data['y'] = train_data['y'].map(lambda x: 2*(x-0.5))
test_data['y'] = test_data['y'].map(lambda x: 2*(x-0.5))
print(train_data.head()) #viewing some row of the dataset

column = train_data.columns.to_numpy()[:-1]
x = train_data[column].to_numpy()
y = train_data.y.to_numpy()
testx = test_data[column].to_numpy()
testy = test_data.y.to_numpy()


mysvm = SVM(trainx = x, trainy = y, n_epoch = 100)
error1 = np.zeros([3,2])
w1a = mysvm.subGD(scheduler = scheduler1, C = 100/873, initial_learning_rate = 0.01)
train_predy = mysvm.predict(x, w1a)
test_predy = mysvm.predict(testx, w1a)
error1[0,0], error1[0,1] = error_rate(train_predy,y), error_rate(test_predy,testy)

w2a = mysvm.subGD(scheduler = scheduler1, C = 500/873, initial_learning_rate = 0.01)
train_predy = mysvm.predict(x, w2a)
test_predy = mysvm.predict(testx, w2a)
error1[1,0], error1[1,1] = error_rate(train_predy,y), error_rate(test_predy,testy)

w3a = mysvm.subGD(scheduler = scheduler1, C = 700/873, initial_learning_rate = 0.01)
train_predy = mysvm.predict(x, w3a)
test_predy = mysvm.predict(testx, w3a)
error1[2,0], error1[2,1] = error_rate(train_predy,y), error_rate(test_predy,testy)

print("columns=[train, test], row 1 is C = 100/873, row 2 is C = 500/873, row 3 is C = 700/873")
print(error1)


print("\n\n-------------2b: schedule of learning rate is gamma_0 / ( 1 + t )")

error2 = np.zeros([3,2])
w1b = mysvm.subGD(scheduler = scheduler2, C = 100/873, initial_learning_rate = 0.01)
train_predy = mysvm.predict(x, w1b)
test_predy = mysvm.predict(testx, w1b)
error2[0,0], error2[0,1] = error_rate(train_predy,y), error_rate(test_predy,testy)

w2b = mysvm.subGD(scheduler = scheduler2, C = 500/873, initial_learning_rate = 0.01)
train_predy = mysvm.predict(x, w2b)
test_predy = mysvm.predict(testx, w2b)
error2[1,0], error2[1,1] = error_rate(train_predy,y), error_rate(test_predy,testy)

w3b = mysvm.subGD(scheduler = scheduler2, C = 700/873, initial_learning_rate = 0.01)
train_predy = mysvm.predict(x, w3b)
test_predy = mysvm.predict(testx, w3b)
error2[2,0], error2[2,1] = error_rate(train_predy,y), error_rate(test_predy,testy)

print("columns=[train, test], row 1 is C = 100/873, row 2 is C = 500/873, row 3 is C = 700/873")
print(error2)

print("\n\n-------------2c: comparison")

print("scheduler in 1a, C = 100/873, w = {}".format(w1a))
print("scheduler in 1b, C = 100/873, w = {}".format(w1b))
print("difference: {}".format(w1a-w1b))
print("scheduler in 1a, C = 500/873, w = {}".format(w2a))
print("scheduler in 1b, C = 500/873, w = {}".format(w2b))
print("difference: {}".format(w2a-w2b))
print("scheduler in 1a, C = 700/873, w = {}".format(w3a))
print("scheduler in 1b, C = 700/873, w = {}".format(w3b))
print("difference: {}".format(w3a-w3b))
print("difference of error, columns=[train, test], row 1 is C = 100/873, row 2 is C = 500/873, row 3 is C = 700/873")
print(error1 - error2)

print("\n\n-------------Problem 3: bank note, dual form optimization")

print("\n\n-------------3a: X dot X.transpose()")
opt_w1 = mysvm.optimal_dual(C = 100/873)
print("the optimal w (the first component is b) is {} when C = 100/873".format(opt_w1))
train_predy = mysvm.predict(x, opt_w1)
test_predy = mysvm.predict(testx, opt_w1)
print("training and testing errors are: ", error_rate(train_predy,y), error_rate(test_predy,testy)) 

opt_w2 = mysvm.optimal_dual(C = 500/873)
print("the optimal w (the first component is b) is {} when C = 500/873".format(opt_w2))
train_predy = mysvm.predict(x, opt_w2)
test_predy = mysvm.predict(testx, opt_w2)
print("training and testing errors are: ", error_rate(train_predy,y), error_rate(test_predy,testy)) 

opt_w3 = mysvm.optimal_dual(C = 700/873)
print("the optimal w (the first component is b) is {} when C = 700/873".format(opt_w3))
train_predy = mysvm.predict(x, opt_w3)
test_predy = mysvm.predict(testx, opt_w3)
print("training and testing errors are: ", error_rate(train_predy,y), error_rate(test_predy,testy)) 

print("\n\n-------------3b: kernel(xi, xj), 3c: the number of support vectors")

sig_list = [0.1,0.5,1,5,100]
C_list = [100/873,500/873,700/873]
error_kernel = np.zeros([15,2])
for i, (sig, C) in enumerate(product(sig_list,C_list)):
    print("{} gamma = {sig} and C = {}.".format(i+1, sig, C))
    alpha, b = mysvm.optimal_dual(C, sig, kernel = True)
    train_predy = mysvm.predict(x, alpha = alpha, b = b, sig = sig, kernal = True)
    test_predy = mysvm.predict(testx, alpha = alpha, b = b, sig = sig, kernal = True)
    error_kernel[i,0], error_kernel[i,1] = error_rate(train_predy,y), error_rate(test_predy,testy)

print("errors, columns=[train, test], \
    \nrow 1~5 are C = 100/873 and gamma = [0.1, 0.5, 1, 5, 100], \
    \nrow 6~10 are C = 500/873 and gamma = [0.1, 0.5, 1, 5, 100], \
    \nrow 11~15 are C = 700/873 and gamma = [0.1, 0.5, 1, 5, 100].")
print(error_kernel)

print("\n\n-------------3d: kernel perceptron")

sig_list = [0.1,0.5,1,5,100]
error_perceptron = np.zeros([5,2])
for i, sig in enumerate(sig_list):
    c = mysvm.kernel_perceptron(sig = sig)
    train_predy = mysvm.perceptron_predict(x, c, sig = sig)
    test_predy = mysvm.perceptron_predict(testx, c, sig = sig)
    error_perceptron[i,0], error_perceptron[i,1] = error_rate(train_predy,y), error_rate(test_predy,testy)
print("errors, columns=[train, test], \
    row 1~5 are gamma = [0.1, 0.5, 1, 5, 100].")
print(error_perceptron)

print("-------------THE END-------------")