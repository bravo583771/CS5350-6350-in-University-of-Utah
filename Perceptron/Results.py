import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#import os
from perceptron import perceptron
from pathlib import Path
np.random.seed(2022)

def error_rate(predy,y):
    if np.reshape(predy,(-1)).shape != np.reshape(y,(-1)).shape:
        raise ValueError("The sample size are not equal.")
    return np.mean(np.reshape(predy,(-1,1))!=np.reshape(y,(-1,1)))

print("CS5350/6350, HW3, Cen-Jhih Li.")

print("\n\n-------------Problem 2: bank note")

print("\n\n-------------2a: standard Perceptron")

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


myperceptron = perceptron(trainx = x, trainy = y, n_epoch = 10)
w = myperceptron.standard(learning_rate = 0.05)
print("the weight is:")
print(w)

predy = myperceptron.predict(testx, w)
print("average test error is")
print(error_rate(predy,testy))


print("\n\n-------------2b: voted Perceptron")

ws, cs = myperceptron.voted(learning_rate = 0.05)
print("w1 = {} \nand c1 = {}".format(ws[0], cs[0]))
print("w5 = {} \nand c5 = {}".format(ws[4], cs[4]))
print("w10 = {} \nand c10 = {}".format(ws[9], cs[9]))
print("w20 = {} \nand c20 = {}".format(ws[19], cs[19]))
print("w50 = {} \nand c50 = {}".format(ws[49], cs[49]))
print("w100 = {} \nand c100 = {}".format(ws[99], cs[99]))
print("w200 = {} \nand c200 = {}".format(ws[199], cs[199]))
print("w500 = {} \nand c500 = {}".format(ws[499], cs[499]))
print("w1000 = {} \nand c1000 = {}".format(ws[999], cs[999]))
print("w1400 = {} \nand c1400 = {}".format(ws[1399], cs[1399]))
print("w1500 = {} \nand c1500 = {}".format(ws[1499], cs[1499]))
print("w1524 = {} \nand c1524 = {}".format(ws[1523], cs[1523]))
print("the whole count list is:")
print(cs)

predy = myperceptron.predict(testx, w = None, ws = ws, cs = cs)
print("average test error is")
print(error_rate(predy,testy))

print("\n\n-------------2c: average Perceptron")

w = myperceptron.average(learning_rate = 0.05)
print("the weight is:")
print(w)

predy = myperceptron.predict(testx, w)
print("average test error is")
print(error_rate(predy,testy))

print("-------------THE END-------------")