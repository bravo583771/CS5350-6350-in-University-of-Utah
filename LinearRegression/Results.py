import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#import os
from linear_regression import linear_regression
from pathlib import Path
np.random.seed(2022)

def error_rate(predy,y):
    if np.reshape(predy,(-1)).shape != np.reshape(y,(-1)).shape:
        raise ValueError("The sample size are not equal.")
    return np.mean(np.reshape(predy,(-1,1))!=np.reshape(y,(-1,1)))
print("CS5350/6350, HW2, Cen-Jhih Li.")

print("\n\n-------------Problem 4: concrete data")

print("\n\n-------------4a: batch gradient descent")

data_path = Path('./data/concrete')
colnames = ['Cement', 'Slag', 'Fly_ash', 'Water', 'SP', 'Coarse_Aggr', 'Fine_Aggr', 'y']
training_path = data_path/'train.csv'
test_path = data_path/'test.csv'

train_data = pd.read_csv(training_path, header=None, names=colnames) 
test_data = pd.read_csv(test_path, header=None, names=colnames) 
print(train_data.head()) #viewing some row of the dataset

column = train_data.columns.to_numpy(copy = True)[:-1]
x = train_data[column].to_numpy(copy = True)
y = train_data.y.to_numpy(copy = True)
testx = test_data[column].to_numpy(copy = True)
testy = test_data.y.to_numpy(copy = True)

#training_error = np.zeros(6,3)
#testing_error = np.zeros(6,3)
regressor1 = linear_regression(trainx = x, trainy = y, testx = testx, testy = testy)
SGD_loss, SGD_test_loss = regressor1.SGD(learning_rate = 0.0025, converge_criteria = 10**-6) #0.0025
regressor2 = linear_regression(trainx = x, trainy = y, testx = testx, testy = testy)
bGD_loss, bGD_test_loss = regressor2.bGD(max_iter = 100000, learning_rate = 0.05, converge_criteria = 10**-6)

plt.plot(bGD_loss)
#plt.plot(bGD_test_loss)
plt.title('BGD')
plt.ylabel('loss')
plt.xlabel('# of iterations')
plt.legend(['training'], loc='upper right')
#plt.savefig('4a.png') 
plt.show()
print("\nThe BGD solution is {}.".format(regressor2.w))
print("\nThe test loss is {}.".format(bGD_test_loss[-1]))

print("\n\n-------------4b: stochastic gradient descent")

plt.plot(SGD_loss)
#plt.plot(SGD_test_loss)
plt.title('SGD')
plt.ylabel('loss')
plt.xlabel('# of iterations')
plt.legend(['training'], loc='upper right')
#plt.savefig('4b.png') 
plt.show()
print("\nThe SGD solution is {}.".format(regressor1.w))
print("\nThe test loss is {}.".format(SGD_test_loss[-1]))

print("\n\n-------------4c: compare with optimal solution")
print("\nThe optimal solution is {}.".format(regressor1.optimal))

print("-------------THE END-------------")