import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn import metrics


## computing functions
def discriminant(xk: np.ndarray, W: np.ndarray) -> np.ndarray:
    return W@xk

def sigmoid(zk: np.ndarray) -> np.ndarray:
    # I = np.ones()
    return 1 / (1 + np.exp(-zk))

def MSE(data, targets, W):
    assert(len(data) == len(targets))

    sum = 0
    for k in range(len(data)):
        xk = np.array(data[k])
        zk = discriminant(xk, W)
        gk = sigmoid(zk)
        tk = np.array(targets[k])

        sum += 1/2 * (gk - tk).T @ (gk - tk)

    return sum

def gradWMSE(data, targets, W):
    assert(len(data) == len(targets))

    sum = 0
    for k in range(len(data)):
        xk = np.array(data[k])
        zk = discriminant(xk, W)
        gk = sigmoid(zk)
        tk = np.array(targets[k])
        sum += np.outer((gk - tk) * gk * (np.ones(len(gk), dtype=int) - gk),xk)

    return sum

def trainingStep(data, targets, W, alpha):
    return W - alpha * gradWMSE(data, targets, W)


## targets
targets = {}
targets['Iris-setosa'] = np.array([1, 0, 0])
targets['Iris-versicolor'] = np.array([0, 1, 0])
targets['Iris-virginica'] = np.array([0, 0, 1])

## retriving data
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']
datafile = "Data/iris.data"
dataframe = pd.read_csv(datafile, names=columns)

# choose which features to be used
data = dataframe.to_numpy()

# choose what data is to be used by what
training = 30
test = 20

trainingData = np.concatenate((data[:training], data[50:(training + 50)], data[100:(training + 100)]))
testData = np.concatenate((data[training:50], data[(training + 50):100], data[(training + 100):]))

# make targets for each datapoint
T_N = np.array([targets[data[-1]] for data in trainingData])
T_T = np.array([targets[data[-1]] for data in testData])

# remove species information
X_N = trainingData[:,:-1].astype(np.float64)
X_T = testData[:,:-1].astype(np.float64)

## Tror ikke vi trenger dette
# add 1 to all datapoints to match x = [x^t 1]^T
# X_N = np.hstack((X_N, np.ones((1, len(X_N))).reshape(-1, 1)))

C = 3
D = 4

W = np.zeros((C, D))


def training(X_N, T_N, alpha, iterations):
    C = len(T_N[0])
    D = len(X_N[0])
    W0 = np.zeros((C, D))
    W = [W0]
    MSEList = [MSE(X_N, T_N, W0)]
    for i in range(iterations):
        # W[i+1] = W[i] - alpha * gradWMSE(X_N, T_N, W[i])
        # MSEList[i+1] = [MSE(X_N, T_N, W[i])]
        W.append(W[i] - alpha * gradWMSE(X_N, T_N, W[i]))
        MSEList.append(MSE(X_N, T_N, W[i]))
        # print(W[i])
    
    return MSEList, W[-1]
    
# iterations = 2000
# alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# tests = [training(X_N, T_N, alpha, iterations)[0] for alpha in alphas]

## Plotting

# for test in tests:
#     plt.plot(range(iterations + 1), test)

# plt.legend(alphas)
# plt.show()


def test(X_T, T_T):
    _, W = training(X_N, T_N, 0.01, 2000)
    print(W)
    print(discriminant(X_T[0], W))
    print(sigmoid(discriminant(X_T[0], W)))

test(X_T, T_T)
# def plotConfusionMatrix()





