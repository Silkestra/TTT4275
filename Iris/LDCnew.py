import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn import metrics


## computing functions

def sigmoid(zk: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-zk))

def MSE(data: np.ndarray, targets, W):
    assert(len(data) == len(targets))

    Z = data @ W.T
    G = sigmoid(Z)
    return 1/2 * np.sum((G - targets)**2)


def gradWMSE(data, targets, W):
    assert(len(data) == len(targets))

    Z = data @ W.T
    G = sigmoid(Z)
    return ((G - targets) * G * (1 - G)).T @ data

def trainingStep(data, targets, W, alpha):
    return W - alpha * gradWMSE(data, targets, W)


##General training and test functions

def training(X_N, T_N, alpha, iterations):
    C = len(T_N[0])
    D = len(X_N[0])
    W0 = np.zeros((C, D))
    W = [W0]
    MSEList = [MSE(X_N, T_N, W0)]
    for i in range(iterations):
        W.append(W[i] - alpha * gradWMSE(X_N, T_N, W[i]))
        MSEList.append(MSE(X_N, T_N, W[i]))
    
    return MSEList, W[-1]

def test(data, targets, W):
    assert(len(data) == len(targets))

    predicted = []
    actual = []
    errors = 0

    N = len(data)

    Z = data @ W.T
    G = sigmoid(Z)

    predicted = np.argmax(G, axis=1)
    actual = np.argmax(targets, axis=1)
    errors = np.sum(predicted != actual)

    cm = metrics.confusion_matrix(actual, predicted)
    errRate = errors/N

    return cm, errRate

def trainAndTest(data, features, partitioning, alpha=0.01, iterations=4000):
    # Remove feature
    partitioning = [slice(0, 30), slice(30, 50)]
    X_N, T_N, X_T, T_T = partitionData(data, features, classes, partitioning)

    _, W = training(X_N, T_N, alpha, iterations)

     # Test for training set
    cm, errRate = test(X_N, T_N, W)
    plotConfusionMatrix(cm, classes.keys(), f'Training set \n error rate: {(errRate * 100):.2} %')

    # Test for test set
    cm, errRate = test(X_T, T_T, W)
    plotConfusionMatrix(cm, classes.keys(), f'Test set \n error rate: {(errRate * 100):.2} %')
    plt.show()

## Plotting functions

def plotConfusionMatrix(cm, classes, title=""):
    
    fig = plt.figure(figsize=(6,4))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Classified')
    plt.ylabel('True')
    if title != "":
        plt.title(title)
    return fig

## Data creating function
def partitionData(df: pd.DataFrame, features: list, classes: dict, partitioning: list[slice]):
    columns = features + ['Species']

    data = df[columns].to_numpy()
    
    training = partitioning[0]
    test = partitioning[1]
    setSize = 50

    setosa = data[:setSize]
    versicolor = data[setSize:2*setSize]
    virginica = data[2*setSize:]
    trainingData = np.concatenate((setosa[training], versicolor[training], virginica[training]))
    testData = np.concatenate((setosa[test], versicolor[test], virginica[test]))

    # make targets for each datapoint (_N for training and _T for test)
    T_N = np.array([classes[data[-1]] for data in trainingData])
    T_T = np.array([classes[data[-1]] for data in testData])

    # remove species information from datapoints
    X_N = trainingData[:,:-1].astype(np.float64)
    X_T = testData[:,:-1].astype(np.float64)

    # Add row of ones to match the case for C > 2 -> linear classifier: g = Wx, x = [x^T, 1]^T
    X_N = np.hstack((X_N, np.ones((X_N.shape[0], 1))))
    X_T = np.hstack((X_T, np.ones((X_T.shape[0], 1))))
    # print(X_N)
    
    return X_N, T_N, X_T, T_T


## class to target mapping
classes = {}
classes['Iris-setosa'] = np.array([1, 0, 0])
classes['Iris-versicolor'] = np.array([0, 1, 0])
classes['Iris-virginica'] = np.array([0, 0, 1])

## retriving data
features = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
columns = features + ['Species']
datafile = "Data/iris.data"
data = pd.read_csv(datafile, names=columns)


def task1B():
    # First 30 dataPoints as training, last 20 as test
    partitioning = [slice(0, 30), slice(30, 50)]
    X_N, T_N, _, _ = partitionData(data, features, classes, partitioning)
    iterations = 4000
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    tests = [training(X_N, T_N, alpha, iterations)[0] for alpha in alphas]

    # Plotting
    for test in tests:
        plt.plot(range(iterations + 1), test)

    plt.legend(alphas)
    plt.grid(True)
    plt.show()

def task1C():
    # First 30 dataPoints as training, last 20 as test
    partitioning = [slice(0, 30), slice(30, 50)]
    X_N, T_N, X_T, T_T = partitionData(data, features, classes, partitioning)
    iterations = 4000
    alpha = 0.01
    # Find weight matrix
    _, W = training(X_N, T_N, alpha, iterations)

    # Test for training set
    cm, errRate = test(X_N, T_N, W)
    plotConfusionMatrix(cm, classes.keys(), f'Training set \n error rate: {(errRate * 100):.2} %')

    # Test for test set
    cm, errRate = test(X_T, T_T, W)
    plotConfusionMatrix(cm, classes.keys(), f'Test set \n error rate: {(errRate * 100):.2} %')
    plt.show()

def task1D():
    # Last 30 dataPoints as training, first 20 as test
    partitioning = [slice(20, 50), slice(0, 20)]

    iterations = 4000
    alpha = 0.01

    trainAndTest(data, features, partitioning, alpha, iterations)


def task2A():
    # Make histogram plots
    sn.set_theme(style="whitegrid")

    fig, axs = plt.subplots(2, 2)
    for feature, ax in zip(features, axs.flat):
        sn.histplot(data=data, x=data[feature], kde=True, hue="Species", legend=ax==axs[1,0], ax=ax)
    
    plt.tight_layout()
    plt.show()

    # Remove most overlapping feature
    features.remove('Sepal width')

    ## Train and test data
    # First 30 samples for training, last 20 for test
    partitioning = [slice(0, 30), slice(30, 50)]
    iterations = 4000
    alpha = 0.01

    trainAndTest(data, features, partitioning, alpha, iterations)

def task2B():
    partitioning = [slice(0, 30), slice(30, 50)]
    iterations = 4000
    alpha = 0.01

    # Remove two most overlapping features
    features.remove('Sepal width')
    features.remove('Sepal length')

    trainAndTest(data, features, partitioning, alpha, iterations)

    # Remove two most overlapping features
    features.remove('Petal length')
    trainAndTest(data, features, partitioning, alpha, iterations)




# task1B()
# task1C()
# task1D()
# task2A()
task2B()

