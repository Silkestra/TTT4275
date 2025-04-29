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

def training(X_D, T_D, alpha, iterations):
    C = len(T_D[0])
    D = len(X_D[0])
    W0 = np.zeros((C, D))
    W = [W0]
    
    MSEList = [MSE(X_D, T_D, W0)]
    for i in range(iterations):
        W.append(trainingStep(X_D, T_D, W[i], alpha))
        MSEList.append(MSE(X_D, T_D, W[i]))
    
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
    # Partition and retrive correct data
    X_D, T_D, X_T, T_T = getData(data, features, classes, partitioning)

    _, W = training(X_D, T_D, alpha, iterations)

     # Test for training set
    cm, errRate = test(X_D, T_D, W)
    trainFig = plotConfusionMatrix(cm, classes.keys(), f'Training set \n iterations: {iterations}, ' + r'$\alpha$' + f': {alpha} \n error rate: {(errRate * 100):.2} %')

    # Test for test set
    cm, errRate = test(X_T, T_T, W)
    testFig = plotConfusionMatrix(cm, classes.keys(), f'Test set \n iterations: {iterations}, ' + r'$\alpha$' + f': {alpha} \n error rate: {(errRate * 100):.2} %')
    return trainFig, testFig

## Plotting functions

def plotConfusionMatrix(cm, classes, title=""):
    
    fig = plt.figure(figsize=(8,6))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Classified')
    plt.ylabel('True/label')
    if title != "":
        plt.title(title)
    return fig

## Data creating function
def getData(df: pd.DataFrame, features: list, classes: dict, partitioning: list[slice]):
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
    T_D = np.array([classes[data[-1]] for data in trainingData])
    T_T = np.array([classes[data[-1]] for data in testData])

    # remove species information from datapoints
    X_D = trainingData[:,:-1].astype(np.float64)
    X_T = testData[:,:-1].astype(np.float64)

    # Add row of ones to match the case for C > 2 -> linear classifier: g = Wx, x = [x^T, 1]^T
    X_D = np.hstack((X_D, np.ones((X_D.shape[0], 1))))
    X_T = np.hstack((X_T, np.ones((X_T.shape[0], 1))))
    
    return X_D, T_D, X_T, T_T

## Saving function
def saveFigsAsPDF(figures, names: list[str], task: str, dirr="Plots/"):
    saveNames = [task + "_" + name + ".pdf" for name in names]

    for fig, saveName in zip(figures, saveNames):
        fig.savefig(dirr + saveName)


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


def task1B(save=False):
    # First 30 dataPoints as training, last 20 as test
    partitioning = [slice(0, 30), slice(30, 50)]
    X_D, T_D, _, _ = getData(data, features, classes, partitioning)
    iterations = 4000
    alphas = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    tests = [training(X_D, T_D, alpha, iterations)[0] for alpha in alphas]

    # Plotting
    fig = plt.figure()
    for test in tests:
        plt.plot(range(iterations + 1), test)

    labels = [r'$\alpha$' + f': {alpha}' for alpha in alphas]
    plt.legend(labels)
    plt.xlabel('Number of iterations')
    plt.ylabel('Mean square error')
    plt.title('Training of LDC')
    plt.grid(True)

    if save:
        figs = [fig]    
        # Name for figures when saving 
        plotNames = ["alphaTraining"]
        saveFigsAsPDF(figs, plotNames, "task1B")
    return

def task1C(save=False):
    # First 30 dataPoints as training, last 20 as test

    partitioning = [slice(0, 30), slice(30, 50)]
    X_D, T_D, X_T, T_T = getData(data, features, classes, partitioning)
    iterations = 4000
    alpha = 0.01

    trainCovMatFig, testCovMatFig = trainAndTest(data, features, partitioning, alpha, iterations)

    trainCovMatFig.canvas.manager.set_window_title("training, first 30 last 20")
    testCovMatFig.canvas.manager.set_window_title("testing, first 30 last 20")

    if save:
        figs = [trainCovMatFig, testCovMatFig]  
        # Name for figures when saving 
        plotNames = ["train", "test"]
        saveFigsAsPDF(figs, plotNames, "task1C")
    return
    

def task1D(save=False):
    # Last 30 dataPoints as training, first 20 as test
    partitioning = [slice(20, 50), slice(0, 20)]

    iterations = 4000
    alpha = 0.01

    trainCovMatFig, testCovMatFig = trainAndTest(data, features, partitioning, alpha, iterations)

    trainCovMatFig.canvas.manager.set_window_title("training, last 30 first 20")
    testCovMatFig.canvas.manager.set_window_title("testing, last 30 first 20")

    if save:
        figs = [trainCovMatFig, testCovMatFig]  
        # Name for figures when saving 
        plotNames = ["train", "test"]
        saveFigsAsPDF(figs, plotNames, "task1D")
    return



def task2A(save=False):
    # Make histogram plots
    sn.set_theme(style="whitegrid")

    histFig, axs = plt.subplots(2, 2)
    for feature, ax in zip(features, axs.flat):
        sn.histplot(data=data, x=data[feature], kde=True, hue="Species", legend=ax==axs[1,0], ax=ax)
    
    # Remove y-axis labels for the right plots
    axs[0, 1].set_ylabel('')
    axs[1, 1].set_ylabel('')
    plt.tight_layout()

    # Remove most overlapping feature
    newFeatures = features.copy()
    newFeatures.remove('Sepal width')

    ## Train and test data
    # First 30 samples for training, last 20 for test
    partitioning = [slice(0, 30), slice(30, 50)]
    iterations = 4000
    alpha = 0.01

    trainCovMatFig, testCovMatFig = trainAndTest(data, newFeatures, partitioning, alpha, iterations)

    trainCovMatFig.canvas.manager.set_window_title("training, one features removed")
    testCovMatFig.canvas.manager.set_window_title("testing, one features removed")

    if save:
        figs = [histFig, trainCovMatFig, testCovMatFig]  
        # Name for figures when saving 
        plotNames = ["hist", "train", "test"]
        saveFigsAsPDF(figs, plotNames, "task2A")
    return

def task2B(save=False):
    partitioning = [slice(0, 30), slice(30, 50)]
    iterations = 4000
    alpha = 0.01

    # Remove two most overlapping features
    newFeatures = features.copy()
    newFeatures.remove('Sepal width')
    newFeatures.remove('Sepal length')

    trainCovMatFigA, testCovMatFigA = trainAndTest(data, newFeatures, partitioning, alpha, iterations)

    trainCovMatFigA.canvas.manager.set_window_title("training, two features removed")
    testCovMatFigA.canvas.manager.set_window_title("testing, two features removed")

    # Remove three most overlapping features
    newFeatures.remove('Petal length')
    trainCovMatFigB, testCovMatFigB = trainAndTest(data, newFeatures, partitioning, alpha, iterations)

    trainCovMatFigB.canvas.manager.set_window_title("training, three features removed")
    testCovMatFigB.canvas.manager.set_window_title("testing, three features removed")

    if save:
        figs = [trainCovMatFigA, testCovMatFigA, trainCovMatFigB, testCovMatFigB]
        # Name for figures when saving 
        plotNames = ["trainA", "testA", "trainB", "testB"]
        saveFigsAsPDF(figs, plotNames, "task2B")
    return

# Toggle saving of figures
save = False

## Choose which task to run
task1B(save)
task1C(save)
task1D(save)
task2A(save)
task2B(save)

plt.show()



