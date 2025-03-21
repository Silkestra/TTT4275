import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


datafiles1 = ["Iris/Data/class_1", "class_2", "class_3"]
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']
# columns2 = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']

#data = {}
#for file in datafiles:
#    data[file] = 

datafile = "Iris/Data/iris.data"
dataframe = pd.read_csv(datafile, names=columns)

def make_dataframe(datafiles, coloumns):
    dataframe = {}
    for file in datafiles:
        dataframe[file] = pd.read_csv(file, sep=',', names=coloumns)
    
    return dataframe
    
#def plot_histogram(data):
    sn.histplot(data)
# sn.histplot(dataframe)
print(dataframe["Iris-setosa"])

dataframe[columns[1]]
sn.pairplot(dataframe, hue='Species')

plt.show()