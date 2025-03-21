import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


class LDC:

    def __init__(self, training_set: pd.DataFrame, test_set: pd.DataFrame, features: list[str], T_N, alpha):
        
        
        self.training_set: pd.DataFrame = training_set    #data frame
        self.test_set: pd.DataFrame = test_set            #data frame
        self.features: list[str] = features
        self.T_N: dict[str, np.ndarray] = T_N                      #target set
        self.alpha = alpha 
        self.C = len(T_N)
        self.D = len(features)
        self.w0 = np.zeros(self.C)
        self.W = np.zeros((self.C, self.D))  #Weight matrix for linear classifier
                                #Bias matrix for linear classifier
        self.mse_list = []
        self.iteration = 0
        self.alpha = alpha

        self.N = len(test_set - 1)
        self.x = np.array(test_set.loc[:, features])
        self.t = np.array([test_set.loc[:, 'Species']])

    def discriminant(self, xk):
        return self.W @ xk


    def sigmoid(self,z):
        for i in range(len(z)):
            z[i] = 1/(1+np.exp(-z[i]))
        return z

    def mse(self):
        mse = 0
        for _, data in self.training_set.iterrows():
            xk = np.array([data[feature] for feature in self.features])
            tk = self.T_N[data['Species']]
            zk = self.discriminant(xk)
            gk = self.sigmoid(zk)
            mse += 1/2 * (gk - tk).T @ (gk -tk)
            # print(f'mse: {mse}')
        return mse
    
    def grad_w_mse(self):
        grad = 0
        for _, data in self.training_set.iterrows():
            xk = np.array([data[feature] for feature in self.features])
            #print(xk)
            tk = self.T_N[data['Species']]
            #print(tk)
            zk = self.discriminant(xk)
            gk = self.sigmoid(zk)
            
            grad += np.outer((gk - tk) * gk * (np.ones(len(gk)) - gk), np.transpose(xk))
        
        return grad

    def training_step(self):
        # print(self.grad_w_mse())
        print(f'W: {self.W}')
        return self.W - self.alpha * self.grad_w_mse()

    def train(self, epsilon=0.01):
        self.mse_list.append(self.mse())
        self.W = self.training_step() 
        self.iteration += 1
        self.mse_list.append(self.mse())

        # while (abs(self.mse_list[self.iteration] - self.mse_list[self.iteration-1]) > epsilon):
        while (self.iteration < 1000):
            if (abs(self.mse_list[self.iteration] - self.mse_list[self.iteration-1]) < epsilon):
                return 
            self.iteration += 1
            print(self.iteration)
            self.W = self.training_step()
            self.mse_list.append(self.mse()) 
            

    
## Load data 
columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']

datafile = "Iris/Data/iris.data"
dataframe = pd.read_csv(datafile, names=columns)

T_N = {}
T_N['Iris-setosa'] = np.array([1, 0, 0])
T_N['Iris-versicolor'] = np.array([0, 1, 0])
T_N['Iris-virginica'] = np.array([0, 0, 1]) 

iris = LDC(dataframe, dataframe, columns[:-1], T_N, 0.01)

iris.train()

print(iris.iteration)
print(iris.mse_list)

plt.plot(range(iris.iteration), iris.mse_list[:-1])
plt.show()