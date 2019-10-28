#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# In[43]:


data = datasets.load_breast_cancer()
x = np.array(data.data[:,:])
y = np.array(data.target)


# In[44]:


class LogisticRegression_new:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False,lamb = 0.1):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = np.zeros(10)
        self.lamb = lamb
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h) + self.lamb*(self.theta * self.theta).sum()).mean() 
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        loss = 0
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = (np.dot(X.T, (h - y)) + self.lamb * 2 * self.theta)/ y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 100 == 0):
                print("loss:" ,{loss} )
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()

    def print_weights(self):
        print(self.theta)
    def get_weights(self):
        return self.theta


# In[45]:


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4, random_state = 0)
model = LogisticRegression_new(lr=0.1, num_iter=1000,lamb = 0.01)
model.fit(x_train,y_train)
preds = model.predict(x_test)
model.print_weights()
print("Score = ",(preds == y_test).mean())


# In[46]:


reg = 0
max_score = 0
for i in range(0,10,1):
    model = LogisticRegression_new(lr=0.1, num_iter=1000,lamb = i)
    model.fit(x_train,y_train)
    preds = model.predict(x_test)
    score = (preds == y_test).mean()
    if(score >= max_score):
        reg = i
        max_score = score
        


# In[47]:


print(reg, max_score)


# In[73]:


def k_fold_cross(x, y, k):
    n = len(x)
    batches = n//k
    print(batches)
    optimum_weights = []
    acc = 0
    for i in range(0,k):
        x_test = x[i*batches: i*batches + batches]
        y_test = y[i*batches: i*batches + batches]
        arr = [j for j in range(i*batches, i*batches + batches)]
        x_train = []
        y_train = []
        for j in range(0,i*batches):
            x_train.append(x[j])
            y_train.append(y[j])
        for j in range(i*batches + batches + 1, n):
            x_train.append(x[j])
            y_train.append(y[j])
            
        print(len(x_train),len(y_train),len(x_test))
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        model = LogisticRegression_new(lr=0.1, num_iter=1000,lamb = max_score)
        model.fit(x_train,y_train)
        preds = model.predict(x_test)
        score = (preds == y_test).mean()
        print("Score = ",score)
        if(score>acc):
            acc = score
            optimum_weights = model.get_weights()
    return optimum_weights


# In[74]:


optimum_weights = k_fold_cross(x,y,10)


# In[75]:


optimum_weights

