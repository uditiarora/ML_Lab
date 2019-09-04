#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
import matplotlib.patches as mpatches


# In[60]:


data_train = pd.read_csv('zip.train.txt', header = None,delimiter = " ")
data_test = pd.read_csv('zip.test.txt',header = None,delimiter=" ")
data_test.head(5)


# In[61]:


data_train_filter = data_train[data_train.iloc[:, 0].isin([1,4])]
data_test_filter = data_test[data_test.iloc[:, 0].isin([1,4])]


# In[62]:


sample1 = np.array(data_train_filter.iloc[4, :])
sample1 = sample1[1:257].reshape((16,16))
plt.imshow(sample1)
plt.show()


# In[63]:


sample2 = np.array(data_test_filter.iloc[2, :])
sample2 = sample2[1:257].reshape((16,16))
plt.imshow(sample2)
plt.show()


# In[64]:


def condition(df):
    temp = np.array(df.iloc[1:257])
    temp = temp.reshape((16,16))
    temp2 = temp[:,::-1]
    
    temp3 = (temp2-temp)
    return (16 * 16 - np.count_nonzero(temp3))/(16*16.0)
def classify(df):
    if(df.iloc[0] == 1):
        return 1
    else:
        return 0
def sum1(df):
    return df.iloc[1:257].sum()/(16*16)
data_train_filter['symmetry'] = data_train_filter.apply(condition,axis=1)
data_train_filter['intensity'] = data_train_filter.apply(sum1, axis=1)
data_train_filter["class"] = data_train_filter.apply(classify,axis = 1)
data_train_filter.head()


# In[65]:


data_test_filter['symmetry'] = data_test_filter.apply(condition,axis=1)
data_test_filter['intensity'] = data_test_filter.apply(sum1, axis=1)
data_test_filter['class'] = data_test_filter.apply(classify,axis = 1)
data_test_filter.head()


# In[66]:


data_train_new = data_train_filter.iloc[:, 0:1]
data_train_new["symmetry"] = data_train_filter["symmetry"]
data_train_new["intensity"] = data_train_filter["intensity"]
data_train_new["class"] = data_train_filter["class"]

data_test_new = data_test_filter.iloc[:, 0:1]
data_test_new["symmetry"] = data_test_filter["symmetry"]
data_test_new["intensity"] = data_test_filter["intensity"]
data_test_new["class"] = data_test_filter["class"]

arr = np.arange(0,len(data_train_new))
colr = ['red' if l == 4 else 'blue' for l in data_train_new.iloc[:,0]]
plt.scatter(arr, data_train_new["intensity"],color = colr)


# In[67]:



arr = np.arange(0,10)
colr = ['red' if l == 4 else 'blue' for l in data_train_new.iloc[:,0]]
y = data_train_new["symmetry"][:10]
plt.scatter(arr, y,color = colr)


# In[68]:


x_train = np.array(data_train_new.loc[:,['intensity','symmetry']])
y_train = np.array(data_train_new["class"])

x_test = np.array(data_test_new.loc[:,['intensity','symmetry']])
y_test = np.array(data_test_new["class"])


# In[69]:


# def sigmoid(output):
#     return (1/(1 + math.exp(-output)))

# def slopes(x,y,m1,m2,b):
#     x1 = np.array(x['intensity'])
#     x2 = np.array(x['symmetry'])
#     y1 = np.array(y)
#     t1,t2,t3 = 0,0,0
#     for i in range(len(x1)):
#         output = b + m1 * x1[i] + m2 * x2[i]
#         pred = sigmoid(output)
#         error = pred - y1[i]
#         t1 = t1 + error * x1[i]
#         t2 = t2 + error * x2[i]
#         t3 = t3 + error
        
#     s1 = t1/len(x1)
#     s2 = t2/len(x1)
#     s3 = t3/len(x1)
#     return s1,s2,s3
# def error(y_pred, y_actual):
#     return np.array((np.array(y_pred) - np.array(y_actual))**2).sum()/len(y_pred)

# def logistic_regression(x,y,epochs,lr):
#     m1,m2,b = 0,0,0
#     for i in range(epochs):
#         s1,s2,s3 = slopes(x,y,m1,m2,b)
#         m1 = m1 - lr * s1
#         m2 = m2 - lr * s2
#         b = b - lr * s3
#     return m1,m2,b


# def accuracy(y_pred,y_actual):
#     temp = np.array(y_pred - y_actual)
#     print(temp)
#     corr = len(y_pred) - np.count_nonzero(temp)
#     return float(corr)/len(y_pred)


# m1,m2,b = logistic_regression(x_train,y_train,10000,0.01)
# print("Weights = ",m1,m2,b)
# y_pred_feed = np.array(m1 * np.array(x_train['intensity']) + m2 * np.array(x_train['symmetry']) + b )
# y_pred = []
# for i in range(len(y_pred_feed)):
#     y_pred.append(sigmoid(y_pred_feed[i]))
# y_pred_feed


# In[70]:


class LogisticRegression_new:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.theta = np.zeros(10)
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        loss = 0
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print("loss:" ,{loss} )
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()

    def print_weights(self):
        print(self.theta)
def error(h,y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
model = LogisticRegression_new(lr=0.1, num_iter=1000)
model.fit(x_train,y_train)
preds = model.predict(x_test)
model.print_weights()
print("Score = ",(preds == y_test).mean())
print("Error = ",error(preds,y_test))


# In[71]:


from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
print("Weights = ",logisticRegr.coef_, logisticRegr.intercept_)
print("Score = ",logisticRegr.score(x_test,y_test))


# In[75]:


data_test_new['sym_3'] = data_test_new['symmetry']**3 
data_test_new['int_3'] = data_test_new['intensity']**3
data_test_new['int_2_sym'] = data_test_new['intensity']**2 * data_test_new['symmetry']
data_test_new['sym_2_int'] = data_test_new['symmetry']**2 * data_test_new['intensity']


# In[76]:


data_train_new['sym_3'] = data_train_new['symmetry']**3 
data_train_new['int_3'] = data_train_new['intensity']**3
data_train_new['int_2_sym'] = data_train_new['intensity']**2 * data_train_new['symmetry']
data_train_new['sym_2_int'] = data_train_new['symmetry']**2 * data_train_new['intensity']


# In[82]:


x_train = np.array(data_train_new.loc[:,['sym_3','int_3','int_2_sym','sym_2_int']])
y_train = np.array(data_train_new["class"])

x_test =np.array( data_test_new.loc[:,['sym_3','int_3','int_2_sym','sym_2_int']])
y_test = np.array(data_test_new["class"])


# In[83]:


model2 = LogisticRegression_new(lr=0.1, num_iter=1000)
model2.fit(x_train,y_train)
preds = model2.predict(x_test)
model2.print_weights()
print("Score = ",(preds == y_test).mean())
print("Error = ",error(preds,y_test))


# In[84]:


logisticRegr2 = LogisticRegression()
logisticRegr2.fit(x_train, y_train)
print("Weights = ",logisticRegr2.coef_, logisticRegr.intercept_)
print("Score = ",logisticRegr2.score(x_test,y_test))

