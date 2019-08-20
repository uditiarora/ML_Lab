#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[48]:


y_data = [i for i in range(1,101)]
x_data = [y_data[i]*2 + 1 for i in range(0,100)]
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 0.4, random_state=0)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# In[49]:


#weights to minimize squared loss in case of linear model
w1 = ((y_train * x_train).mean() - y_train.mean() * x_train.mean())/((x_train * x_train).mean() - x_train.mean() * x_train.mean())
w0 = y_train.mean() - w1 * x_train.mean()
w0,w1


# In[50]:


y_pred = x_test*w1 + w0
y_pred, y_test


# In[51]:


#calculating loss
loss = 0
for i in range(len(y_pred)):
    loss += (y_test[i] - y_pred[i])**2
print(loss)


# In[52]:


#trying on a fourth order polynomial
x_data = np.array(x_data)
x_2 = x_data * x_data
x_3 = x_2 * x_data
x_4 = x_3 * x_data
x_data_new = []
for i in range(len(x_data)):
    x_data_new.append([x_data[i],x_2[i],x_3[i],x_4[i]])
x_data_new = np.array(x_data_new)
y_data = np.array(y_data)
x_train,x_test,y_train,y_test = train_test_split(x_data_new,y_data,test_size = 0.4, random_state=0)
y_test


# In[53]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[54]:


w1 = model.coef_[0]
w2 = model.coef_[1]
w3 = model.coef_[2]
w4 = model.coef_[3]
w0 = model.intercept_
print(w1,w2,w3,w4,w0)


# In[55]:


print(model.score(x_test,y_test))
loss = 0
for i in range(len(x_test)):
    loss += (y_test[i] - (w0 + w1 * x_test[i][0] + w2 * x_test[i][1] + w3 * x_test[i][2] + w4 * x_test[i][3]))**2
print(loss)


# In[56]:


# trying with the third model
a = 1
x_sin = np.sin(x_data - a)
x_data_new2 = []
for i in range(len(x_data)):
    x_data_new2.append([x_data[i],x_sin[i]])
x_train,x_test,y_train,y_test = train_test_split(x_data_new2,y_data,test_size = 0.4, random_state=0)
model2 = LinearRegression()
model2.fit(x_train,y_train)
w1 = model2.coef_[0]
w2 = model2.coef_[1]
w0 = model2.intercept_
print(w1,w2,w0)
print(model2.score(x_test,y_test))
loss = 0
for i in range(len(x_test)):
    loss += (y_test[i] - (w1 * x_test[i][0] + w2 * x_test[i][1] + w0))**2
print(loss)

