#!/usr/bin/env python
# coding: utf-8

# In[76]:


import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt


# In[77]:


n = 100
a = 10
b = 100
c = -20
# defining the line as ax + by + c = 0
f = [a,b,c]
class1_list = []
class2_list = []
for i in range(0,n//2):
    x = random.randint(-1000,1000)
    y = random.randint(-1000,1000)
    while (a*x + b*y + c >= 0):
        x = random.randint(-1000,1000)
        y = random.randint(-1000,1000)
    class1_list.append([x,y,-1])
    
for i in range(0,n//2):
    x = random.randint(-1000,1000)
    y = random.randint(-1000,1000)
    while (a*x + b*y + c < 0):
        x = random.randint(-1000,1000)
        y = random.randint(-1000,1000)
    class2_list.append([x,y,1])
data_list = class2_list + class1_list
data = np.array(data_list)
print(class1_list, class2_list, data)


# In[78]:


fig = plt.figure(figsize = (10,10))
plt.scatter(data[:,0], data[:,1],c = data[:,2])


# In[79]:


#implementing algorithm
w0 = random.randint(-100,100)
w1 = random.randint(-100,100)
w2 = random.randint(-100,100)
flag = False
while flag != True :
    flag = True
    for i in range(0,n):
        x = data[i][0]
        y = data[i][1]
        l = data[i][2]
        temp = w1*x + w2*y + w0
        if l==1 and temp < 0:
            flag = False
            w1 = w1 + x
            w2 = w2 + y
            w0 = w0 + 1
            break
        if (l == -1 and temp>=0):
            flag = False
            w1 = w1 - x
            w2 = w2 - y
            w0 = w0 - 1
            break
print(w0,w1,w2)


# In[80]:


pred = [0 for i in range(0,n)]
for i in range(0,n):
    x = data[i][0]
    y = data[i][1]
    temp = w1*x + w2*y + w0
    if(temp<0):
        pred[i] = -1
    if(temp>=0):
        pred[i] = 1
print(pred)


# In[ ]:





# In[81]:


fig = plt.figure()
ax = plt.axes()

x = np.linspace(-1000, 1000, 1000)
ax.plot(x,(-c - a * x)/b)
ax.plot(x, (-w0 - w1 * x)/w2)

