#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


import tensorflow as tf


# In[3]:


tf.__version__


# In[4]:


print("hello")


# In[5]:


import keras 


# In[6]:


keras.__version__


# In[7]:


import numpy as np


# In[8]:


import pandas as pd


# In[9]:


#importing the dataset


# In[10]:


df = pd.read_excel("C:\\Users\\Sricharan Reddy\\Downloads\\energy.xlsx")


# In[11]:


df.head()


# In[12]:


df.info()


# In[13]:


#here the output variable is pe


# In[14]:


df.isnull().sum()


# In[15]:


df.duplicated().sum()


# In[16]:


#df.drop_duplicates(inplace = True)


# In[17]:


df.duplicated().sum()


# In[18]:


#no duplicates now 


# In[19]:


df.info()


# # Building multi linear regression model 

# In[20]:


#now lets build a linear regression model and check 


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


x = df.drop(columns = ['PE'])


# In[23]:


y = df['PE']


# In[24]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state = 9)


# In[25]:


from sklearn.linear_model import LinearRegression


# In[26]:


model = LinearRegression()


# In[27]:


model.fit(x_train,y_train)


# In[28]:


model.coef_


# In[29]:


model.intercept_


# In[30]:


y_train_pred = model.predict(x_train)


# In[31]:


y_test_pred = model.predict(x_test)


# In[32]:


from sklearn.metrics import mean_squared_error


# In[33]:


train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))


# In[34]:


test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))


# In[35]:


print(train_rmse,test_rmse)


# In[36]:


from sklearn.metrics import r2_score
print(r2_score(y_train,y_train_pred))
print(r2_score(y_test,y_test_pred))


# # Building artificial neural network model

# In[37]:


from keras.models import Sequential


# In[38]:


from keras.layers import Dense


# In[39]:


model = Sequential()


# In[40]:


model.add(Dense(input_dim=4,units = 6,activation= 'relu',kernel_initializer='uniform'))


# In[41]:


model.add(Dense(units = 6,activation= 'relu',kernel_initializer='uniform'))


# In[42]:


#model.add(Dense(units = 6,activation= 'relu',kernel_initializer='uniform'))


# In[43]:


model.add(Dense(units = 1,activation= 'relu',kernel_initializer='uniform'))


# In[44]:


model.compile(optimizer='adam',loss ='mean_squared_error')


# In[45]:


model.fit(x_train,y_train,batch_size=6,epochs=100)


# In[46]:


y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)


# In[47]:


train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))


# In[48]:


print(train_rmse,test_rmse)


# In[ ]:





# In[49]:


print(r2_score(y_train,y_train_pred))
print(r2_score(y_test,y_test_pred))


# In[ ]:




