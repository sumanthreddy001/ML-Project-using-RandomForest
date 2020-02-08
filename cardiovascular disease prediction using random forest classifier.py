#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
data=pd.read_csv("D:/ml datasets/cardio.csv",delimiter=';')
#loading the data


# In[7]:


data.head()


# In[4]:


data=pd.DataFrame(data)
print("number of rows are "+str(len(data.axes[0])))
print("number of coulmns are "+str(len(data.axes[1])))


# In[6]:


data=data.drop_duplicates()
#drop the duplicates


# In[5]:


data=data.dropna()
#drop the rows which are not assigned(empty)


# In[11]:


# data x will be our inputs
# data y will be our outputs
x=data.drop(["id","cardio"],axis="columns")
#having id is of no use ,so we are removing it from our dataframe

#removing target attribute from inputs
y=data.cardio


# In[12]:


x.head()


# In[13]:


y.head()


# In[15]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25)
#here we are spliting our data in to train and test data
#traindata=75% of data
#testdata=25% of data
#the main reason we split the data in to test and train is we use train data to  build the model and we test data to check the accuracy of our model.


# In[16]:


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=200,random_state=0)
model.fit(X_train,Y_train)
#building the model


# In[18]:


Y_predict=model.predict(X_test)
Y_predict
#predicting by giving test data as input


# In[20]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_predict,Y_test)
#confusion matrix


# In[22]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,Y_predict)
#checking the accuracy of our model

