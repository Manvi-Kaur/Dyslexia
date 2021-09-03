#!/usr/bin/env python
# coding: utf-8

# In[23]:


#from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
#import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import classification_report
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


data=pd.read_csv('labeled_dysx.csv')
y=data.Label
X=data.drop(['Label'],axis=1)


# In[9]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.8,random_state=10)


# In[10]:


sc=StandardScaler(copy=False)
sc.fit_transform(X_train)
sc.transform(X_test)


# In[18]:


n_est = {'n_estimators': [10, 100, 500, 1000]}
model = GridSearchCV(RandomForestClassifier(random_state = 0), n_est,scoring='f1_macro')
model.fit(X_train, y_train)
print("Best parameters set found on development set:")
print(model.best_params_)


# In[19]:


pred = model.predict(X_test)


# In[21]:


target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test, pred, target_names=target_names))


# In[24]:


pickle_out = open("model.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()


# In[ ]:




