#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder


from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,precision_score, recall_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from sklearn.svm import SVC


# In[54]:


warnings.filterwarnings('ignore')


# In[55]:


data = pd.read_csv('HR_Dataset.csv')


# In[56]:


data.sample(5)


# In[57]:


data.columns


# In[58]:


data.rename(columns={'Departments ':'departments'},inplace=True)


# In[59]:


data.columns


# In[ ]:





# In[11]:


data.shape #row,col


# In[12]:


import platform; print(platform.platform())
import sys; print("Python", sys.version)
import numpy; print("NumPy", numpy.__version__)
import scipy; print("SciPy", scipy.__version__)
import sklearn; print("Scikit-Learn", sklearn.__version__)


# In[13]:


pip install scikit-learn==0.23.1


# In[14]:


data.info()


# In[ ]:





# 

# In[60]:


data.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


data.duplicated().any()


# In[ ]:





# In[61]:


data = data.drop_duplicates()


# In[90]:


data['left'].value_counts().plot(kind='pie')


# In[ ]:





# In[64]:


X = data.drop(columns=['left'])


# In[65]:


y = data['left']


# In[91]:


preprocessor = ColumnTransformer(transformers=[
    ('num',StandardScaler(),['satisfaction_level',
                            'last_evaluation',
                            'number_project',
                            'average_montly_hours',
                            'time_spend_company',
                            'Work_accident','promotion_last_5years']),
    ('nominal',OneHotEncoder(),['departments']),
    ('ordinal',OrdinalEncoder(),['salary'])
    
    
],remainder='passthrough')

# standardScaler in all numerical feature so that no 2 numerical feature should be given priority based on there scale
# nomial feature--> non ordering features
# ordinal--> order matters for this feature
# all the other colums will be preserved as it is
# column transformer is a package that combine different preprocessing step on different column into one


# In[67]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)


# In[69]:


def model_scorer(model_name,model):
    
    output=[]
    
    output.append(model_name)
    
    pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',model)])
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)
    
    pipeline.fit(X_train,y_train)
    
    y_pred = pipeline.predict(X_test)
    
    output.append(accuracy_score(y_test,y_pred))
    
    output.append(precision_score(y_test,y_pred))
    
    output.append(recall_score(y_test,y_pred))
    
    return output


# In[74]:


model_dict={
    'log':LogisticRegression(),
    'decision_tree':DecisionTreeClassifier(),
    'random_forest':RandomForestClassifier(),
    'XGB':XGBClassifier(),
    'SVM' : SVC()
    
}


# In[75]:


model_output=[]
for model_name,model in model_dict.items():
    model_output.append(model_scorer(model_name,model))


# In[76]:


model_output


# In[77]:


preprocessor = ColumnTransformer(transformers=[
    ('num',StandardScaler(),['satisfaction_level',
                            'last_evaluation',
                            'number_project',
                            'average_montly_hours',
                            'time_spend_company',
                            'Work_accident','promotion_last_5years']),
    ('nominal',OneHotEncoder(),['departments']),
    ('ordinal',OrdinalEncoder(),['salary'])
    
    
],remainder='passthrough')


# In[78]:


pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',RandomForestClassifier())
    
])


# In[79]:


pipeline.fit(X,y)


# In[ ]:





# In[ ]:





# In[ ]:





# In[80]:


sample = pd.DataFrame({
   'satisfaction_level':0.5,
   'last_evaluation':0.53,
    'number_project':2,
    'average_montly_hours':100,
    'time_spend_company':2,
    'Work_accident':0,
    'promotion_last_5years':1,
    'departments':'sales',
    'salary':'low'
    
    
},index=[0])


# In[87]:


result = pipeline.predict(sample)

if result == 1:
    print("An Employee may leave the company")
else:
    print("An Employee may stay with the company")


# In[ ]:





# In[82]:


import pickle


# In[83]:


with open('pipeline.pkl','wb') as f:
    pickle.dump(pipeline,f)


# In[84]:


with open('pipeline.pkl','rb') as f:
    pipeline_saved = pickle.load(f)


# In[86]:


result = pipeline_saved.predict(sample)

if result == 1:
    print("An Employee may leave the company")
else:
    print("An Employee may stay with the company")


# In[ ]:




