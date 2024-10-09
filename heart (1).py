#!/usr/bin/env python
# coding: utf-8

# ## LOAD BASIC LIBRARIES 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# ## READING THE CSV FILE

# In[2]:


df = pd.read_csv('heart.csv')


# In[3]:


df


# ## EXPLORING THE DATA SET TO DERIVE USEFUL INFORMATION 

# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.shape


# this shows our data has 303 rows and 14 columns 

# In[7]:


df.columns


# In[8]:


df.describe()


# In[9]:


df.isnull().sum()


# In[10]:


df.info()

This shows that our data has no null values
# ## FINDING THE CORRELATION AMONG THE ATTRIBUTES

# In[11]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(), annot=True, cmap='rocket')


# We observe positive correlation between target and cp, thalach,slope and also negative correlation between target and sex, exang,ca,thai,oldpeak

# In[12]:


sns.pairplot(data=df)


# In[13]:


df.age.value_counts()[:]


# ## AGE PLOT WITH PEOPLE

# In[14]:


x=df.age.value_counts().index
y=df.age.value_counts().values


# In[15]:


plt.figure(figsize=(15,5))
sns.barplot(x,y,palette = 'rainbow')
plt.xlabel('Age')
plt.ylabel('Age counter')
plt.title('Age analysis')
plt.show


# ## HEART CHECK WITH AGE

# In[16]:


df['target'].value_counts()


# 165 cases of heart diseases

# In[17]:


plt.figure(figsize=(10,5))
x=df.restecg
y=df.age
plt.xlabel('ecg')
plt.ylabel('Age')
plt.title('Heartcheck with age')
sns.barplot(x,y,palette ='rocket')
plt.show()


# # THE VARIATION AS PER SE TYPE

# In[18]:


df['sex'].value_counts()


# 207 males and 96 females

# In[19]:


total_gender_count=len(df.sex)
male_count=len(df[df['sex']==1])
female_count=len(df[df['sex']==0])
print('Total Genders :',total_gender_count )
print('Male count :',male_count )
print('Female count:',female_count )


# In[20]:


#sex(1=male;0=female)
sns.countplot(df.sex, palette= 'rocket')
plt.show


# In[21]:


#male  state & target 1 & 0
male_target_on=len(df[(df.sex==1)&(df['target']==1)])
male_target_off=len(df[(df.sex==1)&(df['target']==0)])
sns.barplot(x=['Male Target On','Male Target Off'],y=[male_target_on,male_target_off], palette='rocket_r')
plt.xlabel('Male and Target state')
plt.ylabel('count')
plt.title('Gender state')
plt.show()


# In[22]:


#Female state & target 1 & 0
female_target_on=len(df[(df.sex==0)&(df['target']==1)])
female_target_off=len(df[(df.sex==0)&(df['target']==0)])
sns.barplot(x=['female Target On','female Target Off'],y=[female_target_on,female_target_off],palette='rocket_r')
plt.xlabel('Female and Target state')
plt.ylabel('count')
plt.title('Gender state')
plt.show()


# ## CHEST PAIN TYPE PLOT

# In[23]:


# chest pain type
df['cp'].value_counts() 


# In[24]:


#0  for  Good condition
#1 for slightly poor condition
#2 for Medium pain
#3 for Too bad condition
sns.countplot(df.cp, palette='magma')
plt.xlabel('Chest pain Type')
plt.ylabel('count')
plt.title('Chestpain VS count')
plt.show()


# ## BP, Density, Cholesterol, Blood Sugar Plot
# 

# ## Maximum heart rate, Exercise angina plot

# In[25]:


df2=df.loc[:,['trestbps','chol','cp','target']]


# In[26]:


df2


# In[27]:


sns.jointplot(x='trestbps',y='chol',kind='scatter',data=df2)
plt.show()


# In[28]:


sns.pairplot(df2)
plt.show()


# In[29]:


sns.pairplot(df2)
plt.show()


# ## Predict heart failure rate as per age,sex,and other issues.

# In[110]:


X = df.iloc[:, [0,3]].values
y = df.iloc[:,6].values


# In[111]:


#Splitting the data frame into test and train data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[112]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[113]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[114]:


y_pred = classifier.predict(X_test) 


# In[115]:


y_pred


# In[116]:


from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)


# In[117]:


cm


# In[118]:


# Visualising the Training set results  
from matplotlib.colors import ListedColormap  
x_set, y_set = X_train, y_train  
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),  
                     np.arange(start = x_set[:,1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),  
             alpha = 0.75, cmap = ListedColormap(('red', 'pink')))  
plt.xlim(X1.min(), X1.max())  
plt.ylim(X2.min(), X2.max())  
for i, j in enumerate(np.unique(y_set)):  
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
                c = ListedColormap(('purple', 'pink'))(i), label = j)  
plt.title('Logistic Regression(Training set)')  
plt.xlabel('VARIOUS ISSUE')  
plt.ylabel('HEART FAILURE RATE')  
plt.legend()  
plt.show()  


# In[119]:


# Visualising the Test set results  
from matplotlib.colors import ListedColormap  
x_set, y_set = X_test, y_test  
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),  
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))  
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),  
             alpha = 0.75, cmap = ListedColormap(('red', 'pink')))  
plt.xlim(X1.min(), X1.max())  
plt.ylim(X2.min(), X2.max())  
for i, j in enumerate(np.unique(y_set)):  
   plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],  
                c = ListedColormap(('purple', 'pink'))(i), label = j)  
plt.title('Logistic Regression (test set)')  
plt.xlabel('ISUUE')  
plt.ylabel('HEART FAILURE RATE')  
plt.legend()  
plt.show()  


# In[120]:


X= df.drop(['target'], axis=1)
y= df['target']


# In[121]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[122]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[123]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)


# In[124]:


y_pred = classifier.predict(X_test) 
y_pred


# In[125]:


from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)
cm


# In[126]:


TP=cm[0][0]
TN=cm[1][1]
FN=cm[1][0]
FP=cm[0][1]
print('Testing Acuracy:',(TN+TP)/(TP+TN+FN+FP))


# In[127]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[128]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

