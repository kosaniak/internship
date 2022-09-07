#Let's import libraries, obtain and view the data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('internship_train.csv')
print(df.info())
df_test = pd.read_csv('internship_hidden_test.csv')
print(df_test.info())

df.describe()
df.head()
df.shape
df_test.shape
df_test.head()

for column in list(df.columns):
    col = (df[column] == 0).sum()
    print('Num of zeros in column {}: {}'.format(column,col))

df.isnull().sum()

#Data distribution
df.hist(figsize=(30,22))
plt.show()

#Correlation
corr_matrix_pearson = df.corr(method='pearson')
plt.figure(figsize=(40,30))
sns.heatmap(corr_matrix_pearson, annot = True)

#Outliers detection
df.plot(kind="box",subplots=True,layout=(18,3),figsize=(40,50));

#Outliers detection
fig,ax = plt.subplots(figsize=(60,20),facecolor='white')
sns.boxplot(data = df , ax = ax ,width = 0.7 , fliersize = 5)

X =  df.drop(["target"],axis = 1)
y = df["target"]

# Split into training and validation groups and scaling
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_predict_test = scaler.fit_transform(df_test)



