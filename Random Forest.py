import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow import keras


data = pd.read_csv(r'D:\classification public data of space observations\Skyserver_SQL2_27_2018 6_51_39 PM.csv')
data.head()
data.shape # (10000, 18)

####### move the target data to the end of the dataframe ############
# def ToTheEnd(df,column):
#     Target_data=df[column] # class
#     df=df.drop([column],axis=1)
#     df[column]=Target_data
#     return df

# data=ToTheEnd(data,'class')

#########################

data.columns
# Drop the object id column from the analysis because they are unnecessary.
data.drop(['objid','specobjid'], axis=1, inplace=True)
data.head()# (10000, 16)
sns.countplot(data['class'])
data['class'].value_counts()
def change_category_to_number(classCat):
    if classCat=='STAR':# == mean does equal
        return 0
    elif classCat=='GALAXY':
        return 1
    else:
        return 2

# assign a numerical value to the categorical field of class, by using the above function
data['classCat'] = data['class'].apply(change_category_to_number)# make a new column (classCat)
data.head()
x= data['classCat']

data.drop(['run','rerun','camcol','field','class'],axis=1,inplace=True) # Useless data
data.head()
data.shape # (10000, 12)
data.dtypes
data['classCat'].value_counts()





################################### splite the data and scaling
X = data.drop('classCat',axis=1)
y = data['classCat']
X.shape
y.shape

##### scaling data
from sklearn.preprocessing import StandardScaler, MinMaxScaler

#ss = StandardScaler()
#X = ss.fit_transform(X)
minmax = MinMaxScaler()
X = minmax.fit_transform(X)
y = y.values.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y = enc.fit_transform(y).toarray()
y.shape # (10000, 3)


### Perform train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=128)
X_train.shape
y_train.shape

########################## Random Forest #########################################

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, y_train)
# ### prediction on test data
y_pred = random_forest.predict(X_test)

Test_acc_random_forest = round(random_forest.score(X_test, y_test) * 100, 3)
print("Score ",Test_acc_random_forest)
