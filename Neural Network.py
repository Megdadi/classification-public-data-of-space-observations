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



####################### Neural Network using Tensorflow and Keras ##################

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(128, input_dim=11, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))

model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_split = 0.1, epochs=40, batch_size=32)

############## evaluating

y_predict = model.predict(X_test)

prediction = []
test = []
for i in range(len(y_test)): 
    prediction.append(np.argmax(y_predict[i]))
    test.append(np.argmax(y_test[i]))
### compartion
prediction = pd.Series(prediction)
test = pd.Series(test)
compare = pd.concat([prediction, test], axis=1, keys=['prediction_test', 'actual_test'])
################# confusion_matrix ########################
from sklearn.metrics import confusion_matrix
# confusion_matrix(test, prediction, labels=[0, 1, 2])
#### or
cm=confusion_matrix(prediction,test)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
classes =['STAR', 'GALAXY', 'QSO']
plot_confusion_matrix(cm, classes)

######## metrics ####################3
from sklearn.metrics import classification_report
print ('Report : ')
print (classification_report(test, prediction,target_names=classes,digits=5 ))

