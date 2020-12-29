#Artificial Neural Network
#Part-1: Data Preprocessing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv('Churn_Modelling_ANN.csv')

#independent variable
X=dataset.iloc[:, 3:13].values
#dependent variable
Y=dataset.iloc[:, 13].values

#encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1=LabelEncoder()
X[:, 1]=labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2=LabelEncoder()
X[:, 2]=labelencoder_X_2.fit_transform(X[:, 2])
#Create Dummy Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer([("Country", OneHotEncoder(), [1])], remainder='passthrough')
X=ct.fit_transform(X)
X = X[:, 1:]

#splitting the dataset into training set & teasting set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y, test_size= 0.2, random_state=0 )

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Part-2: Now Let's make the ANN!
#Import the keras libraries and Packages
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
'''Sequintial package use for initialization of neural network'''
classifier = Sequential()

# Adding the Input Layer and first hidden layer
'''units means no. of nodes in hidden layers i.e. average of input nodes and output node(variables)
   eg. 11+1/2 = 6
   kernal_initializer used to give value of weights near to zero.
   activation relu means rectifier activation
   input_dim means no. of nodes for input independent variables
'''
classifier.add(Dense(units = 6, kernel_initializer   = 'uniform', activation = 'relu', input_dim = 11))

# Adding the Second hidden layer
classifier.add(Dense(units = 6, kernel_initializer   = 'uniform', activation = 'relu'))

#Adding the output Layer
'''Here,no. of nodes use in output layer is 1.Therefore units = 1.
   If dependent variable have more than two catogories then use activation = 'softmax' 
'''
classifier.add(Dense(units = 1, kernel_initializer   = 'uniform', activation = 'sigmoid'))

#Compiling The ANN
'''If dependent variable have more than two catogories then use loss = 'categorical_crossentropy'
'''    
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 10, epochs = 100)

#Part-3: Making the predictions and evaluating the model
#Predecting the test set result
y_pred =  classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Making an confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

    


    










