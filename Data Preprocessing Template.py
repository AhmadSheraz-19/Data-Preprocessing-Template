# -*- coding: utf-8 -*-
"""
Data Preprocessing Template
"""
#importing the libraries

import numpy as np
import pandas as pd

#DataSet Importation

dataset=pd.read_csv("Data.csv")
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,3].values

#Missing Data handeling

from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])

#Encoding Catagorical DATA

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

#Dummy Variable

onehotencoder=OneHotEncoder(catagorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_Y=LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)

#Splittinf the dataset into the training set and the test set

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)

#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)


