import numpy as np
import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('odi_cricket_dataset.csv')
categorical_data = pd.get_dummies(dataset.iloc[:,1:4])
X = np.zeros((588,131),dtype='float64')
X[:,:130]=categorical_data.values
X[:,130:131]=dataset.iloc[:,6:7].values

Y = dataset.iloc[:,8].values

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

classifier = SVC(kernel = "rbf", random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)

accuracy = ((cm[0,0] + cm[1,1]) / 118) * 100

filename = 'model_2.sav'
pickle.dump(classifier, open(filename, 'wb'))
