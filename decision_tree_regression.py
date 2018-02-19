# Decision Tree Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')
X = dataset[['Pclass','Age', 'Sex','SibSp','Parch','Fare']].values
Y = dataset.iloc[:, 1].values
                
testSet = pd.read_csv('test.csv')
X_ = testSet[['Pclass','Age', 'Sex','SibSp','Parch','Fare']].values
#Dealing with encoding the labels
from sklearn.preprocessing import LabelEncoder#, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,2] = labelencoder_X.fit_transform(X[:,2])

X_[:,2] = labelencoder_X.transform(X_[:,2])


#Dealing with missing datas
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer=Imputer(missing_values='NaN',
                strategy='mean', axis=0)
imputer=imputer.fit(X[:,1:2])
X[:,1:2]=imputer.transform(X[:,1:2])
X_[:,1:2]=imputer.transform(X_[:,1:2])

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, Y)



# Predicting a new result
Y_pred = regressor.predict(X_)

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
Y_pred = Y_pred.astype(int)
accuracy_score(Y, Y_pred)
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
print("fp: ", fp, " fn: ", fn)




# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y, test_size=0.2,
                                                  random_state=0)
'''
#Scaling the variables
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
'''

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, Y_train)



# Predicting a new result
Y_pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
Y_pred = Y_pred.astype(int)
accuracy_score(Y_test, Y_pred)
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred).ravel()
print("fp: ", fp, " fn: ", fn)
# Visualising the Decision Tree Regression results
#X_grid = np.arange(min(X), max(X), 0.01)
#X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(fp, fn, color = 'red')
plt.xlabel('False Positive')
plt.ylabel('False Negative')
#plt.axes([0, 0, 100, 100])

plt.show()


plt.close()
import matplotlib.pyplot as plt
plt.plot(17, 22, 'ro', label = 'Sam')
plt.plot(14, 24, 'go', label = 'Blake')
plt.plot(18, 11, 'bo', label = 'Rohit')
plt.plot(16, 23, 'co', label = 'Sierra')
plt.plot(12, 32, 'yo', label = 'Joel')
#plt.plot([15,16,9,26], [23,21,5,38], 'ro')
plt.xlabel('False Positive')
plt.ylabel('False Negative')
plt.axis([0, 25, 0, 35])
plt.legend()
plt.show()

'''
plt.scatter(fp, fn, color = 'red')
plt.plot(fp, color = 'blue')
plt.plot(fn, color = 'green')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
'''