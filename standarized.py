import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('test.csv',header=0)

cols = ['Age','Fare']
for col in cols:
	df[col] = (df[col]-df[col].mean())/df[col].std()

df.to_csv('standarizedTest.csv')