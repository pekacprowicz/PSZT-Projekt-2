# IPython log file

import pandas as pd
data = pd.read_csv("car.data", dtype="category")
buying_series = data.iloc[:,0]
maint_series = data.iloc[:,1]
doors_series = data.iloc[:,2]
persons_series = data.iloc[:,3]
lug_boot_series = data.iloc[:,4]
safety_series = data.iloc[:,5]
deal_series = data.iloc[:,6]
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
buying_encoded = le.fit_transform(buying_series)
maint_encoded = le.fit_transform(maint_series)
doors_encoded = le.fit_transform(doors_series)
persons_encoded = le.fit_transform(persons_series)
lug_boot_encoded = le.fit_transform(lug_boot_series)
safety_encoded = le.fit_transform(safety_series)
deal_encoded = le.fit_transform(deal_series)
y = deal_encoded
X = list(zip(buying_encoded, maint_encoded, doors_encoded, persons_encoded, lug_boot_encoded, safety_encoded))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from sklearn.naive_bayes import CategoricalNB
cnb = CategoricalNB()
y_pred = cnb.fit(X_train, y_train).predict(X_test)
print(f"Procent poprawnych predykcji modelu: {(y_test == y_pred).sum()/len(X_test)*100}%")
from sklearn.model_selection import cross_val_score
import random
data_combined = list(zip(X,y))
random.shuffle(data_combined)
X,y = zip(*data_combined)
scores = cross_val_score(cnb, X, y, cv=5)
scores
#[Out]# array([0.85260116, 0.85260116, 0.85260116, 0.83478261, 0.86666667])
