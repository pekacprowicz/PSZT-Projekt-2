# IPython log file

import pandas as pd
data = pd.read_csv("agaricus-lepiota.data", dtype="category")
habitat_series = data.iloc[:,0]
cap_shape_series = data.iloc[:,1]
cap_surface_series = data.iloc[:,2]
cap_color_series = data.iloc[:,3]
bruises_series = data.iloc[:,4]
odor_series = data.iloc[:,5]
gill_attachment_series = data.iloc[:,6]
gill_spacing_series = data.iloc[:,7]
gill_size_series = data.iloc[:,8]
gill_color_series = data.iloc[:,9]
stalk_shape_series = data.iloc[:,10]
stalk_root_series = data.iloc[:,11]
stalk_surface_above_ring_series = data.iloc[:,12]
stalk_surface_below_ring_series = data.iloc[:,13]
stalk_color_above_ring_series = data.iloc[:,14]
stalk_color_below_ring_series = data.iloc[:,15]
veil_type_series = data.iloc[:,16]
veil_color_series = data.iloc[:,17]
ring_number_series = data.iloc[:,18]
ring_type_series = data.iloc[:,19]
spore_prnt_color_series = data.iloc[:,20]
population_series = data.iloc[:,21]
class_series = data.iloc[:,22]

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
habitat_encoded = le.fit_transform(habitat_series)
cap_shape_encoded = le.fit_transform(cap_shape_series)
cap_surface_encoded = le.fit_transform(cap_surface_series)
cap_color_encoded = le.fit_transform(cap_color_series)
bruises_encoded = le.fit_transform(bruises_series)
odor_encoded = le.fit_transform(odor_series)
gill_attachment_encoded = le.fit_transform(gill_attachment_series)
gill_spacing_encoded = le.fit_transform(gill_spacing_series)
gill_size_encoded = le.fit_transform(gill_size_series)
gill_color_encoded = le.fit_transform(gill_color_series)
stalk_shape_encoded = le.fit_transform(stalk_shape_series)
stalk_root_encoded = le.fit_transform(stalk_root_series)
stalk_surface_above_ring_encoded = le.fit_transform(stalk_surface_above_ring_series)
stalk_surface_below_ring_encoded = le.fit_transform(stalk_surface_below_ring_series)
stalk_color_above_ring_encoded = le.fit_transform(stalk_color_above_ring_series)
stalk_color_below_ring_encoded = le.fit_transform(stalk_color_below_ring_series)
veil_type_encoded = le.fit_transform(veil_type_series)
veil_color_encoded = le.fit_transform(veil_color_series)
ring_number_encoded = le.fit_transform(ring_number_series)
ring_type_encoded = le.fit_transform(ring_type_series)
spore_prnt_color_encoded = le.fit_transform(spore_prnt_color_series)
population_encoded = le.fit_transform(population_series)
class_encoded = le.fit_transform(class_series)



y = class_encoded
X = list(zip(habitat_encoded,cap_shape_encoded,cap_surface_encoded,cap_color_encoded,bruises_encoded,odor_encoded,gill_attachment_encoded,gill_spacing_encoded,gill_size_encoded,gill_color_encoded,stalk_shape_encoded,stalk_root_encoded,stalk_surface_above_ring_encoded,stalk_surface_below_ring_encoded,stalk_color_above_ring_encoded,stalk_color_below_ring_encoded,veil_type_encoded,veil_color_encoded,ring_number_encoded,ring_type_encoded,spore_prnt_color_encoded,population_encoded))
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
