# T2/T3 hiring interview test

# develop classification model for penguins

#STEP1: import libs and data

import pandas as pd
penguins = pd.read_csv("https://github.com/allisonhorst/palmerpenguins/raw/5b5891f01b52ae26ad8cb9755ec93672f49328a8/data/penguins_size.csv")

import numpy as np
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from numpy import array
from numpy import argmax

#STEP2: view data

>>> penguins
    species_short     island  culmen_length_mm  culmen_depth_mm  flipper_length_mm  body_mass_g     sex
0          Adelie  Torgersen              39.1             18.7              181.0       3750.0    MALE
1          Adelie  Torgersen              39.5             17.4              186.0       3800.0  FEMALE
2          Adelie  Torgersen              40.3             18.0              195.0       3250.0  FEMALE
3          Adelie  Torgersen               NaN              NaN                NaN          NaN     NaN
4          Adelie  Torgersen              36.7             19.3              193.0       3450.0  FEMALE
..            ...        ...               ...              ...                ...          ...     ...
339        Gentoo     Biscoe               NaN              NaN                NaN          NaN     NaN
340        Gentoo     Biscoe              46.8             14.3              215.0       4850.0  FEMALE
341        Gentoo     Biscoe              50.4             15.7              222.0       5750.0    MALE
342        Gentoo     Biscoe              45.2             14.8              212.0       5200.0  FEMALE
343        Gentoo     Biscoe              49.9             16.1              213.0       5400.0    MALE

[344 rows x 7 columns]

>>> penguins.shape
(344, 7)

>>> penguins.dtypes
species_short         object
island                object
culmen_length_mm     float64
culmen_depth_mm      float64
flipper_length_mm    float64
body_mass_g          float64
sex                   object
dtype: object

#don't need to do, convert object type to string
#penguins["species_short"] = penguins["species_short"].astype("|S")


#check distriution of values

>>> penguins['species_short'].nunique()
3

>>> penguins['species_short'].value_counts()
b'Adelie'       152
b'Gentoo'       124
b'Chinstrap'     68
Name: species_short, dtype: int64

==> there is good representation of the classes, no imbalance 

>>> penguins['island'].value_counts()
Biscoe       168
Dream        124
Torgersen     52
Name: island, dtype: int64

>>> penguins['sex'].value_counts()
Biscoe       168
Dream        124
Torgersen     52
Name: island, dtype: int64

>>> penguins['sex'].value_counts()
MALE      168
FEMALE    165
.           1
Name: sex, dtype: int64

#remove row 336 that has . in "sex" column
penguins_remove_junk = penguins.drop(penguins.index[336]) 


# count number of NaN in each column

>>> penguins_remove_junk['species_short'].isnull().sum()
0

>>> penguins_remove_junk['culmen_length_mm'].isnull().sum()
2

>>> penguins_remove_junk['culmen_depth_mm'].isnull().sum()
2

>>> penguins_remove_junk['flipper_length_mm'].isnull().sum()
2

>>> penguins_remove_junk['body_mass_g'].isnull().sum()
2

>>> penguins_remove_junk['sex'].isnull().sum()
10

# don't need to do if using XGboost, but anyways dropping rows with NaN

penguins_cleaned = penguins_remove_junk.dropna() 

>>>penguins_cleaned 
    species_short     island  culmen_length_mm  culmen_depth_mm  flipper_length_mm  body_mass_g     sex
0       b'Adelie'  Torgersen              39.1             18.7              181.0       3750.0    MALE
1       b'Adelie'  Torgersen              39.5             17.4              186.0       3800.0  FEMALE
2       b'Adelie'  Torgersen              40.3             18.0              195.0       3250.0  FEMALE
4       b'Adelie'  Torgersen              36.7             19.3              193.0       3450.0  FEMALE
5       b'Adelie'  Torgersen              39.3             20.6              190.0       3650.0    MALE
..            ...        ...               ...              ...                ...          ...     ...
338     b'Gentoo'     Biscoe              47.2             13.7              214.0       4925.0  FEMALE
340     b'Gentoo'     Biscoe              46.8             14.3              215.0       4850.0  FEMALE
341     b'Gentoo'     Biscoe              50.4             15.7              222.0       5750.0    MALE
342     b'Gentoo'     Biscoe              45.2             14.8              212.0       5200.0  FEMALE
343     b'Gentoo'     Biscoe              49.9             16.1              213.0       5400.0    MALE

[333 rows x 7 columns]



#STEP3: XGboost cannot only accept numerical values, so need to convert categorical to numerical one hot encoding 

dataset = penguins_cleaned.values
# split data into X and y
#split out categorical features
X = dataset[:,[1,6]]

>>> X[-10:]
array([['Biscoe', 'FEMALE'],
       ['Biscoe', 'MALE'],
       ['Biscoe', 'FEMALE'],
       ['Biscoe', 'MALE'],
       ['Biscoe', 'MALE'],
       ['Biscoe', 'FEMALE'],
       ['Biscoe', 'FEMALE'],
       ['Biscoe', 'MALE'],
       ['Biscoe', 'FEMALE'],
       ['Biscoe', 'MALE']], dtype='<U9')

X_numerical = dataset[:,[2,3,4,5]] 

>>> X_numerical[-10:]
array([[43.5, 15.2, 213.0, 4650.0],
       [51.5, 16.3, 230.0, 5500.0],
       [46.2, 14.1, 217.0, 4375.0],
       [55.1, 16.0, 230.0, 5850.0],
       [48.8, 16.2, 222.0, 6000.0],
       [47.2, 13.7, 214.0, 4925.0],
       [46.8, 14.3, 215.0, 4850.0],
       [50.4, 15.7, 222.0, 5750.0],
       [45.2, 14.8, 212.0, 5200.0],
       [49.9, 16.1, 213.0, 5400.0]], dtype=object)

#only convert the categorical columns to string
X = X.astype(str)

Y = dataset[:,0]
# encode string input values as integers
encoded_x = None
for i in range(0, X.shape[1]):
        label_encoder = LabelEncoder()
        feature = label_encoder.fit_transform(X[:,i])
        feature = feature.reshape(X.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
        feature = onehot_encoder.fit_transform(feature)
        if encoded_x is None:
                encoded_x = feature
        else:
                encoded_x = numpy.concatenate((encoded_x, feature), axis=1)


>>> encoded_x
array([[0., 0., 1., 0., 1.],
       [0., 0., 1., 1., 0.],
       [0., 0., 1., 1., 0.],
       ...,
       [1., 0., 0., 0., 1.],
       [1., 0., 0., 1., 0.],
       [1., 0., 0., 0., 1.]])

>>> encoded_x.shape
(333, 5)

#concatenate into one feature numpy array
>>> X_total = numpy.hstack((encoded_x,X_numerical))
>>> X_total.shape
(333, 9)
>>> X_total[-10:]
array([[1.0, 0.0, 0.0, 1.0, 0.0, 43.5, 15.2, 213.0, 4650.0],
       [1.0, 0.0, 0.0, 0.0, 1.0, 51.5, 16.3, 230.0, 5500.0],
       [1.0, 0.0, 0.0, 1.0, 0.0, 46.2, 14.1, 217.0, 4375.0],
       [1.0, 0.0, 0.0, 0.0, 1.0, 55.1, 16.0, 230.0, 5850.0],
       [1.0, 0.0, 0.0, 0.0, 1.0, 48.8, 16.2, 222.0, 6000.0],
       [1.0, 0.0, 0.0, 1.0, 0.0, 47.2, 13.7, 214.0, 4925.0],
       [1.0, 0.0, 0.0, 1.0, 0.0, 46.8, 14.3, 215.0, 4850.0],
       [1.0, 0.0, 0.0, 0.0, 1.0, 50.4, 15.7, 222.0, 5750.0],
       [1.0, 0.0, 0.0, 1.0, 0.0, 45.2, 14.8, 212.0, 5200.0],
       [1.0, 0.0, 0.0, 0.0, 1.0, 49.9, 16.1, 213.0, 5400.0]], dtype=object)

# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)


# STEP4: split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X_total, label_encoded_y, test_size=test_size, random_state=seed)


# STEP5: fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


==> WORKS, Accuracy: 99.09%

# STEP5: Plot Feature Importance

from matplotlib import pyplot

print(model.feature_importances_)

# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

# plot feature importance
from xgboost import plot_importance

plot_importance(model)
pyplot.show()





