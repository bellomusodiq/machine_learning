# __main__
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('./datasets/Artificial_Neural_Networks/Churn_Modelling.csv')
dataset.describe()
dataset.head()

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X_train[:, 2] = label_encoder.fit_transform(X_train[:, 2])
X_test[:, 2] = label_encoder.transform(X_test[:, 2])
one_hot_end = OneHotEncoder(drop='first')
X_train_ohe = one_hot_end.fit_transform(X_train[:, 1:2]).toarray()
X_test_ohe = one_hot_end.transform(X_test[:, 1:2]).toarray()

X_train = np.delete(X_train, 1, 1)
X_train = np.concatenate((X_train, X_train_ohe), axis=1)
X_train = np.array(X_train, dtype=float)
X_test = np.delete(X_test, 1, 1)
X_test = np.append(X_test, X_test_ohe, axis=1)
X_test = np.array(X_test, dtype=float)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers.core import Dense

#classifier = Sequential()
#classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
#classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
#classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
#classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#classifier.fit(X_train, y_train, batch_size=10, epochs=100)


new_ohe = one_hot_end.transform([['France']]).toarray()
gender_enc = label_encoder.transform(['Male']).reshape(1,1)
new_customer = np.array([[600, 40, 3, 60000, 2, 1, 1, 50000]])
new_customer = np.concatenate((new_customer,gender_enc, new_ohe), axis=1)


from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(verbose=1, monitor='loss', patience=5)

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs=30, verbose=1)
accuracies = cross_val_score(classifier, X_train, y_train, cv=10)

from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
classifier.add(Dropout(rate=.1))
classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
classifier.add(Dropout(rate=.1))
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=100, callbacks=[early_stop])

from sklearn.model_selection import GridSearchCV


def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform', input_shape=(11,)))
    classifier.add(Dropout(rate=.1))
    classifier.add(Dense(6, activation='relu', kernel_initializer='uniform'))
    classifier.add(Dropout(rate=.1))
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, verbose=1)

parameters = {
            'batch_size': [25,32],
            'epochs': [100,500],
            'optimizer': ['adam', 'rmsprop'],
        }

grid_search = GridSearchCV(classifier, param_grid=parameters, scoring='accuracy',cv=10, n_jobs=1)
grid_search = grid_search.fit(X_train, y_train, callbacks=[early_stop])
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_






