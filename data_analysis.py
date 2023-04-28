#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from random import shuffle
import numpy as np
import pandas as pd 
import math
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import save_model, load_model
import tensorflow as tf
from keras.optimizers import Adam


# In[6]:


df = pd.read_csv('data/data.csv')
df = df.drop(['remove'], axis=1)

df = df.drop(['ball_vel_y', 'ball_vel_angular'], axis=1)

def remove_brackets(value):
    if isinstance(value, float) and math.isnan(value):
        return value
    else:
        return float(value.strip('[]'))

cols_to_process = ['velocity1_x', 'velocity1_y', 'velocity2_x', 'velocity2_y', 'velocity3_x', 'velocity3_y']

for col in cols_to_process:
    df[col] = df[col].apply(remove_brackets)
print(df.shape)

df.dropna(axis=1, inplace=True)


# In[7]:


# process categorical data (one-hot encoding)

scaler = StandardScaler()
scaler.fit(df.iloc[:,:-6])
df.iloc[:,:-6] = scaler.transform(df.iloc[:,:-6])

X = df.iloc[:,:-6].values
y = df.iloc[:,-6:].values

zipped = list(zip(X, y))
shuffle(zipped)
X, y = zip(*zipped)
X = np.array(X)
y = np.array(y)

# split data into train, validation, and test sets

train_size = 0.7
val_size = 0.15
test_size = 0.15

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(train_size), random_state=42)



# In[10]:


def createModel(neurons=10, dropout_rate=0.0, optimizer='adam'):
    '''
    Returns a model with the given parameters
    '''
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(neurons, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(6, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['mae'])
    return model
 


# In[ ]:


# Create model skeleton
model = KerasRegressor(build_fn=createModel, verbose=0)

# Define the grid search parameters
param_grid = {'neurons': [16, 32, 64, 128, 256, 512],
              'dropout_rate': [0.1, 0.2, 0.3],
              'optimizer': ['adam', 'rmsprop', 'sgd']}

# Create random search object for hyperparameter tuning
random_search = RandomizedSearchCV(estimator=model,
                                   param_distributions=param_grid,
                                   n_iter=40,
                                   cv=3,
                                   scoring='neg_mean_absolute_error',
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)
# Fit the random search model
random_search.fit(X_train, y_train, validation_data=(X_val, y_val))

print(random_search.best_params_)
print(random_search.best_score_)

# Save the best model
best_model = random_search.best_estimator_.model


# In[12]:


best_model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001), metrics=['mae'])


# In[ ]:


best_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val), verbose=2)
save_model(best_model, 'model.h5')


# In[9]:
mean = scaler.mean_
std = scaler.scale_

print("Mean:", mean)
print("Std:", std)
