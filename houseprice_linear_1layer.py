# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 16:48:09 2020

@author: KARTH
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras


def house_model(y_new):
    xs = np.array([1,2,3,4,5,6,7],dtype=float)
    ys = np.array([1,1.5,2,2.5,3,3.5,4],dtype=float)
    model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
    model.compile(optimizer='sgd',loss='mean_squared_error')
    model.fit(xs,ys,epochs=100)
    return model.predict(y_new)[0]


prediction = house_model([9.0])
print(prediction)