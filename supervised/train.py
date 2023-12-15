#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 14:11:20 2022

@author: soham
"""

from tensorflow import keras
from read import Read

divisions = 2
data_point_size = 500
extractions = 25
lag = int(data_point_size / divisions)
test = 8
reader = Read(data_point_size, extractions, 
              test, lag, divisions=divisions, verbose=False)
train_data, test_data, train_labels, test_labels = reader.getFormattedData()

# Image quality grade scale
class_names = ['A', 'B']

# Points per datapoint
N = int(data_point_size / divisions)

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(N, 1)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(len(class_names), activation='softmax')
    ])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
# make loss function continuous

model.fit(train_data, train_labels, epochs=8)

test_loss, test_acc = model.evaluate(test_data, test_labels)

print("Tested accuracy: ", test_acc)