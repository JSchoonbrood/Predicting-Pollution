import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
import math
import statistics
import os, sys, csv

from tensorflow import feature_column
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot
from pathlib import Path
from sys import platform

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)

def displayData(data, x, y):
    'Procedure to process and display data in graphs'
    pyplot.tight_layout()
    original_column = 0 # This index will be used to keep track of the column we will be comparing to every other column
    offset_column = 0 # This will be the index that will track every single column in the data to compare against the original_column
    columns = x.columns # columns is an array containing all the columns in the data

    for i in range(len(columns)):
        pyplot.figure(figsize=(15, 15)) # Enlarge size of figures for easier reading
        for i in range(len(columns)):
            pyplot.subplot(len(columns), len(columns), offset_column+1, autoscale_on=True) # Autoscale to scale axis for easier reading to determine correlation
            pyplot.scatter(x[columns[original_column]], x[columns[offset_column]])
            pyplot.title(str(columns[original_column]) + "x" + str(columns[offset_column]))
            offset_column += 1
        pyplot.show()
        original_column += 1
        offset_column = 0

def OHEncoder(dataframe):
    onehot_encoder = LabelBinarizer(sparse_output=False)
    labels = dataframe.pop('Targets')
    labels = onehot_encoder.fit_transform(labels)
    dataframe.update({'Targets': labels})
    print (labels)
    return dataframe

def visualiseModel(history):
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    #pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    #pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    return

def Run():
    # LOAD DATA
    current_dir = Path(os.path.dirname(__file__))
    if sys.platform == "linux" or sys.platform == "linux2":
        data_dir = os.path.join(current_dir.parent, 'RANKED_CSV/')
    elif sys.platform == "win32":
        data_dir =  os.path.join(current_dir.parent, 'RANKED_CSV\\')
    print (data_dir)

    training_files = os.listdir(data_dir)

    dataframes = []
    for fname in training_files:
        data = pd.read_csv(os.path.join(data_dir, fname))
        dataframes.append(data)
    dataframe = pd.concat(dataframes)

    #dataframe1 = dataframe.copy()
    #dataframe.pop('Targets')
    #dataframe1.pop('Rank')
    #dataframe1.pop('Targets')
    #displayData(dataframe, dataframe1, dataframe['Rank'])

    #dataset = OHEncoder(dataframe)

    # DATA SETUP
    train_df, test_df = train_test_split(dataframe, test_size=0.2)
    train_df, val_df =train_test_split(train_df, test_size=0.2)

    train_labels = (train_df.pop('Rank'))
    train_features = (train_df)

    #train_x = tf.data.Dataset.from_tensor_slices((dict(train_df), train_labels))

    print (train_features)
    print (train_labels)

    #val_labels = np.array(val_df.pop('Targets'))
    #val_features = np.array(val_df)

    #test_labels = np.array(test_df.pop('Targets'))
    #test_features = np.array(test_df)

    # NORMALISE DATA

    train_mean = np.mean(train_features)
    train_std = np.std(train_features)
    train_normalised = 0.5 * (np.tanh(0.01 * ((train_features - train_mean) / train_std)) + 1)

    #val_mean = np.mean(val_features)
    #val_std = np.std(val_features)
    #val_normalised = 0.5 * (np.tanh(0.01 * ((val_features - val_mean) / val_std)) + 1)

    #test_mean = np.mean(test_features)
    #test_std = np.std(test_features)
    #test_normalised = 0.5 * (np.tanh(0.01 * ((test_features - test_mean) / test_std)) + 1)

    # DEFINE MODEL

    model = tf.keras.Sequential()
    model.add(Dense(5, input_dim=5))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(64, activation='sigmoid'))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(8, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #loss = tf.keras.losses.CategoricalHinge()#from_logits=True)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #loss = "mean_squared_error"
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=20, batch_size=30)

    visualiseModel(history)

    #history = model.fit(train_ds, validation_data = val_ds, epochs=num_epoch)

Run()
