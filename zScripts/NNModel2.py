import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys

from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from matplotlib import pyplot
from pathlib import Path

import logging
logging.getLogger("tensorflow").setLevel(logging.WARNING)


def visualiseModel(history, eval):
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    # pyplot.plot(eval.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    # pyplot.plot(eval.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    return


def Run():
    # LOAD DATA
    current_dir = Path(os.path.dirname(__file__))
    if sys.platform == "linux" or sys.platform == "linux2":
        data_dir = os.path.join(current_dir.parent, 'RANKED_CSV/')
    elif sys.platform == "win32":
        data_dir = os.path.join(current_dir.parent, 'RANKED_CSV\\')

    training_files = os.listdir(data_dir)

    dataframes = []
    for fname in training_files:
        data = pd.read_csv(os.path.join(data_dir, fname))
        dataframes.append(data)
    dataframe = pd.concat(dataframes)

    # DATA SETUP
    train_df, test_df = train_test_split(dataframe, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    train_labels = (train_df.pop('Rank'))
    train_features = (train_df)

    val_labels = np.array(val_df.pop('Rank'))
    val_features = np.array(val_df)

    test_labels = np.array(test_df.pop('Rank'))
    test_features = np.array(test_df)

    # DEFINE MODEL

    model = tf.keras.Sequential()
    model.add(Dense(5, input_dim=5))
    model.add(BatchNormalization())

    model.add(Dense(512,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(512,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(512,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(512,  activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(9, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # loss = tf.keras.losses.CategoricalHinge()#from_logits=True)
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # loss = "mean_squared_error"
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    history = model.fit(train_features, train_labels, epochs=30, batch_size=30)
    eval = model.evaluate(val_features, val_labels)

    # model.save('PollutionPrediction.model')

    visualiseModel(history, eval)

    test_labels = test_df.pop('Rank')
    # test_df.pop('Total_Neighbours')
    test_features = test_df

    y_actual = []
    y_pred = []
    for i in range(5000):
        test_data = test_features.iloc[i].values.tolist()
        test_data = np.reshape(test_data, (5, 1)).T
        test_targets = test_labels.iloc[i]

        prediction = model.predict(test_data)

        classes = np.argmax(prediction, axis=1)

        y_actual.append(test_targets)
        y_pred.append(classes[0])

        # print ("Prediction:", classes, " Actual:", test_targets)

    conf_matrix_file_html = os.path.join(current_dir.parent,
                                         'ConfusionMatrix.html')
    conf_matrix = pd.crosstab(pd.Series(y_actual), pd.Series(y_pred),
                              rownames=['Actual'], colnames=['Predicted'],
                              margins=True)
    conf_matrix.to_html(conf_matrix_file_html)

    # history = model.fit(train_ds, validation_data = val_ds, epochs=num_epoch)


Run()
