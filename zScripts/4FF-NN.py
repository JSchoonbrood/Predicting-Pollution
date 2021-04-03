import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys

from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
        if fname == ".gitignore":
            pass
        else:
            data = pd.read_csv(os.path.join(data_dir, fname))
            dataframes.append(data)
    dataframe = pd.concat(dataframes)

    # DATA SETUP
    train_df, test_df = train_test_split(dataframe, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    train1_features = train_df.copy()
    train1_features.pop('Rank')
    train1_labels = train1_features.pop('Rank2')
    # train1_features.pop('Length')
    train1_features.pop('Length')

    # train_df.pop('Estimated_Travel_Time')
    # train_df.pop('Total_Neighbours')
    # train_df.pop('Length')
    train_features = train_df[train_df.Rank2 != 2]
    train_features.pop('Rank2')
    train_features.pop('Total_Neighbours')
    train_labels = train_features.pop('Rank')

    #########################################################

    val1_features = val_df.copy()
    val1_features.pop('Rank')
    val1_labels = val1_features.pop('Rank2')
    # val1_features.pop('Length')
    val1_features.pop('Length')

    # val_df.pop('Estimated_Travel_Time')
    # val_df.pop('Total_Neighbours')
    # val_df.pop('Length')
    val_features = val_df[val_df.Rank2 != 2]
    val_features.pop('Rank2')
    val_features.pop('Total_Neighbours')
    val_labels = val_features.pop('Rank')

    #########################################################

    test1_features = test_df.copy()
    test1_features.pop('Rank')
    test1_labels = test1_features.pop('Rank2')
    # test1_features.pop('Length')
    test1_features.pop('Length')

    # test_df.pop('Estimated_Travel_Time')
    # test_df.pop('Total_Neighbours')
    # test_df.pop('Length')
    test_features = test_df[test_df.Rank2 != 2]
    test_features.pop('Rank2')
    test_features.pop('Total_Neighbours')
    test_labels = test_features.pop('Rank')

    print(train_features)

    # MODEL (RANKS 1 OR 2(2 IS 8))

    model1 = tf.keras.Sequential()
    model1.add(Dense(5, input_dim=4))
    model1.add(BatchNormalization())

    model1.add(Dense(10,  activation='relu'))
    model1.add(Dropout(0.25))

    model1.add(Dense(25,  activation='relu'))
    model1.add(Dropout(0.25))

    model1.add(Flatten())
    model1.add(Dense(3, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # loss = tf.keras.losses.CategoricalHinge()#from_logits=True)
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # loss = "mean_squared_error"
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model1.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    history = model1.fit(train1_features, train1_labels, epochs=25,
                         batch_size=80)
    eval = model1.evaluate(val1_features, val1_labels)

    '''y_actual = []
    y_pred = []
    for i in range(5000):
        test_data = test1_features.iloc[i].values.tolist()
        test_data = np.reshape(test_data, (4, 1)).T
        test_targets = test1_labels.iloc[i]

        prediction = model1.predict(test_data)

        classes = np.argmax(prediction, axis=1)

        y_actual.append(test_targets)
        y_pred.append(classes[0])

        # print ("Prediction:", classes, " Actual:", test_targets)

    conf_matrix_file_html = os.path.join(current_dir.parent,
                                         'Rank8ConfMatr.html')
    conf_matrix = pd.crosstab(pd.Series(y_actual), pd.Series(y_pred),
                              rownames=['Actual'], colnames=['Predicted'],
                              margins=True)
    conf_matrix.to_html(conf_matrix_file_html)'''

    # MODEL (RANKS 1-7)

    model2 = tf.keras.Sequential()
    model2.add(Dense(5, input_dim=4))
    model2.add(BatchNormalization())

    model2.add(Dense(10,  activation='relu'))
    #model2.add(BatchNormalization())
    #model2.add(Dropout(0.25))

    model2.add(Dense(20,  activation='relu'))
    #model2.add(BatchNormalization())
    #model2.add(Dropout(0.25))

    model2.add(Dense(15,  activation='relu'))
    #model2.add(BatchNormalization())
    #model2.add(Dropout(0.25))

    model2.add(Flatten())
    model2.add(Dense(9, activation='softmax'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    # loss = tf.keras.losses.CategoricalHinge()#from_logits=True)
    # loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # loss = "mean_squared_error"
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model2.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    history = model2.fit(train_features, train_labels, epochs=60, batch_size=120)
    eval = model2.evaluate(val_features, val_labels)

    model2.save(os.path.join(current_dir, 'Model2-5'))

    # model.save('PollutionPrediction.model')

    # visualiseModel(history, eval)

    # test_df.pop('Total_Neighbours')

    y_actual = []
    y_pred = []
    for i in range(5000):
        test_data = test_features.iloc[i].values.tolist()
        test_data = np.reshape(test_data, (4, 1)).T
        test_targets = test_labels.iloc[i]

        prediction = model2.predict(test_data)

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
