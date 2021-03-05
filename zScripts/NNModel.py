import numpy as np
import pandas as pd
import tensorflow as tf
import pathlib
import math
import statistics
import os, sys, csv

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot

def demo(feature_column, example_batch):
    feature_layer = layers.DenseFeatures(feature_column)
    print (feature_layer(example_batch).numpy())

def dfToDataset(dataframe, batch_size, shuffle=True):
    dataframe = dataframe.copy()
    labels = dataframe.pop('Targets')
    onehot_encoder = LabelBinarizer(sparse_output=False)
    labels = onehot_encoder.fit_transform(labels)
    #print (labels)
    #train_mean = np.mean(dataframe)
    #train_std = np.std(dataframe)
    #normalised_ds = 0.5 * (np.tanh(0.01 * ((dataframe - train_mean) / train_std)) + 1)

    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)

    return ds

def ArchitectureLog(model_summary, epochs, batch_size, normalization, loss, optimizer, accuracy, rank_num=8, rank_divisor=40, train_split=0.67, random_state=42):
    working_directory = os.getcwd() + r"\\SUMO\\"
    ArchitectureLog = working_directory + r"Log.csv"
    with open(ArchitectureLog, 'a', newline='') as log:
        writer = csv.writer(log)
        writer.writerow([model_summary, rank_num, rank_divisor, train_split, random_state, epochs, batch_size, normalization, loss, optimizer, accuracy])
    log.close()
    return

def visualiseModel(history):
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.legend()
    pyplot.show()
    return

def toInteger(ohe):
    integer = 1
    for i in ohe:
        if i == 1:
            break
        else:
            integer += 1
    return integer

def Run():
    working_directory = os.getcwd() + r"\\SUMO\\"
    data_directory = (working_directory + r"RANKED_CSV\\")
    training_files = os.listdir(data_directory)

    dataframes = []
    for file_name in training_files:
        data = pd.read_csv(data_directory + file_name)
        dataframes.append(data)
    dataframe = pd.concat(dataframes)

    print ("Number Of Rows:", len(dataframe))

    test_size = 0.2
    random_state = 42
    train, test = train_test_split(dataframe, test_size=test_size, random_state=random_state)
    train, val = train_test_split(train, test_size=test_size, random_state=random_state)

    print(len(train), 'Train Examples')
    print(len(val), 'Validation Examples')
    print(len(test), 'Test Examples')

    #example_batch = next(iter(train_ds))[0]

    feature_columns = []
    for header in ['Mean_Speed', 'Estimated_Travel_Time', 'Traffic_Level', 'Total_Neighbours', 'Length']:
        feature_columns.append(feature_column.numeric_column(header))

    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

    #demo(rank_one_hot, example_batch)

    batch_size = 64
    train_ds = dfToDataset(train, batch_size=batch_size)

    val_ds = dfToDataset(val, shuffle=False, batch_size=batch_size)
    test_ds = dfToDataset(test, shuffle=False, batch_size=batch_size)

    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128),# kernel_regularizer=tf.keras.regularizers.L2(0.03)),
        layers.BatchNormalization(),
        layers.Activation('sigmoid'),
        layers.Dense(64),
        layers.Activation('sigmoid'),
        layers.Dense(32),
        layers.Activation('sigmoid'),
        #layers.Dropout(0.1),
        layers.Dense(8),
        layers.Activation('softmax'),
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    #loss = tf.keras.losses.CategoricalHinge()#from_logits=True)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #loss = "mean_squared_error"
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])

    num_epoch = 20
    num_batch = 30

    history = model.fit(train_ds, val_ds, epochs=num_epoch)
    eval = model.evaluate(test_ds)


    #real_y = test_ds['targets']
    #predictions = []
    #for i in test_ds:
    #    print (i)
    #    predict = model.predict(i)
    #    prediction = toInteger(predict)
    #    predictions.append(prediction)

    #possible_labels = [[1., 0., 0., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0., 0., 0.], [0., 0., 0., 1., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 0., 0.], [0., 0., 0., 0., 0., 1., 0., 0.], [0., 0., 0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 0., 0., 1.]]
    confusion = tf.math.confusion_matrix(eval, num_classes=8)

    stringlist = []
    model.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    ArchitectureLog(model_summary=short_model_summary, epochs=num_epoch, batch_size=num_batch, normalization="tanh", loss=loss, optimizer=opt, accuracy=history.history['accuracy'][-1])
    visualiseModel(history)

    tf.keras.backend.clear_session()

Run()
