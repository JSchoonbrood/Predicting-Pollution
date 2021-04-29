import numpy as np
import pandas as pd
import tensorflow as tf
import os
import sys
import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import BatchNormalization, Flatten, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot
from pathlib import Path
from keras.models import load_model
from numpy import concatenate

def showCorrelations(dataframe):
    df_corr = dataframe.corr()
    f = pyplot.figure(figsize=(19, 15))
    pyplot.matshow(df_corr, fignum=f.number)
    pyplot.xticks(range(dataframe.select_dtypes(['number']).shape[1]), dataframe.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    pyplot.yticks(range(dataframe.select_dtypes(['number']).shape[1]), dataframe.select_dtypes(['number']).columns, fontsize=14)
    cb = pyplot.colorbar()
    cb.ax.tick_params(labelsize=14)
    pyplot.title('Correlation Matrix', fontsize=16);
    pyplot.show()

def visualiseModel(history):
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(eval.history['val_loss'], label='test')
    pyplot.legend()
    # plot accuracy during training
    #pyplot.subplot(212)
    #pyplot.title('Accuracy')
    #pyplot.plot(history.history['accuracy'], label='train')
    #pyplot.plot(eval.history['val_accuracy'], label='test')
    #pyplot.legend()
    pyplot.show()
    return

class LearningRateReducerCb(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    old_lr = self.model.optimizer.lr.read_value()
    new_lr = old_lr * 0.95
    print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
    self.model.optimizer.lr.assign(new_lr)

def run():
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

    #showCorrelations(dataframe)
    values = dataframe.values
    columns = dataframe.columns

    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaled = scaler.fit_transform(values)

    #df_normalized = pd.DataFrame(scaled, columns=columns)

    train_df, test_df = train_test_split(dataframe, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    train_features = train_df[train_df.Rank2 != 2]
    train_features.pop('Rank2')
    train_features.pop('Traffic_Level')
    train_labels = train_features.pop('Rank')

    val_features = val_df[val_df.Rank2 != 2]
    val_features.pop('Rank2')
    val_features.pop('Traffic_Level')
    val_labels = val_features.pop('Rank')

    test_features = test_df[test_df.Rank2 != 2]
    test_features.pop('Rank2')
    test_features.pop('Traffic_Level')
    test_labels = test_features.pop('Rank')

    history = model(train_features, train_labels, val_features, val_labels)

    #visualiseModel(history)
    saved_model = load_model('/home/jake/github/Predicting-Pollution/model.h5')

    # make a prediction
    y_actual = []
    y_pred = []
    score = 0
    for i in range(5000):
        test_data = test_features.iloc[i].values.tolist()
        test_data = np.reshape(test_data, (4, 1)).T
        test_targets = test_labels.iloc[i]

        prediction = saved_model.predict(test_data)

        classes = np.argmax(prediction, axis=1)

        y_actual.append(test_targets)
        y_pred.append(classes[0])

        if classes[0] == test_targets:
            score += 1

    print ("Accuracy ->", (score/5000)*100)


    conf_matrix_file_html = os.path.join(current_dir.parent,
                                     'ConfusionMatrix.html')
    conf_matrix = pd.crosstab(pd.Series(y_actual), pd.Series(y_pred),
                          rownames=['Actual'], colnames=['Predicted'],
                          margins=True)
    conf_matrix.to_html(conf_matrix_file_html)


def model(train_x, train_y, test_x, test_y):
    model = tf.keras.Sequential()
    model.add(Dense(30, input_dim=(train_x.shape[1])))
    model.add(BatchNormalization())
    model.add(Dense(20))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=50, verbose=0, mode='max', baseline=None)
    checkpoint = ModelCheckpoint('/home/jake/github/Predicting-Pollution/model.h5', monitor='accuracy', mode='max', save_best_only=True)

    history = model.fit(train_x, train_y, epochs=10000, batch_size=75, validation_data=(test_x, test_y), verbose=2, shuffle=False, callbacks=[LearningRateReducerCb(), early_stop, checkpoint])
    return history

run()
