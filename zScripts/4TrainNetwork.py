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

def model(train_x, train_y, test_x, test_y, output_neurons, output_file_name):
    model = tf.keras.Sequential()
    model.add(Dense(64, input_dim=(train_x.shape[1]), activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='sigmoid'))
    model.add(Dense(16, activation='sigmoid'))
    model.add(Dense(output_neurons, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=15, verbose=0, mode='max', baseline=None)
    checkpoint = ModelCheckpoint(output_file_name, monitor='accuracy', mode='max', save_best_only=True)

    history = model.fit(train_x, train_y, epochs=10000, batch_size=75, validation_data=(test_x, test_y), verbose=2, shuffle=False, callbacks=[LearningRateReducerCb(), early_stop, checkpoint])
    return history

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

    dataframe.pop('Traffic_Level')
    train_df, test_df = train_test_split(dataframe, test_size=0.2, shuffle=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=True)

    train_features = train_df.copy()
    train_labels = train_features.pop('OverallRank')

<<<<<<< HEAD:zScripts/4TrainNetwork.py
    val_features = val_df.copy()
    val_labels = val_features.pop('OverallRank')
=======
    val_labels4 = val_features4.pop('Rank4Identifier')

    rank1_identifier = model(train_features1, train_labels1, val_features1, val_labels1, 3, os.path.join(current_dir, 'rank1identifier.h5'))
    rank2_identifier = model(train_features2, train_labels2, val_features2, val_labels2, 3, os.path.join(current_dir, 'rank2identifier.h5'))
    #rank3_identifier = model(train_features, trian_labels3, val_features, val_labels3, 3, os.path.join(current_dir, 'rank3identifier.h5'))
    rank4_identifier = model(train_features4, train_labels4, val_features4, val_labels4, 3, os.path.join(current_dir, 'rank4identifier.h5'))

    #visualiseModel(history)
    rank1 = load_model(os.path.join(current_dir, 'rank1identifier.h5'))
    rank2 = load_model(os.path.join(current_dir, 'rank2identifier.h5'))
    #rank3 = load_model(os.path.join(current_dir, 'rank3identifier.h5'))
    rank4 = load_model(os.path.join(current_dir, 'rank4identifier.h5'))
>>>>>>> 291779c33a4bdd8f3575449c893a30b9a27346b5:zScripts/Train-Network.py

    test_features = test_df.copy()
    test_labels = test_features.pop('OverallRank')

    train_model = model(train_features, train_labels, val_features, val_labels, 13, os.path.join(current_dir, 'EmissionsModel.h5'))
    pred_model = load_model(os.path.join(current_dir, 'EmissionsModel.h5'))

    # make a prediction
    y_actual = []
    y_pred = []
    score = 0
    for i in range(5000):
        test_data = test_features.iloc[i].values.tolist()
        test_data = np.reshape(test_data, (4, 1)).T
        test_targets = test_labels.iloc[i]

        y_actual.append(test_targets)
        rankpred = pred_model.predict(test_data)
        prediction = np.argmax(rankpred, axis=1)
        y_pred.append(prediction[0])

    conf_matrix_file_html = os.path.join(current_dir.parent,
                                     'ConfusionMatrix.html')
    conf_matrix = pd.crosstab(pd.Series(y_actual), pd.Series(y_pred),
                          rownames=['Actual'], colnames=['Predicted'])
    conf_matrix.to_html(conf_matrix_file_html)

run()
