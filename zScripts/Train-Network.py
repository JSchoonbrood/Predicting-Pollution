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

    dataframe.pop('Traffic_Level')
    train_df, test_df = train_test_split(dataframe, test_size=0.2)
    train_df, val_df = train_test_split(train_df, test_size=0.2)

    #trainer
    train_features1 = train_df.copy()
    train_features1.pop('OverallRank')

    train_features2 = train_features1[train_features1.Rank1Identifier != 1]
    train_features2.pop('Rank1Identifier')
    train_features4 = train_features2[train_features2.Rank2Identifier != 1]
    train_features4.pop('Rank2Identifier')

    train_labels1 = train_features1.pop('Rank1Identifier')
    train_features1.pop('Rank2Identifier')
    train_features1.pop('Rank4Identifier')

    train_labels2 = train_features2.pop('Rank2Identifier')
    train_features2.pop('Rank4Identifier')

    train_labels4 = train_features4.pop('Rank4Identifier')

    val_features1 = val_df.copy()
    val_features1.pop('OverallRank')

    val_features2 = val_features1[val_features1.Rank1Identifier != 1]
    val_features2.pop('Rank1Identifier')
    val_features4 = val_features2[val_features2.Rank2Identifier != 1]
    val_features4.pop('Rank2Identifier')

    val_labels1 = val_features1.pop('Rank1Identifier')
    val_features1.pop('Rank2Identifier')
    val_features1.pop('Rank4Identifier')

    val_labels2 = val_features2.pop('Rank2Identifier')
    val_features2.pop('Rank4Identifier')

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

    test_features = test_df.copy()
    test_features.pop('Rank1Identifier')
    test_features.pop('Rank2Identifier')
    test_features.pop('Rank4Identifier')
    test_labels = test_features.pop('OverallRank')

    # make a prediction
    y_actual = []
    y_pred = []
    score = 0
    for i in range(5000):
        test_data = test_features.iloc[i].values.tolist()
        test_data = np.reshape(test_data, (4, 1)).T
        test_targets = test_labels.iloc[i]

        y_actual.append(test_targets)
        rank1pred = rank1.predict(test_data)
        prediction = np.argmax(rank1pred, axis=1)

        if prediction[0] == 1:
            y_pred.append(1)
            if 1 == test_targets:
                score += 1
        else:
            rank2pred = rank2.predict(test_data)
            prediction1 = np.argmax(rank2pred, axis=1)
            if prediction1[0] == 1:
                y_pred.append(2)
                if 2 == test_targets:
                    score += 1
            else:
                rank4pred = rank4.predict(test_data)
                prediction2 = np.argmax(rank4pred, axis=1)
                if prediction2[0] == 1:
                    y_pred.append(4)
                    if 4 == test_targets:
                        score += 1
                else:
                    y_pred.append(3)
                    if 3 == test_targets:
                        score += 1

    print ("Accuracy ->", (score/5000)*100)


    conf_matrix_file_html = os.path.join(current_dir.parent,
                                     'ConfusionMatrix.html')
    conf_matrix = pd.crosstab(pd.Series(y_actual), pd.Series(y_pred),
                          rownames=['Actual'], colnames=['Predicted'],
                          margins=True)
    conf_matrix.to_html(conf_matrix_file_html)


def model(train_x, train_y, test_x, test_y, output_neurons, output_file_name):
    model = tf.keras.Sequential()
    model.add(Dense(35, input_dim=(train_x.shape[1])))
    model.add(BatchNormalization())
    model.add(Dense(25))
    model.add(Dense(15, activation='sigmoid'))
    model.add(Dense(output_neurons, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

    early_stop = keras.callbacks.EarlyStopping(monitor='accuracy', min_delta=0, patience=50, verbose=0, mode='max', baseline=None)
    checkpoint = ModelCheckpoint(output_file_name, monitor='accuracy', mode='max', save_best_only=True)

    history = model.fit(train_x, train_y, epochs=10000, batch_size=75, validation_data=(test_x, test_y), verbose=2, shuffle=False, callbacks=[LearningRateReducerCb(), early_stop, checkpoint])
    return history

run()
