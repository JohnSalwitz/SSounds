import warnings
from unittest import TestCase

import numpy
import pandas
from keras.layers import Dense
from keras.models import Sequential
#from keras.utils import ku
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf


class PeakClassifier:
    seed = 7

    def __init__(self, sample_file, debug = False):
        # fix random seed for reproducibility
        numpy.random.seed(self.seed)

        # load dataset
        dataframe = pandas.read_csv(sample_file, header=None)
        dataframe = dataframe.values

        # 3rd row, first 80 columns
        # z = dataframe[2, 0:80].astype(float)
        # print(z)
        # all rows, first 80 columns (input)
        self.X = dataframe[:, 0:80].astype(float)
        # all rows, last column (output)
        self.Y = dataframe[:, 80]

        if 1:
            warnings.filterwarnings(action='ignore', category=DeprecationWarning)
            # encodes labels (outputs) values as integers
            self.encoder = LabelEncoder()
            self.encoder.fit(self.Y)
            encoded_Y = self.encoder.transform(self.Y)
            # convert integers to dummy variables (i.e. one hot encoded)
            self.dummy_y = tf.keras.utils.to_categorical(encoded_Y)

        self.estimator = KerasClassifier(build_fn=self.baseline_model, epochs=200, batch_size=5, verbose=0)

        self.debug = debug

    # define baseline model
    def baseline_model(self):
        # create model
        model = Sequential()
        model.add(Dense(8, input_dim=80, activation='relu'))

        # to do -- understand this line (jfs).  Looks like it is the number of outputs
        model.add(Dense(6, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        if self.debug:
            self.evaluate()

        return model

    def evaluate(self):
        kfold = KFold(n_splits=10, shuffle=True, random_state=self.seed)
        results = cross_val_score(self.estimator, self.X, self.dummy_y, cv=kfold)
        print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    def train(self):
        if self.debug:
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.dummy_y, test_size=1, random_state=self.seed)
            self.estimator.fit(X_train, Y_train)
            self.predict(X_test)
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.dummy_y)
            self.estimator.fit(X_train, Y_train)

    def predict(self, x_test):
        predictions = self.estimator.predict(x_test)
        if self.debug:
            print(x_test)
            print(predictions)
        prediction_name = self.encoder.inverse_transform(predictions)[0]
        print(prediction_name)
        return prediction_name


class TestPeak_classifier(TestCase):

    def setUp(self):
        self.sound_net = PeakClassifier("peaks_test_data.csv")
        self.sound_net.train()

    def test_evaluate(self):
        dataframe = pandas.read_csv("peaks_test_data.csv", header=None)
        dataset = dataframe.values
        z = dataset[12:13, 0:80].astype(float)
        self.sound_net.predict(z)
