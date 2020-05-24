print('Демонстрация обучения модели на маленьком фрагменте датасета')

import numpy as np
import keras
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, GaussianDropout, LSTM, Bidirectional, BatchNormalization, MaxPooling1D, MaxPooling2D, MaxPooling3D, AveragePooling1D
import keras_metrics as km
import datetime
from sklearn.model_selection import train_test_split

from edf_preprocessor import EDF_Preprocessor
from ind_rnn import IndRNN
from ind_rnn import IndRNNCell, RNN


# Не работает под Windows. Под Docker пока не проверял
# import subprocess
# subprocess.call(["rm", "/rf", "logs/*"])

# BEGIN MODEL DESCRIPTION

ip = Input(shape=(256,23))
x = IndRNN(512, return_sequences=True)(ip)
x = BatchNormalization()(x)
x = AveragePooling1D()(x)
x = IndRNN(512, return_sequences=True)(x)
x = BatchNormalization()(x)
x = AveragePooling1D()(x)
x = IndRNN(512, return_sequences=True)(x)
x = BatchNormalization()(x)
x = AveragePooling1D(2,2)(x)
x = IndRNN(256, return_sequences=True)(x)
x = BatchNormalization()(x)
x = AveragePooling1D()(x)
x = IndRNN(256, return_sequences=True)(x)
x = BatchNormalization()(x)
x = AveragePooling1D()(x)
x = IndRNN(128, return_sequences=True)(x)
x = BatchNormalization()(x)
x = AveragePooling1D()(x)
x = IndRNN(128, return_sequences=True)(x)
x = BatchNormalization()(x)
x = AveragePooling1D()(x)
x = AveragePooling1D()(x)
x = IndRNN(128, return_sequences=False)(x)
x = Dense(128)(x)
x = Dense(128)(x)
predictions = Dense(units=3, activation="softmax")(x)
model = Model(ip, predictions)

adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-3, decay=0, amsgrad=False)

model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy' , km.precision(label=1) , km.recall(label=1)])

# END MODEL DESCRIPTION

epp = EDF_Preprocessor('edfdataset.json')
print('Loading data for training...')
Xl = list()
yl = list()
Xc, yc = epp.get_labeled('chb04_05.edf')
Xl.append(Xc)
yl.append(yc)
Xc, yc = epp.get_labeled('chb04_08.edf')
Xl.append(Xc)
yl.append(yc)
X = np.concatenate(Xl)
y = np.concatenate(yl)
del Xl, yl, Xc, yc

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
del X, y
y_binary_train = to_categorical(y_train)
y_binary_test = to_categorical(y_test)

model.fit(X_train, y_binary_train, batch_size=256, epochs=40, verbose=1, validation_split=0.15, callbacks=[tensorboard])

print('Loading data for training...')
X, y = epp.get_labeled('chb04_28.edf')
y = to_categorical(y, num_classes=3)
model.evaluate(X_test, y_binary_test, batch_size=256)