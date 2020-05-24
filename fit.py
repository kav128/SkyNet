print('Демонстрация обучения модели на маленьком фрагменте датасета')

import numpy as np
import keras
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, GaussianDropout, LSTM, Bidirectional
import keras_metrics as km
import datetime

from edf_preprocessor import EDF_Preprocessor


# Не работает под Windows. Под Docker пока не проверял
# import subprocess
# subprocess.call(["rm", "/rf", "logs/*"])

# BEGIN MODEL DESCRIPTION

inputs = Input(shape=(256,23))
x = GaussianDropout(0.1)(inputs)	#generating noise during training for better generalization

x = LSTM(256)(x)
x = Dropout(0.2)(x)
x = Dense(256)(x)

x = Dropout(0.2)(x)
x = Dense(256)(x)

x = Dropout(0.2)(x)

predictions = Dense(units=3, activation="softmax")(x)

model = Model(inputs, predictions)

# adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-3, decay=0.0, amsgrad=False)
rmsprop = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
model.compile(rmsprop, loss='categorical_crossentropy', metrics=['accuracy', km.precision(label=1), km.recall(label=1)])

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
Y = np.concatenate(yl)
del Xl, yl, Xc, yc

X, y = epp.get_labeled_range(100, 4)
y = to_categorical(y, num_classes=3)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log_dir)
model.fit(X, y, epochs=15, verbose=1, validation_split=0.25, batch_size=256, callbacks=[tensorboard])
del X, y

print('Loading data for training...')
X, y = epp.get_labeled('chb04_28.edf')
y = to_categorical(y, num_classes=3)
model.evaluate(X, y, batch_size=256)