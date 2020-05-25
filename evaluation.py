from keras.models import load_model
from keras.utils import to_categorical
import keras_metrics as km
import numpy as np

from edf_preprocessor import EDF_Preprocessor
from ind_rnn import IndRNN
from ind_rnn import IndRNNCell, RNN

cus = {'binary_precision': km.precision(label=1),
       'binary_recall': km.recall(label=1),
       'IndRNN': IndRNN}
model = load_model('IndRNN_firstPatient_1931190540epochs.h5', custom_objects=cus)

epp = EDF_Preprocessor('edfdataset.json')
print('Loading data...')
X, y = epp.get_labeled_range(100, 4)
y = to_categorical(y, num_classes=3)

eval = model.evaluate(X, y, batch_size=512)
print("Loss:", eval[0], "accuracy:", eval[1], "precision:", eval[2], "recall:", eval[3])