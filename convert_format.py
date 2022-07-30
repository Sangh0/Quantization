import tensorflow as tf
from tensorflow import keras

# h5 to pb file
saved_model_h5 = 'directory of h5 files'
model = tf.keras.models.load_model(saved_model_h5, compile=False)
model.save('directory to save pb file')

# pb to h5 file
saved_model_pb = 'directory of pb files'
model = tf.keras.models.load_model(saved_model_pb)
tf.keras.models.save_model(model, 'directory to save h5 file')