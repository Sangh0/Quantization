import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

assert float(tf.__version__[:3]) >= 2.3


# Load dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype(np.float32) / 255.
X_test = X_test.astype(np.float32) / 255.

# Build CNN model
keras_model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(28,28)),
    tf.keras.layers.Reshape((28,28,1)),
    tf.keras.layers.Conv2D(16, kernel_size=(3,3), strides=1, padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# Model compile 
keras_model.compile(
    optimizer='sgd', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# Train model
keras_model.fit(X_train, y_train, epochs=3, validation_split=0.2)

# Save weight file of model
### ex1) save h5 format file
keras_model.save('./my_model.h5')
### ex1) save pb file
save_dict = './quantization/my_model'
if not os.path.exists(save_dict):
    os.mkdir(save_dict)
keras_model.save(save_dict)

# Load weight file
### ex1) load h5 file
model_h5_path = './my_model.h5'
### ex2) load pd file
model_pb_path = './quantization/my_model' 

# Provide RepresentativeDataset for quantization of parameter data of input and output in model
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
    yield [input_value]

# Convert from weight of model to tflite 
### ex1) keras model
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
### ex2) h5 weight file
converter = tf.lite.TFLiteConverter.from_saved_model(model_h5_path)
### ex3) pb weight file
converter = tf.lite.TFLiteConverter.from_saved_model(model_pb_path)

# Quantize all parameter of model through optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

converter.representative_dataset = representative_data_gen

# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set the input and output tensors to uint8
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

# Save the quantized model
open('save_quantized_file.tflite', 'wb').write(tflite_model_quant)

# Check if quantized
interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)