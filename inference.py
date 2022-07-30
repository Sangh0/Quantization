import numpy as np
import tensorflow as tf

def inference(tflite_file, test_data):
    # initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()

    # get indexes of input and output layers
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    # inference step
    correct = 0
    
    for batch, (image, label) in enumerate(test_data):
        image = tf.expand_dim(image, axis=0)

        # step 1. enter image into the model
        interpreter.set_tensor(input_index, image)
        # step 2. run inference
        interpreter.invoke()
        # step 3. interpret output
        prediction = interpreter.get_tensor(output_index)

        if np.argmax(prediction) == np.argmax(label):
            correct += 1

    # calculate accuracy
    mean_acc = correct / (batch+1)

    print(f'Accuracy of TFLite Model: {mean_acc}')