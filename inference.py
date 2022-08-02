import time
import numpy as np
import tensorflow as tf

def inference(tflite_file, test_images, test_labels):
    # initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()

    # get indexes of input and output layers
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']

    # inference step
    correct = 0
    start = time.time()

    for i in range(len(test_images)):
        image = np.expand_dims(test_images[i], axis=0).astype(np.uint8)
        
        # step 1. enter image into the model
        interpreter.set_tensor(input_index, image)
        # step 2. run inference
        interpreter.invoke()
        # step 3. interpret output
        output = interpreter.tensor(output_index)
        output = np.argmax(output()[0])
        
        if output == test_labels[i]:
            correct += 1
        
    accuracy = correct / (len(test_images))
    print(f'Accuracy : {accuracy}, Time: {time.time() - start}')
    
    return accuracy