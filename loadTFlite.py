import numpy as np
import tensorflow as tf
import sys
import cv2

tflite_path = sys.argv[1]
print(tflite_path)

image_path = sys.argv[2]
print(image_path)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)

width = input_details[0]['shape'][2]
height = input_details[0]['shape'][1]

img = cv2.imread(image_path)
img = cv2.resize(img, (width, height))

input_data = np.expand_dims(img, axis=0)

# convert to float32
# input_data = np.float32(input_data) / 255.0
input_data = input_data.astype('float32')

interpreter.set_tensor(input_details[0]['index'], input_data)

print("start")
interpreter.invoke()
print("done")

output_data = interpreter.get_tensor(output_details[0]['index'])

print("\n\nOutput:")
print(output_data)
