import numpy as np
import sys
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

model_path = 'models/digit-model2.h5'  # Replace with the path to your HDF5 file

if len(sys.argv) < 2:
    print("Usage: predict path_to_input_image")
    exit(1)

image_path = sys.argv[1]

# Read image
image_to_identify = cv2.imread(image_path,0)

# Display image
fig, axes = plt.subplots(1, 1, figsize=(5, 5))
axes.imshow(image_to_identify, cmap='gray')

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model(model_path) 

# prepare image and run prediction
image_to_identify = image_to_identify / 255.0   # normalize values between 0 and 1
image_to_identify = image_to_identify.reshape(1, 28, 28)
prediction = model.predict(image_to_identify)

# You can now work with the 'predictions' array as needed
# array index, 0 offset, corresponds to digit e.g. prediction[4] corresponds to 4
print(prediction[0])

max_index, max_value = max(enumerate(prediction[0]), key=lambda x: x[1])
max_value = round(max_value * 100, 2)

print("//----------------------------//")
print(f"With {max_value}% certainty the digit is a: {max_index}")
print("//----------------------------//")
plt.show(block=True)

