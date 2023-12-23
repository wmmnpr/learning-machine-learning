import tensorflow as tf
import numpy as np
import datetime

# Define the inputs and labels for the AND gate
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([[0], [0], [0], [1]])

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

# Compile the model with mean squared error loss
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
# Train the model for more epochs
model.fit(inputs, labels, epochs=1000, verbose=0, callbacks=[tensorboard_callback])

# Evaluate the model
loss, accuracy = model.evaluate(inputs, labels)

# Print the predicted outputs
predictions = model.predict(inputs)
print("Predictions:")
print(predictions.round())

# Print the accuracy
print(f"Model Accuracy: {accuracy}")
