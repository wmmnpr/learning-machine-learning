
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import datetime
import shutil

print(tf.__version__)
runtime = datetime.datetime.now()
runtime_ts = runtime.strftime("%Y%m%d-%H%M%S")


# Load the Fashion MNIST dataset
mnist = tf.keras.datasets.mnist

# Load the training and test split of the Fashion MNIST dataset
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
# You can put between 0 to 59999 here
index = 0

number_distinct_training_labels = len(set(training_labels))
print(f"number of distinct training labels is {number_distinct_training_labels}")
# Set number of characters per row when printing
np.set_printoptions(linewidth=320)

# Print the label and image
#print(f'LABEL: {training_labels[index]}')
#print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')

#print("Visualize the image")
#plt.imshow(training_images[index])

print("Swap background color with digit one and then normalize the pixel values of the train and test images")
training_images  = 255.0 - training_images
test_images = 255.0 - test_images

training_images  = training_images / 255.0
test_images = test_images / 255.0


nodes_layer_1 = 128
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(nodes_layer_1, activation=tf.nn.relu), 
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])

print("Compiling the classification model")
optimizer = tf.keras.optimizers.legacy.Adam()
model.compile(optimizer = optimizer,
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"logs/fit/{runtime_ts}", histogram_freq=1, write_images=True),
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.99): # Experiment with changing this value
      print("\nReached 90% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

from confusion_matrix_callback import ConfusionMatrixCallback
confusion_matrix_callback = ConfusionMatrixCallback(test_images, test_labels, log_dir="logs/confusion")


epochs = 30
print(f"Fitting the classification model with epochos {epochs} ------------------------------------- ")
history = model.fit(training_images, training_labels, epochs = epochs, callbacks=[tensorboard_callback, confusion_matrix_callback])

#Saving the model
model.save('digit-model3.h5')


print("Evaluate the model on unseen data --------------------------------------- ")
model.evaluate(test_images, test_labels)

print("Predict test image data    ----------------------")
classifications = model.predict(test_images)

tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# Save model architecture as JSON
# can be viewed with netron model.json
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)

tst_idx = 10
print(f"Classifications of test images at {tst_idx}")
print(classifications[tst_idx])

print(f"test label at {tst_idx} ")
print(test_labels[tst_idx])

#print(f'LABEL: {test_labels[tst_idx]}')
#print(f'\nIMAGE PIXEL ARRAY:\n {test_images[tst_idx]}')

print("Visualize the image")
#plt.imshow(test_images[tst_idx])

# Creates a file writer for the log directory.
file_writer = tf.summary.create_file_writer("logs/train_data/{runtime_ts}")

# Using the file writer, log the reshaped image.
with file_writer.as_default():
  # Don't forget to reshape.
  images = np.reshape(training_images[0:25], (-1, 28, 28, 1))
  tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    #plt.plot(history.history[f'val_{metric}'])
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    #plt.legend([metric, f'val_{metric}'])
    plt.show()
    
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

print("run: tensorboard --logdir logs")
