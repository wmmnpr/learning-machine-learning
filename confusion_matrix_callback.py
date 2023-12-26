import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import datetime
import seaborn
import matplotlib.pyplot as plt

# Define a custom callback to calculate and log the confusion matrix as an image
class ConfusionMatrixCallback(tf.keras.callbacks.Callback):


    def __init__(self, x_train, y_train, log_dir):
        self.x_train = x_train
        self.y_train = y_train
        self.tb_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        # Get predictions on the training data
        y_pred = np.argmax(self.model.predict(self.x_train), axis=1)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(self.y_train, y_pred)

        # Log confusion matrix as an image in TensorBoard
        with self.tb_writer.as_default():
            tf.summary.image('Confusion Matrix', self.plot_confusion_matrix(conf_matrix), step=epoch)


    def get_image_summary(self, conf_matrix, step):
        figure = self.plot_confusion_matrix(conf_matrix)
        image = self.plot_to_image(figure)

        # Create a summary object
        image_summary = tf.keras.utils.Summary()
        image_summary.value.add(tag='Confusion Matrix', image=image, step=step)

        return image_summary

    def plot_confusion_matrix(self, conf_matrix):
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Create a heatmap using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                    yticklabels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        buf = self.plot_to_image(plt)
        plt.close()

        return buf

    def plot_to_image(self, figure):
        import io
        from PIL import Image

        # Save the figure to a PNG image in memory
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        plt.close()

        # Convert the PNG image to a Tensorflow Image Summary
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)

        return image
