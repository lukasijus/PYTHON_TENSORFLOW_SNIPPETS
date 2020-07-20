import tensorflow as tf
import datetime

start_time = datetime.datetime.now()
model_name = '/Users/macbook/Software/PYTHON_TENSORFLOW_SNIPPETS/models/2937_SECONDS_(224, 224, 3)0.03_LOSS_0.99_ACCURACY_14_EPOCHS_3_LAYERS_.h5'
plot_name  = '2937_SECONDS_(224, 224, 3)_0.03_LOSS_0.99_ACCURACY_14_EPOCHS_167_LAYERS'
reloaded = tf.keras.models.load_model(model_name)

from CREATE_DATASET_2_2TF import create_dataset
train_data, test_data, image_shape, BATCH_SIZE, classes = create_dataset()

test_data = next(iter(test_data))
test_images = test_data[0]
test_labels = test_data[1]

from CREATE_DATASET_2_2TF import class_names

class_names = list(class_names().values())

from plot_images import plotbatch_v2, predict_image, predict_webcam_image

predictions = reloaded.predict(test_images)


# plotbatch_v2(plot_name, BATCH_SIZE, test_labels,test_images, image_shape, class_names, predictions, save=True)

file_name = '/Users/macbook/Software/PYTHON_TENSORFLOW_SNIPPETS/Lenna.png'
file_name_2 = '/Users/macbook/Software/PYTHON_TENSORFLOW_SNIPPETS/opencv_frame_0.png'
# predict_image(file_name_2, image_shape, reloaded, class_names)

predict_webcam_image(image_shape, reloaded, class_names)
