import tensorflow as tf
import datetime

start_time = datetime.datetime.now()
model_name = '/Users/macbook/Software/PYTHON_TENSORFLOW_SNIPPETS/models/2937_SECONDS_(224, 224, 3)0.03_LOSS_0.99_ACCURACY_14_EPOCHS_3_LAYERS_.h5'
xception_model = '/Users/macbook/Software/PYTHON_TENSORFLOW_SNIPPETS/models/5371_SECONDS_Xception_(224, 224, 3)_0.02_LOSS_1.0_ACCURACY_5_EPOCHS_134_LAYERS_natural_images_.h5'
plot_name  = '5371_SECONDS_Xception_(224, 224, 3)_0.02_LOSS_1.0_ACCURACY_5_EPOCHS_134_LAYERS_natural_images'

reloaded = tf.keras.models.load_model(xception_model)
print(reloaded)
from CREATE_DATASET_2_2TF import create_dataset
train_data, test_data, image_shape, BATCH_SIZE, classes = create_dataset()

test_data = next(iter(test_data))
test_images = test_data[0]
test_labels = test_data[1]

from CREATE_DATASET_2_2TF import class_names

class_names = list(class_names().values())

predictions = reloaded.predict(test_images)

from plot_images import plotbatch_v2, predict_image, predict_webcam_image

# plotbatch_v2(plot_name, BATCH_SIZE, test_labels,test_images, image_shape, class_names, predictions, save=True)

file_name = '/Users/macbook/Software/PYTHON_TENSORFLOW_SNIPPETS/dataset/natural_images/test/car/car_0001.jpg'

# predict_image(file_name, image_shape, reloaded, class_names)

predict_webcam_image(image_shape, reloaded, class_names)
