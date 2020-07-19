import tensorflow as tf
import os


model_dir = os.path.join(os.path.abspath(''), 'models')
model_list = os.listdir(model_dir)
model_name = model_list[0]
model_path = os.path.join(model_dir, model_name)
model = tf.keras.models.load_model(model_path)
print(model_path)
print(model.summary())

from CREATE_DATASET_2_2TF import create_dataset
train_data, test_data, image_shape, BATCH_SIZE, class_names = create_dataset()

test_batch = next(iter(test_data))
test_images = test_batch[0]
test_labels = test_batch[1]
predictions = model.predict(test_images)
print(predictions)