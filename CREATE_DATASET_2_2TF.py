import tensorflow as tf
from SPLIT_DATASET import make_train_and_test_sets
from plot_images import plotImagesGrid, plot_images_grid_with_labels

train_examples, test_examples, classes, train_dir, test_dir = make_train_and_test_sets()

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

TARGET_SIZE = (224, 224)
IMAGE_SHAPE = (TARGET_SIZE[0], TARGET_SIZE[1], 3)
BATCH_SIZE = 32
CLASS_MODE = 'binary'

def image_shape():
    return  IMAGE_SHAPE

def class_names():
    return classes

def create_dataset(plot = False):
    train_data = data_gen.flow_from_directory(
        train_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE)

    test_data = data_gen.flow_from_directory(
        test_dir,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        class_mode=CLASS_MODE)

    train_data_iter = iter(train_data)
    batch = next(train_data_iter)
    batch_images = batch[0]
    batch_labels = batch[1]

    labels = []
    for label in batch_labels:
        labels.append(classes[label])
    if plot:
        plot_images_grid_with_labels(batch_images, labels, row=4, col=8)
    return train_data, test_data, IMAGE_SHAPE, BATCH_SIZE, classes

# train_data, test_data = create_dataset()








