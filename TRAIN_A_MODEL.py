from CREATE_DATASET_2_2TF import create_dataset
from plot_images import plotbatch
import tensorflow as tf
import datetime
import os
import math

start_time = datetime.datetime.now()

train_data, test_data, image_shape, BATCH_SIZE, classes = create_dataset()


# Build a model
from BUILD_A_MODEL import model, mobilenet_model

model = model()

# Compile a model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# # Iteration parameters
LAYERS = len(model.layers)
EPOCH = 2


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class_names = list(classes.values())

earlystoppping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

# V Train the model V
history = model.fit(
    train_data,
    epochs=EPOCH,
    validation_data=test_data,
    callbacks=[tensorboard_callback, earlystoppping_callback]
)

# V Model Evaluation V
test_loss, test_accuracy = model.evaluate(
    test_data
)

print('Accuracy of the model:', test_accuracy)


# Create model and history object file names
loss_str = float("{:.2f}".format(test_loss))
accuracy_str = float("{:.2f}".format(test_accuracy))

EPOCH = len(history.history['loss'])

exec_time = datetime.datetime.now() - start_time
TIME = str(exec_time.seconds) + '_SECONDS_'
IMG_SHAPE = str(image_shape)
LOSS = str(loss_str) + '_LOSS_' + str(accuracy_str) + '_ACCURACY_'
EPOCH = str(EPOCH) + '_EPOCHS_' + str(LAYERS) + '_LAYERS_'
model_name = TIME + IMG_SHAPE + LOSS + EPOCH + '.h5'
model_file_name = os.path.join(os.path.join(os.path.abspath(''), 'models'), model_name)

# Save model
print(model_file_name)
model.save(model_file_name)



