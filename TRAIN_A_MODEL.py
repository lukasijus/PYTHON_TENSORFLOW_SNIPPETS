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

model = mobilenet_model()

# Compile a model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

# # Iteration parameters
LAYERS = len(model.layers)
EPOCH = 50


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class_names = list(classes.values())

earlystoppping_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

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


# Plot Training and Validation Graphs

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

loss_str = float("{:.2f}".format(test_loss))
accuracy_str = float("{:.2f}".format(test_accuracy))

epochs_range = range(EPOCH)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
exec_time = datetime.datetime.now() - start_time
TIME = str(exec_time.seconds) + '_SECONDS_'
IMG_SHAPE = str(image_shape)
LOSS = str(loss_str) + '_LOSS_' + str(accuracy_str) + '_ACCURACY_'
EPOCH = str(EPOCH) + '_EPOCHS_' + str(LAYERS) + '_LAYERS_'
plot_name = TIME + IMG_SHAPE + LOSS + EPOCH + '.png'
plot_file_name = os.path.join(os.path.join(os.path.abspath(''), 'loss_acc_plots'), plot_name)
print(plot_file_name)
plt.savefig(plot_file_name)

model_name = TIME  + str(loss_str) + '_LOSS_' + str(accuracy_str) + '_ACCURACY_'  + str(EPOCH) + '_EPOCHS_' + str(LAYERS) + '_LAYERS_' + '.h5'
model_file_name = os.path.join(os.path.join(os.path.abspath(''), 'models'), model_name)
# Save model
print(model_file_name)
model.save(model_file_name)
# # Make predictions
# from plot_images import plotbatch
# name = '_LAYERS_' + str(LAYERS) + '_BATCH_SIZE_=_' + str(BATCH_SIZE) + '_Epoch_=_' + str(EPOCH)
# test_batch = next(iter(test_data))
# test_images = test_batch[0]
# test_labels = test_batch[1]
# predictions = model.predict(test_images)
# print(predictions)
# import matplotlib.pyplot as plt
# import numpy as np
# for i in range(BATCH_SIZE):
#     pr_i =  np.argmax(predictions[i])
#     pr_i_name = classes[pr_i]
#     print(test_labels[i], class_names[int(test_labels[i])], predictions[i], pr_i, pr_i_name )
#
#


