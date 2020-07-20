import tensorflow as tf
import os
import datetime

start_time = datetime.datetime.now()

model_dir = os.path.join(os.path.abspath(''), 'models')
model_list = os.listdir(model_dir)
model_name = model_list[0]
model_path = os.path.join(model_dir, model_name)
print(model_path)
reloaded = tf.keras.models.load_model(model_path)

LAYERS = len(reloaded.layers)

from CREATE_DATASET_2_2TF import create_dataset
train_data, test_data, image_shape, BATCH_SIZE, classes = create_dataset()

EPOCH = 10
# V Train the model V
history = reloaded.fit(
    train_data,
    epochs=EPOCH,
    validation_data=test_data
)


# V Model Evaluation V
test_loss, test_accuracy = reloaded.evaluate(
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
IMG_SHAPE = str(image_shape) + '_IMG_SHAPE_'
LOSS = str(loss_str) + '_LOSS_' + str(accuracy_str) + '_ACCURACY_'
EPOCH = str(EPOCH) + '_EPOCHS_' + str(LAYERS) + '_LAYERS_'
plot_name = TIME + IMG_SHAPE + LOSS + EPOCH + '.png'
plot_file_name = os.path.join(os.path.join(os.path.abspath(''), 'loss_acc_plots'), plot_name)
print(plot_file_name)
plt.savefig(plot_file_name)

model_name = TIME + IMG_SHAPE + LOSS + EPOCH  +'.h5'
model_file_name = os.path.join(os.path.join(os.path.abspath(''), 'models'), model_name)
# Save model
print(model_file_name)
reloaded.save(model_file_name)
