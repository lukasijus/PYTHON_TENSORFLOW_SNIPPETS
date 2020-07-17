# Import Libraries

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

# Load and prepare MNIST handwritten digits data

mnist = tf.keras.datasets.mnist
(img_train, labels_train), (img_test, labels_test) = mnist.load_data()
img_train, img_test = img_train / 255.0, img_test / 255.0

# Build the tf.keras.Sequential model by stacking layers. Choose an optimizer and loss function for training:

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# For each example the model returns a vector of "logits" or "log-odds" scores, one for each class.

predictions = model(img_train[:1]).numpy()

# The tf.nn.softmax function converts these logits to "probabilities" for each class:
# Note: It is possible to bake this tf.nn.softmax in as the activation function for the last layer of the network. While this can make the model output more directly interpretable, this approach is discouraged as it's impossible to provide an exact and numerically stable loss calculation for all models when using a softmax output.
predictions = tf.nn.softmax(predictions).numpy()

# The losses.SparseCategoricalCrossentropy loss takes a vector of logits and a True index and returns a scalar loss for each example.

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.log(1/10) ~= 2.3.

loss_fn(img_train[:1], predictions).numpy()

# Compile model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# The Model.fit method adjusts the model parameters to minimize the loss:

model.fit(img_train, labels_train, epochs=5)

# The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".

model.evaluate(img_test, labels_test)

# If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])