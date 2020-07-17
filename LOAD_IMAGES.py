import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import pathlib

# url = '' means you already have image folders in C:\Users\%USER\.keras\datasets\
# Otherwise use "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz" for flower dataset
url = ''
data_dir = tf.keras.utils.get_file(
    origin='',
    fname='flower_photos')

data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

# LOAD an Image
roses = list(data_dir.glob('roses/*'))
rose = PIL.Image.open(str(roses[0]))
# rose.show()

# PLOT multiple images
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 5
for i in range(1, columns*rows +1):
    img = plt.imread(str(roses[i]))
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)
plt.show()

