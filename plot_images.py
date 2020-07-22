# This function plots images
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import os

from colour import Color
import numpy as np
import cv2

red = Color("red")

def plotimages(class_names, dataset, imageCount):
    i = 0
    for image, label in dataset.take(imageCount):
        image = image.numpy().reshape((28, 28))
        x_axis_length = int(round(math.sqrt(imageCount)))
        y_axis_length = int(round(math.sqrt(imageCount)))
        fig, ax = plt.subplot(x_axis_length, y_axis_length, i + 1)
        ax.imshow(image, cmap=plt.cm.binary)
        rect = patches.Rectangle((1, 1), 26, 26, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        plt.grid(False)
        plt.xlabel(class_names[label])
        i = i + 1
    plt.show()

def plotbatch(name, loss, accuracy, batch_size, test_labels,class_names,predictions,test_images, save):
    index = 0
    loss = float("{:.2f}".format(loss))
    accuracy = float("{:.2f}".format(accuracy))
    loss1 = loss * 100
    accuracy1 = accuracy * 100
    fig, axes = plt.subplots(nrows=int(round(math.sqrt(batch_size))), ncols=int(round(math.ceil(math.sqrt(batch_size)))))
    fig.suptitle(name + '\n' + 'Loss: ' + str(loss) + ' ' + 'Accuracy: ' + str(accuracy))
    axes = axes.flatten()
    plt.subplots_adjust(
        left=0.03,
        bottom=0.13,
        right=0.98,
        top=0.93,
        wspace=0.95,
        hspace=0.98)
    plt.grid(False)
    for image in test_images:
        rect = patches.Rectangle((0, 0), 27, 27, linewidth=5, edgecolor='g', facecolor='none')
        class_name_index = 0
        highest_prediction = 0
        highest_prediction_index = 0
        for value in predictions[index]:
            if value > highest_prediction:
                highest_prediction = value
                highest_prediction_index = class_name_index
            class_name_index += 1
        axes[index].set_xlabel('P: ' + str(float("{:.2f}".format(highest_prediction))) +
                   '\n' + 'PC: ' + class_names[highest_prediction_index] +
                   '\n' + 'GTC: ' + class_names[test_labels[index]],
                   fontsize = 6)
        if highest_prediction_index != test_labels[index]:
            rect = patches.Rectangle((0, 0), 27, 27, linewidth=5, edgecolor='r', facecolor='none')
        axes[index].add_patch(rect)
        axes[index].xaxis.set_label_coords(0.05, -0.025)
        axes[index].imshow(test_images[index].reshape((28,28)), cmap=plt.cm.binary)
        axes[index].set_yticklabels([])
        axes[index].set_xticklabels([])
        index += 1
    if save:
        name  = 'LOSS_' + str(loss) + '_ACCURACY_' + str(accuracy) + name + '.png'
        plt.savefig(name)
    plt.show()


def plotImagesGrid(img_arr, col = 1, row = 1):
    fig, axes = plt.subplots(col, row, figsize = (20, 20))
    axes = axes.flatten()
    for img, ax in zip(img_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

def plot_images_grid_with_labels(img_arr, labels = [], row = 1, col = 1):
    fig, axes = plt.subplots(row, col, figsize = (20, 20))
    axes = axes.flatten()
    count = 0
    for img, ax in zip(img_arr, axes):
        ax.imshow(img)
        ax.set_title(labels[count])
        count += 1
    plt.tight_layout()
    plt.show()

def plot_images_grid_with_labels(img_arr, labels = [], predictions = [], row = 1, col = 1):
    fig, axes = plt.subplots(row, col, figsize = (20, 20))
    axes = axes.flatten()
    count = 0
    for img, ax in zip(img_arr, axes):
        ax.imshow(img)
        title = labels[count] + predictions[count]
        ax.set_title(title)
        count += 1
    plt.tight_layout()
    plt.show()

def plotbatch_v2(name, batch_size, test_labels,test_images, image_shape, class_names,predictions, save=False):
    index = 0
    rows = math.ceil(math.sqrt(batch_size) * 0.6)
    cols = math.ceil(math.sqrt(batch_size) * 1.4)
    fig, axes = plt.subplots(rows, cols)
    fig.suptitle(name)
    axes = axes.flatten()
    plt.subplots_adjust(
        left=0.03,
        bottom=0.13,
        right=0.98,
        top=0.93,
        wspace=0.95,
        hspace=0.98)
    plt.grid(False)
    for image in test_images:
        rect = patches.Rectangle((0, 0), image_shape[0], image_shape[1], linewidth=5, edgecolor='g', facecolor='none')
        class_name_index = 0
        highest_prediction = 0
        highest_prediction_index = 0
        for value in predictions[index]:
            if value > highest_prediction:
                highest_prediction = value
                highest_prediction_index = class_name_index
            class_name_index += 1
        axes[index].set_xlabel('P: ' + str(float("{:.2f}".format(highest_prediction))) +
                   '\n' + 'PC: ' + class_names[highest_prediction_index] +
                   '\n' + 'GTC: ' + class_names[int(test_labels[index])],
                   fontsize = 6)
        if highest_prediction_index != test_labels[index]:
            rect = patches.Rectangle((0, 0), image_shape[0], image_shape[1], linewidth=5, edgecolor='r', facecolor='none')
        axes[index].add_patch(rect)
        axes[index].xaxis.set_label_coords(0.05, -0.025)
        axes[index].imshow(test_images[index])
        axes[index].set_yticklabels([])
        axes[index].set_xticklabels([])
        index += 1
    if save:
        name = name + '.png'
        name = os.path.join(os.path.join(os.path.abspath(''), 'prediction_plots'), name)
        plt.savefig(name)
    plt.show()

def predict_image(name, image_shape, model, class_names):
    img = cv2.imread(name)
    size = (image_shape[0], image_shape[1])
    img = cv2.resize(img, dsize=size)
    img = np.asarray(img)
    # convert from integers to floats
    img = img.astype('float32')
    # normalize to the range 0-1
    img /= 255.0
    img_to_prediction = np.reshape(img,[1,image_shape[0],image_shape[1],image_shape[2]])
    print(img_to_prediction.shape)
    prediction = model.predict(img_to_prediction)
    print(prediction)
    index = np.argmax(prediction)
    name = class_names[index]
    percent = float("{:.2f}".format(prediction[0][index]))
    print('Predicted: ', name, ' with ', percent, '% accuracy')
    img_name = 'Predicted: ' + name + ' with ' + str(percent) + '% accuracy'
    cv2.imshow(img_name, img)
    cv2.waitKey()

def predict_webcam_image(image_shape, model, class_names):
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame[0:400, 0:400])

        k = cv2.waitKey(1)
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            # img_name = "opencv_frame_{}.png".format(img_counter)
            # cv2.imwrite(img_name, frame)
            # print("{} written!".format(img_name))
            img_counter += 1
            size = (image_shape[0], image_shape[1])
            frame = frame[0:400, 0:400]
            img = cv2.resize(frame, dsize=size)
            img = np.asarray(img)
            # convert from integers to floats
            img = img.astype('float32')
            # normalize to the range 0-1
            img /= 255.0
            img_to_prediction = np.reshape(img, [1, image_shape[0], image_shape[1], image_shape[2]])
            prediction = model.predict(img_to_prediction)
            index = np.argmax(prediction)
            name = class_names[index]
            percent = float("{:.2f}".format(prediction[0][index]))
            print('Predicted: ', name, ' with ', percent, '% accuracy')
            img_name = 'Predicted: ' + name + ' with ' + str(percent) + '% accuracy'
            cv2.imshow(img_name, img)
    cam.release()
    cv2.destroyAllWindows()
