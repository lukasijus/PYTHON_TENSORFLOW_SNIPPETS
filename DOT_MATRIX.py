import cv2 as cv
import numpy as np
import opencv_functions.FEATURE_DETECTION as features

file_path = '/Users/macbook/Software/PYTHON_TENSORFLOW_SNIPPETS/dataset/dot_matrix/dota_matrix_7_by_5.png'
img = cv.imread(file_path)
kernel = np.ones((10,10),np.uint8)
erosion = cv.erode(img,kernel,iterations = 1)
stacked = np.vstack((img, erosion))
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', '/',
               'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
               'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
x = 0
y = 0
for i in range(len(class_names)):
    crop_image = img[20 + y:140 + y, 20 + x:140 + x]
    x += 105
    if class_names[i] == '/' and class_names[i] == 'L' and class_names[i] == 'Z' :
        x = 0
    cv.imshow(class_names[i], crop_image)


    cv.waitKey()



