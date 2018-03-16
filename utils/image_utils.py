# some image loading and processing functions
import cv2
import numpy as np

def load_image(image_path, shape=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # handle grayscale input images
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
    return img
