import numpy as np
import cv2
import os
import pickle

directory = r'cats_dogs_object_detection_dataset/'
type = ['cat', 'dog']

data = []

for t in type:
    path = os.path.join(directory, t)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        label = type.index(t)
        arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            new_arr = cv2.resize(arr, (224, 224))
            data.append([new_arr, label])
        else:
            print("Error:", img_path)


data[0][1]

X = []
Y = []

for feature, label in data:
    X.append(feature)
    Y.append(label)

X = np.array(X)
Y = np.array(Y)

pickle.dump(X, open('X.pkl', 'wb'))
pickle.dump(Y, open('Y.pkl', 'wb'))