import os    # to work with operating system
import cv2
import numpy as np

img_dir = os.path.join(os.getcwd(), "images")
print()

def preprocess(img):
    img = cv2.resize(img, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

images = []
labels = []
for i in os.listdir(img_dir):
    img = cv2.imread(os.path.join(img_dir, i))
    img = preprocess(img)
    images.append(img)
    labels.append(i.split("_")[0])
    
images = np.array(images)
labels = np.array(labels)


import pickle     # store python variable data

# store the data
with open("images.p", "wb") as f:
    pickle.dump(images, f)

with open("labels.p", "wb") as f:
    pickle.dump(labels, f)
    