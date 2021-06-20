import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os 
import cv2
from sklearn.model_selection import train_test_split

Hippo = []
Penguin = []
Giraffe = []
Panda = []
Elephant = []
Y = []


# def normalize(x):
#   return (x-np.min(x))/(np.max(x)-np.min(x))


img_dir = ("C:/code/dataset/zoodata/1")
img_list = os.listdir(img_dir)
for i in range(1,131):
    image_file = "/code/dataset/zoodata/1/img_%d_resize.jpeg"%i
    image = cv2.imread(image_file)
#     no_X = normalize(image)
    Hippo.append(image)
    Hippo_np = np.array(Hippo)
    Y.append(1)
    
img_dir = ("C:/code/dataset/zoodata/2")
img_list = os.listdir(img_dir)
for i in range(1,131):
    image_file = "/code/dataset/zoodata/2/img_%d_resize.jpeg"%i
    image = cv2.imread(image_file)
#     no_X = normalize(image)
    Penguin.append(image) 
    Penguin_np = np.array(Penguin)
    Y.append(2)   
for i in range(1,131):  
    image_file = "/code/dataset/zoodata/3/img_%d_resize.jpeg"%i 
    image = cv2.imread(image_file)
#     no_X = normalize(image)
    Giraffe.append(image)
    Giraffe_np = np.array(Giraffe)
    Y.append(3)
    
img_dir = ("C:/code/dataset/zoodata/4")
img_list = os.listdir(img_dir)
for i in range(1,131):
    image_file = "/code/dataset/zoodata/4/img_%d_resize.jpeg"%i
    image = cv2.imread(image_file)
#     no_X = normalize(image)
    Panda.append(image)
    Panda_np = np.array(Panda)
    Y.append(4)


    
img_dir = ("C:/code/dataset/zoodata/5")
img_list = os.listdir(img_dir)
for i in range(1,131):
    image_file = "/code/dataset/zoodata/5/img_%d_resize.jpeg"%i
    image = cv2.imread(image_file)
#     no_X = normalize(image)
    Elephant.append(image)
    Elephant_np = np.array(Elephant)
    Y.append(5)

# print(giraffe)
# print(panda)
# print(a)
print(Hippo_np.shape)
print(Penguin_np.shape)
print(Giraffe_np.shape)
print(Panda_np.shape)
print(Elephant_np.shape)
           
    
BeforeExtraction_X = np.concatenate((Hippo_np,Penguin_np,Giraffe_np,Panda_np,Elephant_np), axis = 0)
BeforeExtraction_Y = np.array(Y)


train_X, test_X, train_Y, test_Y = train_test_split(BeforeExtraction_X, BeforeExtraction_Y, test_size = 0.3)

# x_mean = train_X.mean()
# x_std = train_X.std() 
# train_X = (train_X - x_mean) / x_std
# test_X = (test_X - x_mean) / x_std

print(test_X)
train_X = train_X / 255.0
test_X = test_X / 255.0

print(train_X)
print(train_Y)

model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(input_shape = (128,128,3),kernel_size = (3,3), filters = 32, padding = 'same', activation = 'relu'),
  tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 64, padding = 'same', activation = 'relu'),
  tf.keras.layers.MaxPool2D(pool_size = (2,2)),
#   tf.keras.layers.Dropout(rate = 0.5),
  tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 128, padding = 'same', activation = 'relu'),
  tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 256, padding = 'valid', activation = 'relu'),
  tf.keras.layers.MaxPool2D(pool_size = (2,2)),
  tf.keras.layers.Dropout(rate = 0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units = 512, activation= 'relu'),
#   tf.keras.layers.Dropout(rate = 0.5),
  tf.keras.layers.Dense(units = 256, activation= 'relu'),
  tf.keras.layers.Dropout(rate = 0.5),
  tf.keras.layers.Dense(units = 10, activation= 'softmax'),
])

model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.summary()

history = model.fit(train_X, train_Y, epochs = 50, validation_split = 0.2)

import matplotlib.pyplot as plt

plt.figure(figsize = (12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], 'b-', label = 'loss')
plt.plot(history.history['val_loss'], 'r--', label = 'val_loss')
plt.xlabel('Epochs')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], 'g-', label = 'accuracy')
plt.plot(history.history['val_accuracy'], 'k--', label = 'val_accuracy')
plt.xlabel('Epochs')
plt.ylim(0.7, 1)
plt.legend()

plt.show()

print(model.evaluate(test_X, test_Y, verbose = 0))
