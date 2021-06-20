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