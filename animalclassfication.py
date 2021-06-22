#globalPath에 현제 실행시킬 환경에 데이터가 저장되어 있는 위치를 입력


from PIL import Image
import os
from glob import glob
import numpy as np
from sklearn.utils import shuffle

class dataSet():
    
    #초기함수
    def __init__(self, globalPath):
        self.globalPath = globalPath
        self.x = []
        self.y = []

        #in splitData
        self.train_x = []
        self.train_y = []
        self.test_x = []
        self.test_y = []
    
    #이미지를 불러오는 함수 (path = 지금 현재 데이터가 저장되어 있는 파일 주소)
    def imageRead(self, path):
        x = Image.open(path)
        y = path.split('\\')[-2]
        #.\\dataset\\1\\1.jpg
        # print(x,y)
        return x, int(y)-1

    #실제로 모든 데이터를 읽어들이는 함수
    def getFilesInFolder(self, path):
        #모든 경로들을 다 가져와서 result에 넣음
        result = [ y for x in os.walk(path) for y in glob(os.path.join(x[0], '*.*'))]
        # print(result)

        for localPath in result:
            img, target = self.imageRead(localPath)
            self.x.append(img)
            self.y.append(target)
        # print(len(self.x), len(self.y))
        return self.x, self.y
    
    #이전에 설정한 dim값으로 데이터 전체를 사이즈를 변환함
    def resizeAll(self, X, Y, dim):
        
        resizedX = []
        resizedY = []

        N = len(X)

        for i in range(N):
            resized = X[i].resize((dim, dim))
            npImg = np.array(resized)

            if len(npImg.shape) == 3:
                resizedX.append(npImg)
                resizedY.append(Y[i])
           # print(npImg.shape)
        
        self.x = np.array(resizedX)
        self.y = np.array(resizedY)
        self.y = np.reshape(self.y, (-1, 1))
        #print(self.x.shape, self.y.shape)
    
    #학습데이터랑 테스트데이터로 나누는 함수
    def splitDataset(self, ratio):
        train_idx = int(len(self.x) * ratio)
        print(train_idx)
        self.train_x, self.train_y = self.x[:train_idx, :, :, :], self.y[:train_idx, :]

        self.test_x, self.test_y = self.x[train_idx:, :, :, :], self.y[train_idx:, :]

        return self.train_x, self.train_y, self.test_x, self.test_y
    
    
    #데이터를 섞는 함수
    def shuffleData(self, x, y):
        x = np.array(x)
        y = np.array(y)
        x, y = shuffle(x, y)
        return x, y

    #정규화 함수
    def normZT(self, x):
        x = (x - np.mean(x) / np.std(x))
        return x
    
    #MinMax정규화 함수
    def normMinMax(self, x):
        x = (x - np.min(x)) / (np.max(x) - np.min(x))
        return x

    #위에 함수들을 다 실행시키는 함수
    def load_data(self, dim, ratio):
        self.getFilesInFolder(self.globalPath) #전체 데이터 가져옴
        self.resizeAll(self.x, self.y, dim) # numpy화 되어 있음
        self.x, self.y = self.shuffleData(self.x, self.y) #데이터 섞기
        self.splitDataset(ratio) #훈련용, 시험용으로 쪼개기
        self.train_x = self.normZT(self.train_x) #train 정규화
        self.test_x = self.normZT(self.test_x) #test 정규화

        return self.train_x, self.train_y, self.test_x, self.test_y



globalPath = 'C:\\code\\dataset\\anmals\\'
ds = dataSet(globalPath)
train_x, train_y, test_x, test_y = ds.load_data(64, 0.8)


model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(input_shape = (128,128,3),kernel_size = (3,3), filters = 32, padding = 'same', activation = 'relu'),
  tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 64, padding = 'same', activation = 'relu'),
  tf.keras.layers.MaxPool2D(pool_size = (2,2)),
  tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 128, padding = 'same', activation = 'relu'),
  tf.keras.layers.Conv2D(kernel_size = (3,3), filters = 256, padding = 'valid', activation = 'relu'),
  tf.keras.layers.MaxPool2D(pool_size = (2,2)),
  tf.keras.layers.Dropout(rate = 0.5),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(units = 512, activation= 'relu'),
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