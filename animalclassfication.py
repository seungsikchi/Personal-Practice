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



#불러온 데이터를 convolution을 사용하여 특징을 추출하고 다시 디코딩하기
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters = 32, kernel_size = 2, strides=(2,2), activation = 'elu', input_shape = (64,64,3)),
    tf.keras.layers.Conv2D(filters = 64, kernel_size = 2, strides=(2,2), activation = 'elu'),
    tf.keras.layers.Conv2D(filters = 128, kernel_size = 2, strides=(2,2), activation = 'elu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'elu'),
    tf.keras.layers.Dense(64, activation = 'elu'),
    tf.keras.layers.Dense(8*8*128, activation = 'elu'),
    tf.keras.layers.Reshape(target_shape=(8,8,128)),
    tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=2, strides=(2,2), padding ='same', activation = 'elu'),
    tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=2, strides=(2,2), padding ='same', activation = 'elu'),
    tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=2, strides=(2,2), padding ='same', activation = 'sigmoid')
])

model.compile(optimizer =tf.optimizers.Adam(), loss= 'mse', metrics = ['accuracy'])
model.summary()

print(train_x.shape)
print(train_y.shape)

#위에서 정의한 모델 학습
history = model.fit(train_x, train_x, epochs=50, batch_size= 10)

#랜덤으로 불러와 위에서 디코딩이 잘 되었는지 확인
import random

plt.figure(figsize= (4,8))
for c in range(4):
  plt.subplot(4, 2, c*2+1)
  rand_index = random.randint(0, train_x.shape[0])
  plt.imshow(train_x[rand_index].reshape(64,64,3), cmap = 'gray')
  plt.axis('off')

  plt.subplot(4, 2, c*2+2)
  img = model.predict(np.expand_dims(train_x[rand_index], axis=0))
  plt.imshow(img.reshape(64, 64,3), cmap = 'gray')
  plt.axis('off')

plt.show()

model.evaluate(test_x, test_x)
