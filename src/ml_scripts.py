import cv2
import numpy as np 

import os

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


PATH = "static/data"

X_data = []
y_data = []

list_of_files = os.listdir(PATH)
for filename in list_of_files:
	print(PATH+'/'+filename)

	img = cv2.imread(PATH+'/'+filename)
	res = cv2.resize(img, dsize=(720,720), interpolation=cv2.INTER_CUBIC)

	X_data.append(res)
	print(filename.split('-')[-1].replace('.jpg', ''))
	y_data.append(np.array([float(filename.split('-')[-1].replace('.jpg', ''))]))

	print(img.shape)
	print(res.shape)

	# cv2.imshow('image window', res)
	cv2.waitKey(0)

X_data = np.array(X_data)
y_data = np.array(y_data)

print(X_data.shape)
print(y_data.shape)


model = Sequential()

model.add(Conv2D(32, (3,3), input_shape=(720, 720, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['mse'])

model.fit(
        X_data, y_data,
        epochs=20,
        verbose=1)

model.save_weights("static/results/model.h5")