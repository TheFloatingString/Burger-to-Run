import cv2
import numpy as np 

import os


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

	cv2.imshow('image window', res)
	cv2.waitKey(0)

X_data = np.array(X_data)
y_data = np.array(y_data)

print(X_data.shape)
print(y_data.shape)
