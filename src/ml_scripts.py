# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array

# img = load_img("static/data/big-mac-1-550.jpg")
# img_array = img_to_array(img)
# print(img_array)

import cv2
import numpy as np 

img = cv2.imread("static/data/big-mac-6-550.jpg")
res = cv2.resize(img, dsize=(720,720), interpolation=cv2.INTER_CUBIC)

print(img.shape)
print(res.shape)

cv2.imshow('image window', res)
cv2.waitKey(0)

