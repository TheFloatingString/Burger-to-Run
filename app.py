from flask import Flask, render_template, request


import cv2
from PIL import Image
import io
import numpy as np

from keras.models import load_model


import tensorflow as tf


model = load_model("static/results/model.h5")

app = Flask(__name__)

@app.route("/")
def home():
	return render_template("home.html")


@app.route("/uploadPhoto", methods=["GET", "POST"])
def uploadPhoto():



	file = request.files["file"]
	file.save("static/temp_file.png")

	image_array = Image.open("static/temp_file.png").convert("RGB")
	image_array = np.asarray(image_array)

	print(type(image_array))
	print(image_array.shape)

	image_array = cv2.resize(image_array, dsize=(720, 720), interpolation=cv2.INTER_CUBIC)
	image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

	calories = float(model.predict(np.array([image_array])))*1000


	return {"filename": file.filename, "calories": calories, "distance": float(calories/60)}

if __name__ == '__main__':
	app.run()