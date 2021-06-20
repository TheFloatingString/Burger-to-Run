from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates


import cv2
from PIL import Image
import io
import numpy as np

from keras.models import load_model


model = load_model("static/results/model.h5")

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/")
def home(request: Request):
	return templates.TemplateResponse("home.html", {"request": request})

@app.post("/form")
async def form(file: UploadFile = File(...)):
	print(file)
	print(type(file))
	return {"abc": file}


@app.post("/uploadPhoto")
async def uploadPhoto(file: UploadFile = File(...)):

	image_array = Image.open(io.BytesIO(file.file.read())).convert("RGB")
	image_array = np.asarray(image_array)

	image_array = cv2.resize(image_array, dsize=(720, 720), interpolation=cv2.INTER_CUBIC)
	image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

	calories = float(model.predict(np.array([image_array])))*1000

	return {"filename": file.filename, "calories": calories}
