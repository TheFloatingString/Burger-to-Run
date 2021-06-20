from fastapi import FastAPI, UploadFile, File
import cv2
from PIL import Image
import io
import numpy as np

app = FastAPI()

@app.get("/")
def home():
	return "test!"

@app.post("/uploadPhoto")
async def uploadPhoto(file: UploadFile = File(...)):

	image_array = Image.open(io.BytesIO(file.file.read())).convert("RGB")
	image_array = np.asarray(image_array)

	image_array = cv2.resize(image_array, dsize=(720, 720), interpolation=cv2.INTER_CUBIC)
	image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)

	return {"filename": file.filename}
