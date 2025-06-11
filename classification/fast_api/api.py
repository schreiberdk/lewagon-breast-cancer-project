from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def index():
    return {"status": "I like big b**bs and I can not lie.🍉🍈🍒"}

app.state.model = load_model("models/CNN_Breast_Cancer.keras")

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    # PREPROCESS image for prediction
    contents = Image.open(io.BytesIO(contents)).resize((224,224))
    contents = img_to_array(contents)/255
    contents = contents.reshape((-1, 224, 224, 3))
    model = app.state.model
    res = model.predict(contents)
    return  {"probability of for malignant BC:": round(float(res),2)}

    # try exept?
