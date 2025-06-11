from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
from classification.ml_logic.preprocessor import Preprocessor

app = FastAPI()

@app.get("/")
def index():
    return {"status": "I like big b**bs and I can not lie.🍉🍈🍒"}

# app.state.model = load_model("models/CNN_Breast_Cancer.keras")
app.state.preprocessor = Preprocessor()

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()
    print(f'Image type after img.read: {type(contents)}')

    # PREPROCESS image for prediction
    contents = Image.open(io.BytesIO(contents))
    print(f'Image type after opening with io.Bytes: {type(contents)}')
    contents = img_to_array(contents)
    print(f'Image type after img_to_array conversion: {type(contents)}')
    print(f'Image shape after img_to_array conversion: {contents.shape}')
    preprocessor = app.state.preprocessor
    image = preprocessor.preprocess_image(contents)
    print(f'Image type after preprocessing: {type(image)}')
    print(f'Image shape after preprocessing: {image.shape}')
    # contents = contents.reshape((-1, 224, 224, 3))
    # model = app.state.model
    # res = model.predict(image)
    # return  {"probability of for malignant BC:": round(float(res),2)}

    # try exept?
