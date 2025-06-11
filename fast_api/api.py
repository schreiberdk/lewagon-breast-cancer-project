from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.models import load_model

app = FastAPI()

@app.get("/")
def index():
    return {"status": "I like big b**bs and I can not lie.🍉🍈🍒"}

#app.state.model = load_model("models/InceptionV3_Breast_Cancer.keras")

@app.post('/upload_image')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()

    # PREPROCESS image?

    #model = app.state.model
    #res = model.predict(contents)
    res = "not predicting yet"
    return  {"0 for neg. and 1 for pos. for malignant BC:": str(res)}

    # try exept?
