from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from classification.ml_logic.preprocessor import Preprocessor
from segmentation.ml_logic.preprocessor import Preprocessor
import numpy as np
from PIL import Image, ImageOps
import io
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = FastAPI()

@app.get("/")
def index():
    return {"status": "Up and running."}



# --------------------------- CLASSIFICATION -----------------------------------
app.state.modelclass = load_model("models/CNN_Breast_Cancer2.keras",
                                compile=False)
app.state.preprocessor = Preprocessor()

@app.post('/classification')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()
    print(f'Image type after img.read: {type(contents)}')

    # try exept?
    contents = Image.open(io.BytesIO(contents))
    print(f'Image type after opening with io.Bytes: {type(contents)}')
    contents = img_to_array(contents)
    print(f'Image type after img_to_array conversion: {type(contents)}')
    print(f'Image shape after img_to_array conversion: {contents.shape}')

    # PREPROCESS image for prediction
    preprocessor = app.state.preprocessor
    contents = preprocessor.preprocess_image(contents)
    print(f'Image type after preprocessing: {type(contents)}')
    print(f'Image shape after preprocessing: {contents.shape}')
    contents = contents.reshape((-1, 224, 224, 3))

    # Prediction
    model = app.state.modelclass
    res = model.predict(contents)
    prob = float(res)*100
    return  {"Probability of Malignant Breast Cancer": f"{round(prob,2)}%"}



# --------------------------- SEGMENTATION ------------------------------------
app.state.modelseg = load_model("models/unet_seg_model_11_6_2025_2.keras",
                                compile=False)
def plot_overlay(image, mask, alpha=0.5, mask_color='Reds'):
    """
    image: numpy array, shape (H, W, 3)
    mask: numpy array, shape (H, W) or (H, W, 1)
    alpha: transparency of the mask overlay
    mask_color: colormap for the mask (e.g., 'Reds', 'Greens', 'jet')
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(mask, cmap=mask_color, alpha=alpha)
    ax.axis('off')
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    return StreamingResponse(buf, media_type='image/png')

@app.post('/segmentation')
async def receive_image(img: UploadFile=File(...)):
    ### Receiving and decoding the image
    contents = await img.read()
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    pil_image = pil_image.resize((224, 224))
    if app.state.preprocessor.is_black_on_white(img_to_array(pil_image)):
                pil_image = ImageOps.invert(pil_image)
    #import ipdb; ipdb.set_trace()
    # Step 2: Preprocess image
    img_array = img_to_array(pil_image) / 255.0  # shape: (224, 224, 3)
    img_for_model = img_array.reshape((-1, 224, 224, 3))

    # Prediction
    #import ipdb; ipdb.set_trace()
    model = app.state.modelseg
    res = model.predict(img_for_model)
    res_binary = (res > 0.5).astype(np.uint8).squeeze()
    return plot_overlay(img_array, res_binary)
