import os
import glob
#from colorama import Fore, Style

import tensorflow as tf
from keras import Model
import numpy as np

import matplotlib.pyplot as plt

from segmentation.params import *



#Losses
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def bce_dice_loss(y_true, y_pred):
    bce = BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

#Load model

def load_model():
    """
    Load model from path
    """
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models/segmentation")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        print(f"\n❌ No model found")
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print("\nLoading latest model from disk...")

    latest_model = tf.keras.models.load_model(most_recent_model_path_on_disk, custom_objects={ 'loss': bce_dice_loss() }) # type: ignore

    print("✅ Model loaded")

    return latest_model

def predict(
        model: Model,
        X: np.ndarray,
    ):
    """
    Predict with model
    """

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    prediction = model.predict(X)

    return prediction
