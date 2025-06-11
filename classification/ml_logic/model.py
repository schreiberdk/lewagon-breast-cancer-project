import os
import glob
#from colorama import Fore, Style

import tensorflow as tf
from keras import Model
import numpy as np

from classification.params import *

def load_model():
    """
    Load model from path
    """
    local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    local_model_paths = glob.glob(f"{local_model_directory}/*")

    if not local_model_paths:
        print(f"\n❌ No model found")
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print("\nLoading latest model from disk...")

    latest_model = tf.keras.models.load_model(most_recent_model_path_on_disk)

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
