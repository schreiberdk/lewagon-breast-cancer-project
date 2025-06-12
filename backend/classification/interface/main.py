import numpy as np

from classification.params import *
from classification.ml_logic.model import load_model, predict

def pred(X_pred: np.ndarray) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X_pred is None:
        print(f"\n❌ No image to predict on")

    model = load_model()
    assert model is not None

    y_pred = model.predict(X_pred)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")

    return y_pred

if __name__ == '__main__':
    print("")
