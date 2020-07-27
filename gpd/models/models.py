#!/usr/bin/env python3

""" Load models. """

import os
import tensorflow as tf

from keras.models import model_from_json
from keras.engine.training import Model


MODEL_DIR = os.path.dirname(__file__)


def load_model(filename: str, weights_filename: str) -> Model:
    """ 
    Load a model from json file and weights hdf5 file. 
    """
    with open(filename, "r") as f:
        model = f.read()
    model = model_from_json(model, custom_objects={'tf': tf})

    model.load_weights(weights_filename)

    return model


MODELS = {
    "Ross_original": load_model(
        filename=f"{MODEL_DIR}/model_pol.json",
        weights_filename=f"{MODEL_DIR}/model_pol_best.hdf5")
}

if __name__ == "__main__":
    import doctest

    doctest.main()
