from utils import build_model, tf, keras
import numpy as np

def test_model():

    model = build_model()

    # Check model architecture
    assert len(model.layers) == 2
    assert type(model.layers[0]) == tf.keras.layers.Dense
    assert type(model.layers[1]) == tf.keras.layers.Dense   

    # Check model loss
    assert model.loss == 'sparse_categorical_crossentropy'
