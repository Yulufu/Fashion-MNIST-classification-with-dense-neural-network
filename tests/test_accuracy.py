from utils import preprocess_data
import numpy as np

def test_accuracy():

    y_pred = np.load('y_pred.npy')
    y_test = np.load('y_test.npy')

    # Check accuracy is above 88%
    assert np.mean(y_pred == y_test) >= 0.88