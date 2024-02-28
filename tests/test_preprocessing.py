from utils import preprocess_data
import numpy as np

def test_preprocessing():

    train_size = 60
    val_size = 10
    im_size = 28
    im_max = 255

    x_train = np.random.randint(im_max, size=(train_size, im_size, im_size))
    y_train = np.random.randint(im_max, size=(train_size,))

    x_test = np.random.randint(im_max, size=(val_size, im_size, im_size))
    y_test = np.random.randint(im_max, size=(val_size,))

    x_train, y_train, x_val, y_val, x_test, y_test = preprocess_data(x_train, 
                                                                 y_train, 
                                                                 x_test, 
                                                                 y_test,
                                                                 val_size=val_size)

    # Check shapes
    assert x_train.shape == (train_size-val_size, im_size*im_size)
    assert y_train.shape == (train_size-val_size,)
    assert x_test.shape == (val_size, im_size*im_size)
    assert y_test.shape == (val_size,)

    # Check ranges
    assert x_train.min() >= 0.0
    assert x_train.max() <= 1.0
    assert x_test.min() >= 0.0
    assert x_test.max() <= 1.0
    assert x_test.min() >= 0.0
    assert x_test.max() <= 1.0