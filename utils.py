import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def load_data(explore=True):
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    return x_train, y_train, x_test, y_test

def preprocess_data(x_train, y_train, x_test, y_test, val_size=10000):
    """
    Preprocesses the input data and creates train, validation, and test sets.
    Resulting data is in the correct shape to be fed into a Dense layer and normalized in the range [0, 1].

    Parameters:
    x_train (np.ndarray): Training image data, of shape (num_train, 28, 28).
    y_train (np.ndarray): Training labels, of shape (num_train,).
    x_test (np.ndarray): Test image data, of shape (num_test, 28, 28).
    y_test (np.ndarray): Test labels, of shape (num_test,).
    val_size (int): The size of the validation set, default=10000.

    Returns:
    np.ndarray: preprocessed training data
    np.ndarray: preprocessed training data
    np.ndarray: preprocessed validation data
    np.ndarray: preprocessed validation data
    np.ndarray: preprocessed test data
    np.ndarray: preprocessed test data

    """

    # YOUR CODE HERE
    
    # Normalize the data to be within the range [0,1]
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Slice the training data into train and validation sets,
    # where the validation set is the same size as the test set
    x_val = x_train[-val_size:]
    y_val = y_train[-val_size:]
    x_train = x_train[:-val_size]
    y_train = y_train[:-val_size]

    #reshape the data
    x_train = x_train.reshape((x_train.shape[0], -1))
    x_val = x_val.reshape((x_val.shape[0], -1))
    x_test = x_test.reshape((x_test.shape[0], -1))

    return x_train, y_train, x_val, y_val, x_test, y_test




def explore_data(x_train, y_train, x_test, y_test):

    # define the class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # plot the distribution of classes in the training, validation, and test sets
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # plot the distribution of classes in the training set
    train_class_counts = np.bincount(y_train)
    ax[0].bar(range(10), train_class_counts)
    ax[0].set_xticks(range(10))
    ax[0].set_xticklabels(class_names, rotation=45)
    ax[0].set_title('Training set')

    # plot the distribution of classes in the test set
    test_class_counts = np.bincount(y_test)
    ax[1].bar(range(10), test_class_counts)
    ax[1].set_xticks(range(10))
    ax[1].set_xticklabels(class_names, rotation=45)
    ax[1].set_title('Test set')

    plt.show()

    print(" ")  # add space between figures

    # plot a sample of the images
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show()


def build_model():
    """
    Build a Keras sequential model using Dense layers, and compile it with an optimizer and a sparse_categorical_crossentropy loss.
    The model should have two layers, and the last layer should use a softmax activation and should have the correct output dimension. 
    Compile the model to use the adam optimizer and a sparse_categorical_crossentropy loss. Accuracy should be monitored during training.

    Returns:
    model (Sequential)

    """
    # YOUR CODE HERE
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),  # `input_shape` for the first layer
        keras.layers.Dense(10, activation='softmax')  # Assuming 10 classes
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, x_val, y_val, epochs=5, batch_size=32):
    # train the model using train and validation sets
    history = model.fit(x_train, y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        validation_data=(x_val, y_val))
    return history


def plot_loss(history):
    # plot the training and validation loss side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # plot the training and validation loss
    ax[0].plot(history.history['loss'], label='train')
    ax[0].plot(history.history['val_loss'], label='val')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # plot the training and validation accuracy
    ax[1].plot(history.history['accuracy'], label='train')
    ax[1].plot(history.history['val_accuracy'], label='val')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()


def test_model(model, x_test, y_test):
    """
    Test the accuracy of a trained model on a given test set.

    Parameters:
    model (keras.engine.sequential.Sequential): A trained Keras sequential model.
    x_test (np.ndarray): The input test data.
    y_test (np.ndarray): The ground truth test labels.

    Returns:
    test_acc (float): The test accuracy.
    y_pred (np.ndarray): The predicted labels of the test set.

    """
    # YOUR CODE HERE

    # Make predictions using the model
    predictions = model.predict(x_test)

    # Convert the predicted probabilities to class labels
    y_pred = np.argmax(predictions, axis=1)

    # Calculate the accuracy by comparing the predicted labels with the ground truth labels and computing the mean
    test_acc = np.mean(y_pred == y_test)
    # Print the result
    return test_acc, y_pred




