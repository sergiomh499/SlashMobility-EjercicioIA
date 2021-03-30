from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

def NeuralNetwork_model(x_train, y_train, x_val, y_val):
    """
    NeuralNetwork_model: Implementation of the Neural Network

    args:
        x_train: input dataset for training
        y_train: real solution dataset for training
        x_val: input dataset for validation
        y_val: real solution dataset for validation
    out:
        model: trained NN model
        history: information about training
    """
    # define and fit the final model
    model = Sequential()
    model.add(Dense(units=6, input_dim=5, activation='relu'))
    model.add(Dense(units=12, activation='relu'))
    model.add(Dense(units=12, activation='relu'))
    model.add(Dense(units=6, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    # training
    history = model.fit(x_train, y_train,
                        epochs=300,
                        batch_size=200,
                        validation_data=(x_val, y_val),
                        verbose=1)
    

    return model, history

def create_data(size=1000):
    """
    create_data: Create the dataset for the Neural Network for addition operation

    args:
        size[=1000]: size of the dataset
    out:
        x: input of the NN
        y: real solution
    """
    x = np.zeros(shape=(size,5))
    y = np.zeros(shape=(size,1))
    for index in range(size):
        x_aux = np.random.randint(100, size=5)
        y_aux = sum(x_aux)
        x[index] = x_aux
        y[index] = y_aux
    return x, y

def main():
    # dataset creation
    x_train, y_train = create_data(size=2000)
    x_val, y_val = create_data(size=600)

    # creating and training neural network
    model, history = NeuralNetwork_model(x_train=x_train, y_train=y_train,
                                         x_val=x_val, y_val=y_val)

    # save the model
    model.save('NN_addition_operation.h5')

    # plotting:
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    # summarize history for loss
    plt.semilogy(history.history['loss'])
    plt.semilogy(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig("loss.png")

    # testing neural network
    x_pred = np.array([[5, 1, 1, 1, 1]])
    y_pred = model.predict(x_pred)
    print("X=%s, Predicted=%s" % (x_pred[0], y_pred[0]))


if __name__ == '__main__':
    main()
