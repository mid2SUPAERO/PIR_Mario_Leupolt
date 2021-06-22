import numpy as np
import tensorflow as tf
import csv
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Constant
from tensorflow.keras.initializers import RandomUniform
from sklearn import linear_model
import matplotlib.pyplot as plt
from tensorflow.keras.models import save_model
from sklearn.metrics import r2_score
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# final model for the lift coefficient
# all the file links must be changed

# gives a plot for the prediction quality
def plot_prediction(y_test, y_pred):
    a =  plt.axes(aspect='equal')
    lims = [0, 1.5]
    y_test = y_test[:, 0]
    y_pred = y_pred[:]
    #linear regression through the predicted values and the true values
    regr = linear_model.LinearRegression()
    y_pred_fit = np.reshape(y_pred, newshape=(-1,1))
    y_test_fit = np.reshape(y_test, newshape=(-1,1))
    regr.fit(y_test_fit, y_pred_fit)
    #calculating the r2-score
    r2 = r2_score(y_test_fit, y_pred_fit)
    #coefficients for the linear regression
    a = regr.intercept_
    b = regr.coef_[0]
    y_pred_lin = []
    for i in lims:
        y_pred_lin.append(b*i + a)
    #creating the actual plot
    plt.xlim(lims)
    plt.ylim(lims)
    plt.scatter(y_test, y_pred, color = "lightskyblue")
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.plot(lims, lims, label = "true_val", color = "blue")
    plt.grid(True)
    plt.plot(lims, y_pred_lin, label = "pred_val", linestyle = "--", color = "orange")
    plt.legend()
    plt.show()
    print(f"R2-Value: {r2}")

# creates a plot for the loss function -> development of the loss over time
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 0.1])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MAE]')
    plt.legend()
    plt.grid(True)
    plt.show()

# loads the dataset
def getDataset():
    # data from the Database:
    # Bouhlel, M. A., He, S., and Martins, J. R. R. A., “mSANN Model Benchmarks,” Mendeley Data, 2019. https://doi.org/10.17632/ngpd634smf.1
    # all the Subsonic moment coefficient data was put into one csv file
    with open('C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/dataKerasCl.csv', 'r') as file:
        reader = csv.reader(file, delimiter=";")
        values = np.array(list(reader), dtype = np.float32)
        dim_values = values.shape
        x = values[:,:dim_values[1]-1]
        y = values[:,-1]
        return x, y

# builds the model
def build_model(num_features: int, num_targets: int) -> Sequential:
    #randomly initilaizes the weight and bias in a predetermined range
    init_w = RandomUniform(minval=0.0, maxval=0.1)
    init_b = Constant(value=0.0)

    model = Sequential()
    model.add(Dense(units=100, kernel_initializer=init_w, bias_initializer=init_b, input_shape = (num_features,)))
    model.add(Activation("tanh"))
    model.add(Dense(units=120, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("sigmoid"))
    model.add(Dense(units=140, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("selu"))
    model.add(Dense(units=160, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(Activation("selu"))
    model.add(Dense(units=180, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(Dense(units=180, kernel_initializer=init_w, bias_initializer=init_b))
    model.add(LeakyReLU(alpha = 0.1))
    model.add(Dense(units=num_targets, kernel_initializer=init_w, bias_initializer=init_b))
    model.summary()

    return model


if __name__ == "__main__":
    #obtaining of the input and the output from the dataset
    x, y = getDataset()
    y = np.reshape(y, newshape=(-1,1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    num_features = x_train.shape[1]
    num_targets = y_train.shape[1]

    # building the model
    model = build_model(num_features, num_targets)

    #setting the optimizer options
    opt = Adam(learning_rate = 0.0005)

    # compile the model
    model.compile( 
        loss = "mae",
        optimizer = opt,
        metrics = [RootMeanSquaredError()]
    )

    #train the model
    history = model.fit(
        x = x_train,
        y = y_train,
        epochs = 500,
        batch_size = 256,
        verbose = 1,
        validation_data = (x_test, y_test)
        )

    #evaluate the model
    scores = model.evaluate(
        x = x_test,
        y = y_test,
        verbose = 0
    )
    
    # issue a prediction for the plot of quality of prediction
    y_pred = model.predict(x_test).flatten()
    plot_prediction(y_test, y_pred)
    plot_loss(history)
    print(scores)
    #save model
    model.save("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/cl.h5")
