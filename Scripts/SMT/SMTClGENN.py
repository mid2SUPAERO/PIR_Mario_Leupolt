import numpy as np
import matplotlib.pyplot as plt
from smt.surrogate_models.genn import GENN, load_smt_data
import csv
from sklearn.model_selection import train_test_split 
import pickle
# SMT model for the drag coefficient
# all file links must be changed
# required: pip install SMT
# https://github.com/SMTorg/SMT

#getting datasets
def getData():
    # gets the input vector and the outputs
    # data from the Database:
    # Bouhlel, M. A., He, S., and Martins, J. R. R. A., “mSANN Model Benchmarks,” Mendeley Data, 2019. https://doi.org/10.17632/ngpd634smf.1
    # dataset different from Keras one: some derivatives were missing
    with open('C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/SMT/dataSMTCl.csv', 'r') as file:
        reader = csv.reader(file, delimiter=";")
        values = np.array(list(reader), dtype = np.float32)
        dim_values = values.shape
        x = values[:,:dim_values[1]-1]
        y = values[:,-1]
    #gets the gradient information
    # data from the Database:
    # Bouhlel, M. A., He, S., and Martins, J. R. R. A., “mSANN Model Benchmarks,” Mendeley Data, 2019. https://doi.org/10.17632/ngpd634smf.1
    with open ("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/SMT/DataDySMTCl.csv", "r") as file:
        reader = csv.reader(file, delimiter=";")
        dy = np.array(list(reader), dtype = np.float32)
        return x, y, dy             

x, y, dy = getData()
# splitting the dataset
x_train, x_test, y_train, y_test, dy_train, dy_test = train_test_split(x, y, dy, train_size= 0.8)
# building and training the GENN
genn = GENN()
genn.options["alpha"] = 0.001  # learning rate that controls optimizer step size
genn.options[
    "lambd"
] = 0.1  # lambd = 0. = no regularization, lambd > 0 = regularization
genn.options[
    "gamma"
] = 1.0  # gamma = 0. = no grad-enhancement, gamma > 0 = grad-enhancement
genn.options["deep"] = 2  # number of hidden layers
genn.options["wide"] = 6  # number of nodes per hidden layer
genn.options[
    "mini_batch_size"
] = 256  # used to divide data into training batches (use for large data sets)
genn.options["num_epochs"] = 25  # number of passes through data
genn.options[
    "num_iterations"
] = 10  # number of optimizer iterations per mini-batch
genn.options["is_print"] = True  # print output (or not)
load_smt_data(
    genn, x_train, y_train, dy_train
)  # convenience function to read in data that is in SMT format
genn.train()

# saving the SMT model
filename = "Subsonic/models/ClSMT.pkl"
with open(filename, "wb") as f:
    pickle.dump(genn, f)

genn.plot_training_history()  # non-API function to plot training history (to check convergence)
genn.goodness_of_fit(
    x_test, y_test, dy_test
)  # non-API function to check accuracy of regression
y_pred = genn.predict_values(
    x_test
)  # API function to predict values at new (unseen) points

# Plot
fig, ax = plt.subplots()
ax.plot(x_test, y_pred)
ax.plot(x_test, y_test, "k--")
ax.set(xlabel="x", ylabel="y", title="GENN")
ax.legend(["Predicted", "True", "Test", "Train"])
plt.show()