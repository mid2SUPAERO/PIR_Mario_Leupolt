import tensorflow as tf
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from ModeShapeReconstruct import reconstruct_airfoil
from smt.surrogate_models import genn
import pickle
# to get predictions first the models must be trained and saved
# all links must be changed to your file path
# airfoil_modeshapes: computed mode_shapes of random airfol geometry with parameterise_airfoil
# Ma: desired Mach number for evaluation in range [0.3,0.6]


# gives a scalar prediction using the models trained in Keras
# alpha scalar in range [-2, 6]
def scalarPredictionsKeras(airfoil_modeshapes, Ma, alpha):
    #loading of the models
    modelcd = load_model('C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/cd.h5', compile = True)
    modelcl = load_model("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/cl.h5", compile = True)
    modelcm = load_model("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/cm2.h5", compile = True)

    # input array in neural network is created out of airfoil mode shapes, Mach number and alpha
    input_array = np.zeros(shape=(1,16))
    input_array[0,:14] = airfoil_modeshapes
    input_array[0,14] = Ma
    input_array[0,-1] = alpha

    # predictions are made
    cd_pred = modelcd.predict(input_array).flatten()
    cl_pred = modelcl.predict(input_array).flatten()
    cm_pred = modelcm.predict(input_array).flatten()

    return cd_pred, cl_pred, cm_pred, Ma, alpha

# gives a scalar prediction using the models trained in SMT
# alpha scalar in range [-2, 6]
def scalarPredictionsSMT(airfoil_modeshapes, Ma, alpha):
    #loading of the models (not yet a direct function in SMT, that is why the way over pickle)
    modelcd = None
    modelcl = None
    modelcm = None
    with open("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/CdSMT.pkl", "rb") as file:
        modelcd = pickle.load(file)
    with open("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/ClSMT.pkl", "rb") as file:
        modelcl = pickle.load(file)
    with open("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/CmSMT.pkl", "rb") as file:
        modelcm = pickle.load(file)

    # input array in neural network is created out of airfoil mode shapes, Mach number and alpha
    input_array = np.zeros(shape=(1,16))
    input_array[0,:14] = airfoil_modeshapes
    input_array[0,14] = Ma
    input_array[0,-1] = alpha

    # predictions are made
    cd_pred = modelcd.predict_values(input_array)
    cl_pred = modelcl.predict_values(input_array)
    cm_pred = modelcm.predict_values(input_array)

    return cd_pred, cl_pred, cm_pred, Ma, alpha

# gives an array of predicted aerodynamic coefficients
# plot: bool if graph should be created or not -> if None; just returns the arrays for the aerodynamic coefficients
# airfoil_name: string 
# alpha scalar in range [-2, 6]
def graphPredictionsKeras(airfoil_modeshapes, airfoil_name, Ma, plot: bool):
    #loading of the models    
    modelcd = load_model('C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/cd.h5', compile = True)
    modelcl = load_model("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/cl.h5", compile = True)
    modelcm = load_model("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/cm.h5", compile = True)

    #input arrays are created -> alpha is linearily distributed over the range of -2 to 6 degrees while Ma is kept constant
    input_array = np.zeros(shape = (1,15))
    input_array[0,:14] = airfoil_modeshapes
    input_array[0,-1] = Ma
    new_input_array = np.zeros(shape = (1,15))
    new_input_array[0,:14] = airfoil_modeshapes
    new_input_array[0,-1] = Ma
    for i in range(0,49):
        new_input_array = np.concatenate((new_input_array, input_array), axis = 0)
    alpha = np.zeros(shape = (50,1))
    for i in range(0,50):
        alpha[i,0]= -2 + 0.16* i
    input_array = np.concatenate((new_input_array, alpha), axis = 1)

    # predictions are made
    cd_pred = modelcd.predict(input_array).flatten()
    cl_pred = modelcl.predict(input_array).flatten()
    cm_pred = modelcm.predict(input_array).flatten()

    # graphs for the single aerodynamic coefficients are computed -> through bool: plot it is to decide if graphs are computed or not
    if plot == True:
        x, y_comp = reconstruct_airfoil(airfoil_modeshapes)

        plt.plot(x, y_comp)
        plt.axis([-0.1,1.2,-0.6,0.6])
        plt.grid(True)
        plt.title(airfoil_name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"Airfoil_{airfoil_name}")
        plt.close()

        plt.plot(alpha, cl_pred)
        plt.grid(True)
        plt.title("Lift coefficient Keras")
        plt.xlabel("Alpha")
        plt.ylabel("Cl")
        plt.savefig(f"Keras_Cl_{airfoil_name}")
        plt.close()

        plt.plot(alpha, cd_pred)
        plt.grid(True)
        plt.title("Drag coefficient Keras")
        plt.xlabel("Alpha")
        plt.ylabel("Cd")
        plt.savefig(f"Keras_Cd_{airfoil_name}")
        plt.close()

        plt.plot(alpha, cm_pred)
        plt.grid(True)
        plt.title("Moment coefficient Keras")
        plt.xlabel("Alpha")
        plt.ylabel("Cm")
        plt.savefig(f"Keras_Cm_{airfoil_name}")
        plt.close()    

    return cd_pred, cl_pred, cm_pred, alpha, airfoil_name

# gives an array of predicted aerodynamic coefficients
# plot: bool if graph should be created or not -> if None; just returns the arrays for the aerodynamic coefficients
# airfoil_name: string 
# alpha scalar in range [-2, 6]
def graphPredictionsSMT(airfoil_modeshapes, airfoil_name, Ma, plot: bool):
    #loading of the models   
    modelcd = None
    modelcl = None
    modelcm = None
    with open("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/CdSMT.pkl", "rb") as file:
        modelcd = pickle.load(file)
    with open("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/ClSMT.pkl", "rb") as file:
        modelcl = pickle.load(file)
    with open("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/Subsonic/models/CmSMT.pkl", "rb") as file:
        modelcm = pickle.load(file)

    #input arrays are created -> alpha is linearily distributed over the range of -2 to 6 degrees while Ma is kept constant
    input_array = np.zeros(shape = (1,15))
    input_array[0,:14] = airfoil_modeshapes
    input_array[0,-1] = Ma
    new_input_array = np.zeros(shape = (1,15))
    new_input_array[0,:14] = airfoil_modeshapes
    new_input_array[0,-1] = Ma
    for i in range(0,49):
        new_input_array = np.concatenate((new_input_array, input_array), axis = 0)
    alpha = np.zeros(shape = (50,1))
    for i in range(0,50):
        alpha[i,0]= -2 + 0.16* i
    input_array = np.concatenate((new_input_array, alpha), axis = 1)

    # predictions are made
    cd_pred = modelcd.predict_values(input_array)
    cl_pred = modelcl.predict_values(input_array)
    cm_pred = modelcm.predict_values(input_array)

    # graphs for the single aerodynamic coefficients are computed -> through bool: plot it is to decide if graphs are computed or not
    if plot == True:
        x, y_comp = reconstruct_airfoil(airfoil_modeshapes)
        plt.plot(x, y_comp)
        plt.axis([-0.1,1.2,-0.6,0.6])
        plt.grid(True)
        plt.title(airfoil_name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(f"Airfoil_{airfoil_name}")
        plt.close()

        plt.plot(alpha, cl_pred)
        plt.grid(True)
        plt.title("Lift coefficient SMT")
        plt.xlabel("Alpha")
        plt.ylabel("Cl")
        plt.savefig(f"SMT_Cl_{airfoil_name}")
        plt.close()

        plt.plot(alpha, cd_pred)
        plt.grid(True)
        plt.title("Drag coefficient SMT")
        plt.xlabel("Alpha")
        plt.ylabel("Cd")
        plt.savefig(f"SMT_Cd_{airfoil_name}")
        plt.close()

        plt.plot(alpha, cm_pred)
        plt.grid(True)
        plt.title("Moment coefficient SMT")
        plt.xlabel("Alpha")
        plt.ylabel("Cm")
        plt.savefig(f"SMT_Cm_{airfoil_name}")
        plt.close()    
    # array for the aerodynamic coeffs and alpha and the airfoil name
    return cd_pred, cl_pred, cm_pred, alpha, airfoil_name

# example
from ModeShapeReconstruct import *
NACA4412 = np.loadtxt("C:/Users/Mario/OneDrive/Universität/Semester 8 21 SoSe/PIR/Material/NACA4412.txt")
mycamber,mythickr, scinewairfoil, airfoil_modeshapes = parameterise_airfoil(NACA4412)
print(airfoil_modeshapes.shape)
cd_pred, cl_pred, cm_pred, Ma, alpha = scalarPredictionsKeras(airfoil_modeshapes, 0.5, 0)
print("Keras", cd_pred)
cd_pred, cl_pred, cm_pred, Ma, alpha = scalarPredictionsSMT(airfoil_modeshapes, 0.5, 0)
print("SMT", cd_pred)
graphPredictionsKeras(airfoil_modeshapes, "NACA4412", Ma, True)
graphPredictionsSMT(airfoil_modeshapes, "NACA4412", Ma, True)

