The calculation of the aerodynamic coefficients of an airfoil with established methods is a time consuming and computationally expensive process. 
This work takes a deep learning approach to determine the aerodynamic coefficients of a given airfoil much faster. It uses an already existing 
data base and tries three different platforms (Keras and SMT package in python and Monolith AI) to predict the coefficients as accurately as possible. 
The newly obtained surrogate model can then be used to give scalar and graph predictions for certain Mach numbers and angles of attack or in 
optimization algorithms to calculate the aerodynamic coefficients in the optimization steps.
