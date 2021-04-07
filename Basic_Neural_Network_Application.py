import tensorflow as tf
import numpy as np
from tensorflow import keras

#%% Define and Compile the Neural Network 1
#we will create the simplest possible neural network
#It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value.
model = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])

#%% Define and Compile the Neural Network 2 
#We give a loss function and optimizer function to model. Loss function measures the result as
# how well or how badly it did. And that optimizer function make another guess for result.
#the different and appropriate loss and optimizer functions use for different scenarios.
#MEAN SQUARED ERROR for loss function
#STOCHASTIC GRADIENT DESCENT for optimizer function.
model.compile(optimizer='sgd', loss='mean_squared_error')

#%% Providing the Data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

#%%Training the Neural Network
#Model will guess the relationship between xs and ys for number of epochs.
#For training we use the model.fit function.
model.fit(xs, ys, epochs=500)

#%%Predict
# we can see the relationship between these two variable with use of model.predict function.
print(model.predict([15.0]))

