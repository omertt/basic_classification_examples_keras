import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%% Load Dataset
mnist_dataset = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist_dataset.load_data()

#%% Values
np.set_printoptions(linewidth = 200)
plt.imshow(training_images[17])
print(training_labels[17])
print(training_images[17])

#%% Normalizing
training_images  = training_images / 255.0
test_images = test_images / 255.0

#%% Model Designing 
#Relu is a activation function and means "If X>0 return X, else return 0"  
#Softmax is a activation function too. It takes a set of value and picks the biggest one.
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#%% define optimizer and loss function than train the model.
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

#%% testing with test_images
model.evaluate(test_images, test_labels)

#%% Prediction
#It gives the probability that this item is each of the 10 classes
classifications = model.predict(test_images)

print(classifications[0])

#%% Notes
# If we add more neurons to second layer, process slow down but we get more accurate
# Number of neurons of last layer must match the number of class
# If we add one more layer to model we dont get significant impact for this case.
#Because this case have simple datas (images with gray values) But more complex data(rgb images) often need extra layers.
#If we try more epochs than five we will probably get better loss.At the same time, we might see
#the loss value stops decreasing or sometimes increasing. That is a sideeffect called overfitting.
#If you reach a desired value for loss you can stop the training with using callback.






