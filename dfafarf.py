import tensorflow as tf

# numpy is a math library
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
# numpy is a math library
import numpy as np
SAMPLES = 1000
SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)
x_values = np.random.uniform(low=0, high=2*math.pi, size=SAMPLES)
np.random.shuffle(x_values)
y_values = np.sin(x_values)
y_values +=0.1*np.random.randn(*y_values.shape)
plt.plot(x_values, y_values, 'b.')
plt.show()

TRAIN_SPLIT = int(0.6*SAMPLES)
TEST_SPLIT = int(0.2*SAMPLES + TRAIN_SPLIT)

x_train, x_validate, x_test = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
y_train, y_validate, y_test = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

#double check that the splits add up correct;u

assert(x_train.size + x_validate.size + x_test.size) == SAMPLES

#plot the data in each partition in different colors
plt.plot(x_train, y_train,'b.', label="train")
plt.plot(x_validate, y_validate, 'y.', label="validate")
plt.plot(x_test, y_test, 'r.' , label="test")
plt.legend
plt.show()
# use keras to creat a simple model architectture
from tensorflow.keras import layers
model_1 = tf.keras.Sequential()

#first layer takes a scalar input and feeds it through to neurons, the neurons decide whether to active based on the relu activation function.
model_1.add(layers.Dense(16, activation='relu', input_shape=(1,))) #inputshape 1dimension

#final layer is a single neuron, since the output is a single value
#the activation numbers from the first layer will be fed as inputs to our second layer, whcih is defined in the following :
model_1.add(layers.Dense(1))

#compile the model using a standard optimizer and loss function for regression
model_1.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#print a summary of model's architecture
model_1.summary()
history_1 = model_1.fit(x_train, y_train, epochs=1000, batch_size=16, validation_data=(x_validate, y_validate))

#graphing the history
loss = history_1.history['loss']
val_loss = history_1.history['val_loss']
epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'g.', label='training loss')
plt.plot(epochs, val_loss,'b.',label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# exclude the first few epochs so that the graph is easier to read

SKIP = 100

plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='training loss')
plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='validation loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#DRAW A GRAPH OF MEAN ABSOLUTE ERROR, WHICH IS ANOTHER WAY OF MEASURRING THE AMOUNT OF ERROR IN THE PREDICTION

mae = history_1.history['mae']
val_mae = history_1.history['val_mae']

plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='training MAE')
plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='validation mae')
plt.title('training and validation mean absolute error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

# now using the model to make predictions from our validation data
predictions = model_1.predict(x_train)

#plot the prediction along with the test data
plt.clf()
plt.title('Training data predicted vs actual values')
plt.plot(x_test, y_test, 'b.', label='Actual')
plt.plot(x_train, predictions, 'r.', label='Predicted')
plt.legend()
plt.show()

#improing the model. add another layer of neuron.

model_2 = tf.keras.Sequential()

model_2.add(layers.Dense(16, activation='relu', input_shape=(1,)))
model_2.add(layers.Dense(16, activation='relu'))

model_2.add(layers.Dense(1))

#compile the model using a standard optimizer and loss function for regression
model_2.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
#print a summary of model's architecture
model_2.summary()

history_2 = model_2.fit(x_train, y_train, epochs=600, batch_size=16,
                        validation_data=(x_validate, y_validate))

#graphing, evaluate the performence of the model
#graph of loss
loss = history_2.history['loss']
val_loss = history_2.history['val_loss']
epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'g.', label='training loss')
plt.plot(epochs, val_loss,'b.',label='validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

SKIP = 100

plt.clf() #clear the current figure
plt.plot(epochs[SKIP:], loss[SKIP:], 'g.', label='training loss')
plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='validation loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf
mae = history_2.history['mae']
val_mae = history_2.history['val_mae']

plt.plot(epochs[SKIP:], mae[SKIP:], 'g.', label='training MAE')
plt.plot(epochs[SKIP:], val_mae[SKIP:], 'b.', label='validation mae')
plt.title('training and validation mean absolute error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

#testing.
#calculate and print the loss on the test dataset
loss = model_2.evaluate(x_test, y_test)

#make the predictions based on the testing data
predictions = model_2.predict(x_test)

#graph the predictions aginst the actual values

plt.clf()
plt.title('comparison of predictions and actual values')
plt.plot(x_test, y_test, 'b.', label='actual')
plt.plot(x_test, predictions, 'r.', label='Predicted')
plt.legend()
plt.show()

#converting the model for tensorflow lite
#convert the model to tensorlow lite format with out quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
tflite_model = converter.convert()

#save the model to disk
open("sine_model.tflite", "wb").write(tflite_model)

#convert the model to the tensorflow lite format with quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
#indicate default optimization
converter.optimizations = [tf.lite.Optimize.DEFAULT]
#definea generator function that provides the test data's x values as a representativa_dataset, and tell the converter to use it
def representative_dataset_generator():
  for value in x_test:
    yield [np.array(value, dtype=np.float32, ndmin=2)]
converter.representative_dataset = representative_dataset_generator
#converter the model
tflite_model = converter.convert()

#save the model to disk
open("sine_model_quantized.tflite", "wb").write(tflite_model)

# Instantiate an interpreter for each model
sine_model = tf.lite.Interpreter('sine_model.tflite')
sine_model_quantized = tf.lite.Interpreter('sine_model_quantized.tflite')
# Allocate memory for each model
sine_model.allocate_tensors()
sine_model_quantized.allocate_tensors()
# Get indexes of the input and output tensors
sine_model_input_index = sine_model.get_input_details()[0]["index"]
sine_model_output_index = sine_model.get_output_details()[0]["index"]
sine_model_quantized_input_index = sine_model_quantized.get_input_details()[0]["index"]
sine_model_quantized_output_index = sine_model_quantized.get_output_details()[0]["index"]

# Create arrays to store the results
sine_model_predictions = []
sine_model_quantized_predictions = []
# Run each model's interpreter for each value and store the results in arrays
for x_value in x_test:
  # Create a 2D tensor wrapping the current x value
  x_value_tensor = tf.convert_to_tensor([[x_value]], dtype=np.float32)
   # Write the value to the input tensor
  sine_model.set_tensor(sine_model_input_index, x_value_tensor)
  # Run inference
  sine_model.invoke()
  # Read the prediction from the output tensor
  sine_model_predictions.append(sine_model.get_tensor(sine_model_output_index)[0])
# Do the same for the quantized model
  sine_model_quantized.set_tensor(sine_model_quantized_input_index, x_value_tensor)

# Run inference for the quantized model
  sine_model_quantized.invoke()

# Append the predictions for the quantized model
  sine_model_quantized_predictions.append(sine_model_quantized.get_tensor(sine_model_quantized_output_index)[0])



  # See how they line up with the data
plt.clf()
plt.title('Comparison of various models against actual values')
plt.plot(x_test, y_test, 'bo', label='Actual')
plt.plot(x_test, predictions, 'ro', label='Original predictions')
plt.plot(x_test, sine_model_predictions, 'bx', label='Lite predictions')
plt.plot(x_test, sine_model_quantized_predictions, 'gx', label='Lite quantized predictions')
plt.legend()
plt.show()

#campare the size of 2 model


import os
basic_model_size = os.path.getsize("sine_model.tflite")
print("Basic model is %d bytes" % basic_model_size)
quantized_model_size = os.path.getsize("sine_model_quantized.tflite")
print("Quantized model is %d bytes" % quantized_model_size)
difference = basic_model_size - quantized_model_size
print("Difference is %d bytes" % difference)

#install xxd if it is not available

!apt-get -qq install xxd
# Save the file as a C source file
!xxd -i sine_model_quantized.tflite > sine_model_quantized.cc
# Print the source file
!cat sine_model_quantized.cc

