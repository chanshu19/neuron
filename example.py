import numpy
from neuron import *
import cv2
"""
Convolutional neural network implementation using NumPy
A tutorial that helps to get started (Building Convolutional Neural Network using NumPy from Scratch) available in these links: 
    https://www.linkedin.com/pulse/building-convolutional-neural-network-using-numpy-from-ahmed-gad
    https://towardsdatascience.com/building-convolutional-neural-network-using-numpy-from-scratch-b30aac50e50a
    https://www.kdnuggets.com/2018/04/building-convolutional-neural-network-numpy-scratch.html
It is also translated into Chinese: http://m.aliyun.com/yunqi/articles/585741
"""

train_inputs = numpy.load("dataset_inputs.npy")
train_outputs = numpy.load("dataset_outputs.npy")
print(train_inputs[0].shape)

sample_shape = train_inputs.shape[1:]
num_classes = 4

model = Sequential()

model.add(Input2D(input_shape=sample_shape))
model.add(Conv2D(num_filters=2, kernel_size=3, previous_layer=model.network_layers[0]))
model.add(ReLU(previous_layer=model.network_layers[1]))
model.add(AveragePooling2D(pool_size=2, previous_layer=model.network_layers[2], stride=2))
model.add(Conv2D(num_filters=3, kernel_size=3, previous_layer=model.network_layers[3]))
model.add(ReLU(previous_layer=model.network_layers[4]))
model.add(MaxPooling2D(pool_size=2, previous_layer=model.network_layers[5], stride=2))
model.add(Conv2D(num_filters=1, kernel_size=3, previous_layer=model.network_layers[6]))
model.add(ReLU(previous_layer=model.network_layers[7]))
model.add(AveragePooling2D(pool_size=2, previous_layer=model.network_layers[8], stride=2))
model.add(Flatten(previous_layer=model.network_layers[9]))
model.add(Dense(num_neurons=100, previous_layer=model.network_layers[10], activation_function="relu"))
model.add(Dense(num_neurons=num_classes, previous_layer=model.network_layers[11], activation_function="softmax"))
model.summary()
img = model.feed_forward(train_inputs[0])
# print(img)
# while True:
#     cv2.imshow("priview",img)
#     key = cv2.waitKey(1)
#     if key == ord('q'):
#         break
# cv2.destroyAllWindows()

#-------------------------------------------------------------------------------------------------------------------
"""
model.add(Input(input_shape=10))
model.add(Dense(num_neurons=8, previous_layer=model.network_layers[0], activation_function="sigmoid"))
model.add(Dense(num_neurons=num_classes, previous_layer=model.network_layers[1], activation_function="softmax"))
model.summary()
print(model.feed_forward(numpy.array([45,22,33,66,55,88,11,8,9,10])))
"""