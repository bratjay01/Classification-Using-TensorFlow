import tensorflow
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPool2D, BatchNormalization, GlobalAvgPool2D

from deeplearningmodels import functional_model, MyCustomModel
from my_utils import display_some_examples

# Three approaches to build neural network/deep learning models
#tensorflow.keras.Sequential: Just stack diff layers together
seq_model=tensorflow.keras.Sequential(
    [
      Input(shape=(28,28,1)),     # 1 is one channel # This line is basically taking in an image 
      Conv2D(32, (3,3), activation='relu'),  #32 is how many filters each filter is 3x3 of size of activation with rectified linear unit. How many parameters we choose is basically experimentation
      Conv2D(64, (3,3), activation='relu'),
      MaxPool2D(), #takes output from convolution
      BatchNormalization(),
    

      Conv2D(128, (3,3), activation='relu'),# so many researchers follow these numbers they double the no of filters
      MaxPool2D(),
      BatchNormalization(),

      GlobalAvgPool2D(),   #Takes normalized output from normalization and computes avg acc to some axes and we'll get some values
      Dense(64, activation='relu'),  # A vector containing values
      Dense(10, activation='softmax') #10 values with output of the neural network # We want this whole network to tell us which value it is from 0-9,10 probabilities. Highest probability is the output predicted


    ]
)        



if __name__=='__main__':  #always good to add this to separate behaviour of importing scripts & running scripts
    
    (x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

    print("x_train.shape = ", x_train.shape)
    print("y_train.shape = ", y_train.shape)
    print("x_test.shape = ", x_test.shape)
    print("y_test.shape = ", y_test.shape)

    if False:
       display_some_examples(x_train, y_train)

    x_train = x_train.astype('float32') / 255  #255 is white and zero is black# we're taking in unsigned int 8 bit type we need to convert to float32 type to show exact value 
    x_test = x_test.astype('float32') / 255

    x_train = np.expand_dims(x_train, axis=-1) #we need to expand dimensions of the array to include that channel 1 (28,28,1)
    x_test = np.expand_dims(x_test, axis=-1) #we can do 3 or -1 which means expand in endof array
    
    #if you want to use categoricalcrossentropy this is how you'll convert to one hot encoding
    
    y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
    y_test = tensorflow.keras.utils.to_categorical(y_test, 10)
    
    #model = functional_model()
    model = MyCustomModel()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')  #optimizer represents the algorithm we're gonna use to optimize the cost function by optimizing we're trying to find the global min of our cost function
    #accuracy predicts how many values we're predicting correctly out of all the datasets if a dataset of 100images prediction:88 correct then it is 88%
   
    #model training

    model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.2) #epoch represents the number of times your model is gonna see all of your datasets
    # train ,validation, test
# Training Dataset: The sample of data used to fit the model.
# Validation Dataset: The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. 
# The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.
# Test Dataset: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.

    #evaluation test set

    model.evaluate(x_test, y_test, batch_size=64)

