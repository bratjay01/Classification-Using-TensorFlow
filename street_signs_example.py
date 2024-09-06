import os
import glob
from PIL.Image import FASTOCTREE
from sklearn.model_selection import train_test_split
import shutil
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping  #to save the best model #if you're training for higher number of epoch and the model is not improving you can stop

from my_utils import create_generators, split_data, order_test_set

from deeplearningmodels import streetsigns_model
import tensorflow as tf


if __name__=="__main__":

    if False:
       path_to_data = "D:\\German_traffic_sign\\Train"
       path_to_save_train = "D:\\German_traffic_sign\\training_data\\train"
       path_to_save_val = "D:\\German_traffic_sign\\training_data\\val"
       split_data(path_to_data, path_to_save_train=path_to_save_train, path_to_save_val=path_to_save_val)
    if False:        
       path_to_images = "D:\\German_traffic_sign\\Test"
       path_to_csv = "D:\\German_traffic_sign\\Test.csv"
       order_test_set(path_to_images, path_to_csv)

    path_to_save_train = "D:\\German_traffic_sign\\training_data\\train"
    path_to_save_val = "D:\\German_traffic_sign\\training_data\\val"
    path_to_test = "D:\\German_traffic_sign\\Test"   
    batch_size = 64
    epochs = 15

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_save_train, path_to_save_val, path_to_test)
    nbr_classes = train_generator.num_classes
    
    TRAIN=False
    TEST=True

    if TRAIN: 
        path_to_save_model = './Models'
        ckpt_saver = ModelCheckpoint(
        path_to_save_model,          #to save the model with high validation accuracy we can also have val loss and mode to min
        monitor="val_accuracy",
        mode='max',
        save_best_only=True,
        save_freq='epoch', #when to save
        verbose=1  #the model has been saved or not
      )

        early_stop = EarlyStopping(monitor="val_accuracy", patience=10)
        model = streetsigns_model(nbr_classes)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(train_generator,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=val_generator,
              callbacks=[ckpt_saver, early_stop]   
            )


    if TEST:

      model = tf.keras.models.load_model('./Models')
      model.summary()

      print("Evaluating validation set :")
      model.evaluate(val_generator)

      print("Evaluating test set :")
      model.evaluate(test_generator)