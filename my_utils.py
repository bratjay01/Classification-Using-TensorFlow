import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
import shutil
import csv
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def display_some_examples(examples,labels):

    plt.figure(figsize=(10,10))

    for i in range(25):

        idx = np.random.randint(0, examples.shape[0]-1) #basically choosing any image between 59,999 & 0
        img = examples[idx]
        label = labels[idx]

        plt.subplot(5,5, i+1)
        plt.title(str(label)) #transforming label to string
        plt.tight_layout()#in order to arrange the alignment of output pictures
        plt.imshow(img, cmap='gray')  #show image#in order to tell matplotlib that this is grayscale
    plt.show()


def split_data(path_to_data, path_to_save_train, path_to_save_val, split_size=0.1):
    
    folders = os.listdir(path_to_data) 

    for folder in folders:   #loop to go through all the folders in the main folder
        full_path = os.path.join(path_to_data, folder) #to attach name of subfolders with main folder "full path"
        images_paths = glob.glob(os.path.join(full_path, "*.png")) #to give a list of all files with .png
        
        x_train, x_val = train_test_split(images_paths, test_size=split_size) #split the list of all of the images into train and val

        for x in x_train:        #for a new train set which we made

            path_to_folder = os.path.join(path_to_save_train, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)  #copy images and paste it inside a new dir which we constructed

        for x in x_val:      #for a new val set which we made

            path_to_folder = os.path.join(path_to_save_val, folder)

            if not os.path.isdir(path_to_folder):
                os.makedirs(path_to_folder)

            shutil.copy(x, path_to_folder)



def order_test_set(path_to_images, path_to_csv):
    
   

    try:
        with open(path_to_csv, 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter= ',')

            for i, row in enumerate(reader):

                if i==0:
                    continue

                img_name = row[-1].replace('Test/', '')  #go through all index values of each line of csv file and get the last two values
                label = row[-2]

                path_to_folder = os.path.join(path_to_images, label)

                if not os.path.isdir(path_to_folder):
                    os.makedirs(path_to_folder)

                img_full_path = os.path.join(path_to_images, img_name)
                shutil.move(img_full_path, path_to_folder)
    except:
        print('[INFO] : Error reading csv file')


def create_generators(batch_size, train_data_path, val_data_path, test_data_path):

    preprocessor = ImageDataGenerator(
        rescale = 1 / 255.          # to rescale the images
    
    )

    train_generator = preprocessor.flow_from_directory(   #Takes a path to a folder and gonna look at all folders that exists, and its gonna classsify images that belong to the same class    
        train_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode='rgb',
        shuffle=True, #basically the order of images will not be the same for each epoch or batches= helps our model to be more generalized and unbiased
        batch_size=batch_size
    )

    val_generator = preprocessor.flow_from_directory(       
        val_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode='rgb',
        shuffle=False, 
        batch_size=batch_size
    )
    
    test_generator = preprocessor.flow_from_directory(       
        test_data_path,
        class_mode="categorical",
        target_size=(60,60),
        color_mode='rgb',
        shuffle=False, 
        batch_size=batch_size
    )
    
    return train_generator, val_generator, test_generator