import tensorflow as tf
import numpy as np



def predict_with_model(model, imgpath):

    image = tf.io.read_file(imgpath)   #load this image and save it inside img var
    image = tf.image.decode_png(image, channels=3)   #to decode the image
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)  #convert the image pixels as it is unsigned int it converts to float32 and scale them 
    image = tf.image.resize(image, [60,60])  #(60,60,3)
    image = tf.expand_dims(image, axis=0)  #(1,60,60,3) one image of 60,60,3 dimension
    
    predictions = model.predict(image) #to return the list of probabilities of that image belonging to one of the classes
    predictions = np.argmax(predictions)

    return predictions

if __name__== '__main__':

    img_path = "D:\\German_traffic_sign\\Test\\2\\00464.png"
    #img_path = "D:\\German_traffic_sign\\Test\\0\\03474.png"
    #img_path = "D:\\German_traffic_sign\\Test\\27\\04562.png"
    model = tf.keras.models.load_model('./Models')
    prediction = predict_with_model(model, img_path)

    print(f"prediction = {prediction}")