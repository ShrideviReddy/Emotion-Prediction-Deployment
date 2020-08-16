import tensorflow as tf
import streamlit as st
import cv2
from PIL import Image, ImageOps
import numpy as np


def import_and_predict(image_data, model):
    
        size = (48,48)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        image = np.expand_dims(image, axis = 0)
        image /= 255
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        #img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(image)
        
        return prediction

model = tf.keras.models.load_model('model_filter.hdf5')
st.write("""
         # Facial Emotion Prediction
         """
         )
st.write("This is a simple image classification web app to predict emotion based on facial expression")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    if np.argmax(prediction) == 0:
        st.write("Angry!")
    elif np.argmax(prediction) == 1:
        st.write("Disgust!")
    elif np.argmax(prediction) == 2:
        st.write("Fear!")
    elif np.argmax(prediction) == 3:
        st.write("Happy!")
    elif np.argmax(prediction) == 4:
        st.write("Sad")
    elif np.argmax(prediction) == 5:
        st.write("Surprise")
    else:
        st.write("Neutral")
