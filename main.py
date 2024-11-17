import streamlit as st
import tensorflow as tf
import numpy as np
import tempfile
import matplotlib.image as mpimg
model = tf.keras.models.load_model('3.keras')

def preprocess_image(image_path):
    img = mpimg.imread(image_path)
    img = np.expand_dims(img, axis=0)
    return img


def predict(image_path):
    # Preprocess the image
    processed_img = preprocess_image(image_path)
    
    # Make prediction
    prediction = np.argmax(model.predict(processed_img)[0])
    return prediction

st.title('Keras Model Deployment with Streamlit')
st.write('Upload an image for prediction')
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    temp_image = tempfile.NamedTemporaryFile(delete=False)
    temp_image.write(uploaded_file.read())
    st.image(temp_image.name, caption='Uploaded Image', use_column_width=True)
    prediction = predict(temp_image.name)
    if(prediction==0):
        prediction = "Early"
    elif(prediction==1):
        prediction = "Late"
    else:
        prediction = "Healthy"
    st.write('Prediction:',prediction)
