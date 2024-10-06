import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the quantized TFLite model
interpreter = tf.lite.Interpreter(model_path='dental_health_model_quant.tflite')
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(img):
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    categories = ['Disklorasi Gigi', 'Gigi Sehat', 'Karies Gigi', 'Radang Gusi']
    return categories[np.argmax(output_data)]

# Streamlit interface
st.title('Dental Health Check')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((150, 150))
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        prediction = predict_image(img)
        st.write(f"Prediction: {prediction}")
