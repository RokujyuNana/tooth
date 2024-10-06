import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the quantized TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path='tooth_health_model_quant.tflite')
    interpreter.allocate_tensors()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()  # Stop further execution if model loading fails

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_image(img):
    # Preprocess the image
    img = img.resize((150, 150))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Set the tensor for input
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()  # Run the inference

    # Get the output and return the predicted category
    prediction = interpreter.get_tensor(output_details[0]['index'])
    categories = ['Disklorasi Gigi', 'Gigi Sehat', 'Karies Gigi', 'Radang Gusi']
    return categories[np.argmax(prediction)]

# Streamlit interface
st.title('Dental Health Check')

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    if st.button('Predict'):
        prediction = predict_image(img)
        st.write(f"Prediction: {prediction}")
