import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('mnist_digits_model.h5')

# Function to preprocess the canvas image
def preprocess_image(image, target_size=(28, 28)):
    if image.mode != "L":
        image = image.convert("L")
    image = image.resize(target_size)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image / 255.0

# Set up Streamlit layout
st.title("MNIST Digit Classifier")
st.markdown("Draw a digit (0-9) below and click classify to see the model's prediction.")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color = "#000000",
    stroke_width = 10,
    stroke_color = "#FFFFFF",
    background_color = "#000000",
    width = 280,
    height = 280,
    drawing_mode = "freedraw",
    key = "canvas",
)

# Button to classify the drawing
if st.button('Classify'):
    if canvas_result.image_data is not None:
        # Process the image from canvas
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA') 
        img = preprocess_image(img)

        # Make prediction
        preds = model.predict(img)  # [0.01, 0.0003,0.3,0.01, .......,0.02]
        pred_class = np.argmax(preds, axis=1)[0]
        st.write(f'Prediction: {pred_class}')
    else:
        st.write("Please draw a digit to classify.")
