import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('edelweis_model.h5')

# Initialize label encoder (make sure to use the same labels as in your training)
lbl = LabelEncoder()
lbl.fit(['Leucogenes_Grandiceps', 'Anaphalis_Javanica', 'Leontopodium_Alpinum'])

# Function to predict the uploaded image
def predict_uploaded_image(uploaded_image):
    # Open the uploaded image
    img = Image.open(uploaded_image).convert('RGB')

    # Resize the image to match model's expected input size
    img_resized = img.resize((150, 150))  # Resize to 150x150

    # Preprocessing: Normalizing and adding batch dimension
    img_array = np.array(img_resized) / 255.0  # Normalization
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make prediction with the model
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = lbl.inverse_transform([predicted_class])[0]

    return img, predicted_label

# Streamlit app layout
st.title('Edelweiss Flower Classification')
st.write('Upload an image of Edelweiss flower to predict its species.')

# File uploader widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Predict and display result
    img, predicted_label = predict_uploaded_image(uploaded_file)
    
    st.write(f"Predicted Motif: {predicted_label}")
    
    # Display the image and prediction result
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(f'Predicted Motif: {predicted_label}')
    plt.axis('off')  # Hide axes for better visualization
    st.pyplot(plt)