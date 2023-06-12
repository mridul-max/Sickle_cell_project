import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from streamlit_option_menu import option_menu


st.title('Sickle Cell Image Classification')

#Lets add a sidebar menu

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu", #Required
        options = ["Home","Projects ", "Contact"],
    
    )
# Load the trained model
model = tf.keras.models.load_model("model.h5")

def preprocess_image(image):
    img = image.resize((299, 299))
    img = img.convert('RGB')  # Convert image to RGB format
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
    return img_array

def get_class_label(class_index):
    class_labels = ['Phase1', 'Phase2', 'Phase3', 'Phase4', 'Phase5']
    return class_labels[class_index]

def classify_image(image):
    # Preprocess the image
    img = preprocess_image(image)

    # Make prediction using the loaded model
    prediction = model.predict(img)
    class_index = np.argmax(prediction)
    class_label = get_class_label(class_index)
    percentage_predictions = [round(float(p) * 100, 2) for p in prediction[0]]

    # Create a pandas DataFrame for visualization
    data = pd.DataFrame({
        'Class': [get_class_label(i) for i in range(len(percentage_predictions))],
        'Percentage': percentage_predictions
    })

    # Visualize the percentages as horizontal bars
    st.bar_chart(data.set_index('Class'))

    return class_label, percentage_predictions

def main():
    uploaded_files = st.file_uploader('Upload multiple images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        # Store the uploaded images
        images = []
        for uploaded_file in uploaded_files:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            images.append(image)

        if st.button('Classify Images'):
            all_predictions = []
            all_percentage_predictions = []
            for image in images:
                # Perform classification on each uploaded image
                class_label, percentage_predictions = classify_image(image)
                st.write('Predicted Class:', class_label)
                st.write('Percentage Predictions:', percentage_predictions)
                all_predictions.append(class_label)
                all_percentage_predictions.append(percentage_predictions)
                st.write('---')  # Add a separator between images

            if all_percentage_predictions:
                # Calculate average percentage predictions
                average_prediction = np.mean(all_percentage_predictions, axis=0)

                # Create a pandas DataFrame for visualization
                data = pd.DataFrame({
                    'Class': [get_class_label(i) for i in range(len(average_prediction))],
                    'Percentage': average_prediction
                })

                # Visualize the average percentages as horizontal bars
                st.bar_chart(data.set_index('Class'))

if __name__ == '__main__':
    main()
