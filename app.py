import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from streamlit_option_menu import option_menu

main_menu_options = ["Home", "Deforms ChangePoints", "Contact"]

# Create a function to handle the different pages
def handle_menu_option(option):
    if option == "Home":
        show_home_page()
    elif option == "Deforms ChangePoints":
        show_deformability_page()
    elif option == "Contact":
        show_contact_page()

# Define the home page content
def show_home_page():
    st.title('Sickle Cell Image Classification')
    # Load the trained model
    model = tf.keras.models.load_model("model.h5")

    def preprocess_image(image):
        img = image.resize((299, 299))
        img = img.convert('RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.inception_v3.preprocess_input(img_array)
        return img_array

    def get_class_label(class_index):
        class_labels = ['Phase1', 'Phase2', 'Phase3', 'Phase4', 'Phase5']
        return class_labels[class_index]

    def classify_image(image, image_label):
        img = preprocess_image(image)
        prediction = model.predict(img)
        class_index = np.argmax(prediction)
        class_label = get_class_label(class_index)
        percentage_predictions = [round(float(p) * 100, 2) for p in prediction[0]]

        data = pd.DataFrame({
            'Class': [get_class_label(i) for i in range(len(percentage_predictions))],
            'Percentage': percentage_predictions
        })

        st.bar_chart(data.set_index('Class'))

        return class_label, percentage_predictions, image_label

    uploaded_files = st.file_uploader('Upload multiple images', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

    if uploaded_files:
        images = []
        image_labels = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            images.append(image)
            image_labels.append(uploaded_file.name)

        if st.button('Classify Images'):
            all_predictions = []
            all_percentage_predictions = []
            for image, image_label in zip(images, image_labels):
                class_label, percentage_predictions, image_label = classify_image(image, image_label)
                st.write('Uploaded Image Label:', image_label)
                st.write('Predicted Class:', class_label)
                st.write('Percentage Predictions:', percentage_predictions)
                all_predictions.append(class_label)
                all_percentage_predictions.append(percentage_predictions)
                st.write('---')

            if all_percentage_predictions:
                average_prediction = np.mean(all_percentage_predictions, axis=0)
                max_index = np.argmax(average_prediction)
                predicted_phase = get_class_label(max_index)
                st.write('Predicted Phase:', predicted_phase)
                data = pd.DataFrame({
                    'Class': [get_class_label(i) for i in range(len(average_prediction))],
                    'Percentage': average_prediction
                })
                st.bar_chart(data.set_index('Class'))


# Define the sickle cell images deformability change points page content

def show_deformability_page():
    st.write("This is the Sickle Cell Images Deformability Change Points")

    # Display the image
    image_path = "C:\\Users\\Inholland\\Desktop\\Sickle_cell_project\\segmented_mse_plot.png"
    image = Image.open(image_path)
    st.image(image, caption='Sickle Cell Images Deformability Change Points', use_column_width=True)


# Define the contact page content
def show_contact_page():
    st.write("This is the Contact page")
    st.write("Contact information goes here")

# Create a sidebar menu
with st.sidebar:
    selected_option = option_menu(
        menu_title="Main Menu",
        options=main_menu_options
    )

# Handle the selected menu option
handle_menu_option(selected_option)