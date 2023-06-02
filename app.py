import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow import keras


st.title('Sickle Cell detection')

#Lets add a sidebar menu

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu", #Required
        options = ["Home","Projects ", "Contact"],
    
    )
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

    return class_label


def main():
    st.title('Sickle Cell Image Classification')
    uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform classification on the uploaded image
        class_label = classify_image(image)
        st.write('Predicted Class:', class_label)

if __name__ == '__main__':
    main()
