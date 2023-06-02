import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import tensorflow as tf
from sklearn.metrics import confusion_matrix

st.title('Sickle Cell detection')

#Lets add a sidebar menu

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu", #Required
        options = ["Home","Projects ", "Contact"],

    )


# Load the model
try:
    model = tf.keras.models.load_model('model.h5')
    st.write("Model loaded successfully!")
except:
    st.write("Failed to load the model!")



