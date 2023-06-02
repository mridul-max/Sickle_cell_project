import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import tensorflow as tf
from sklearn.metrics import confusion_matrix
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