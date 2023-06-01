import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu


st.title('Sickle Cell detection')

#Lets add a sidebar menu

with st.sidebar:
    selected = option_menu(
        menu_title = "Main Menu", #Required
        options = ["Home","Projects ", "Contact"],

    
    )