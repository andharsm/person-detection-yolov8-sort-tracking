import streamlit as st
import cv2
from person_detection import display_demo
from overview import display_overview

VIDEO_PATH = 'person_detection_yolov8/videos/person in public area.mp4'

# title
st.title("Person Detection in Public Area")

# Initialize session state if not already done
if 'menu_selection' not in st.session_state:
    st.session_state.menu_selection = "Overview"

# Sidebar menu
menu_selection = st.sidebar.selectbox("Menu", ["Overview", "Demo Project"], index=0 if st.session_state.menu_selection == "Overview" else 1)

# Update session state based on menu selection
st.session_state.menu_selection = menu_selection

if menu_selection == "Overview":
    st.sidebar.markdown("Overview selected.")
    display_overview()
    
elif menu_selection == "Demo Project":
    st.sidebar.markdown("Demo Project selected.")
    display_demo()
