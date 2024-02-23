import streamlit as st
import face_rec

st.set_page_config(page_title = 'Attendance System', layout = 'wide')

st.header('Attendance Checking System')

with st.spinner("Loading Model and Connecting to Database..."):
    import face_rec

st.success ('Model Loaded Successfully')
st.success ('Database Successfully Connected')