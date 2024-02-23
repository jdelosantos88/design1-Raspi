import cv2
import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

st.set_page_config(page_title='Prediction')
st.subheader('Real Time Attendance system')


# Retrieve Data from the database----------------------------------------------------------------------------------

with st.spinner('Retrieving Data from Database...'):
    redis_face_db = face_rec.retreive_data(name='register')


st.success('Data Successfully Loaded from the Database')

# Time where saving in logs interval to avoid a lot of data processing--------------------------------------------

waitTime = 10
setTime = time.time()
realTimepred = face_rec.RealTimePrediction()

# Real Time Attendance Check---------------------------------------------------------------------------------------

# 'streamlit webrtc' for video capture


def video_frame_callback(frame):
    global setTime
    new_width, new_height = 3840, 2160

    img = frame.to_ndarray(format="bgr24") # 3 dimension numpy array
    pred_img = realTimepred.face_prediction(img, redis_face_db,'face_embeddings',
                                        ['Name','Course','IDnumber','SPN','GPN'], thresh = 0.5)
    higher_resolution_img = cv2.resize(pred_img, (new_width, new_height))

    

    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime: 
        realTimepred.save_logs_db()
        setTime = time.time()
        print('Save Data to database')

    return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

webrtc_streamer(
    key="RealtimeAttendance",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 2500, "min": 1280},
            "height": {"ideal": 1000, "min": 720},
        },
    }
)
