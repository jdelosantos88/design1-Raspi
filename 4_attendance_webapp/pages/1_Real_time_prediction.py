import cv2
import streamlit as st
from Home import face_rec
from streamlit_webrtc import webrtc_streamer
import av
import time

side_bar = """
<style>
[data-testid="stSidebarContent"]{
    background-color: #114B26; 
    padding: 20px;
    border-radius: 10px; 
}
</style>
"""

st.set_page_config(page_title='Prediction')

st.markdown(side_bar, unsafe_allow_html=True)
st.subheader('Real Time Attendance system')


# Retrieve Data from the database----------------------------------------------------------------------------------

with st.spinner('Preparing Face Recognition Model...'):
    redis_face_db = face_rec.retreive_data(name='register')
    st.success('Attendance System is eady to use')

# Time where saving in logs interval to avoid a lot of data processing--------------------------------------------

waitTime = 10
setTime = time.time()
realTimepred = face_rec.RealTimePrediction()

# Real Time Attendance Check---------------------------------------------------------------------------------------


# cv2 Videocapture

import cv2
import av
import time
import numpy as np

display_width, display_height = 1280, 720

def video_frame_callback(frame):
    global setTime
    new_width, new_height = 3840, 2160

    pred_img = realTimepred.face_prediction(frame, redis_face_db,'face_embeddings',
                                        ['FName','LName','Course','IDnumber','SPN'], thresh=0.5)
    higher_resolution_img = cv2.resize(pred_img, (new_width, new_height))

    timenow = time.time()
    difftime = timenow - setTime
    if difftime >= waitTime: 
        realTimepred.save_logs_db()
        setTime = time.time()
        print('Save Data to database')

    # Resize the frame to fit the display size
    resized_frame = cv2.resize(higher_resolution_img, (display_width, display_height))

    return resized_frame

# OpenCV camera capture
cap = cv2.VideoCapture(0)

# Set the window size to match the display size
cv2.namedWindow('Realtime Attendance', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Realtime Attendance', display_width, display_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed_frame = video_frame_callback(frame)

    # Display the processed frame
    cv2.imshow('Realtime Attendance', processed_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy any OpenCV windows
cap.release()
cv2.destroyAllWindows()




# WECRTC videocapture

# def video_frame_callback(frame):
#     global setTime
#     new_width, new_height = 3840, 2160

#     img = frame.to_ndarray(format="bgr24") # 3 dimension numpy array
#     pred_img = realTimepred.face_prediction(img, redis_face_db,'face_embeddings',
#                                         ['FName','LName','Course','IDnumber','SPN'], thresh = 0.5)
#     higher_resolution_img = cv2.resize(pred_img, (new_width, new_height))

#     timenow = time.time()
#     difftime = timenow - setTime
#     if difftime >= waitTime: 
#         realTimepred.save_logs_db()
#         setTime = time.time()
#         print('Save Data to database')

#     return av.VideoFrame.from_ndarray(pred_img, format="bgr24")

# webrtc_streamer(
#     key="RealtimeAttendance",
#     video_frame_callback=video_frame_callback,
#     media_stream_constraints={
#         "video": {
#             "width": {"ideal": 2500, "min": 1280},
#             "height": {"ideal": 1000, "min": 720},
#         },
#     }
# )
