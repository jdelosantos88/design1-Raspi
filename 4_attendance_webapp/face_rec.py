import numpy as np
import pandas as pd 
import cv2
import redis
import os

# importing insightface (responsible for gathering of face embeddings and data from the persons face)
from insightface.app import FaceAnalysis
# importing a library for the machine learning, that is responsible for identidying 
# the identity of the user by comparing face from database to the live web cam
from sklearn.metrics import pairwise

import time
from datetime import datetime
#-------------------------------------------------------------------------------------------------------------------

# database configuration
hostname = 'redis-15423.c326.us-east-1-3.ec2.cloud.redislabs.com'
portnumber = 15423
password = 'VhOs0K8XmzE8pQn7dF26scwa2ji92Ut4'

r = redis.StrictRedis(host = hostname,
                     port = portnumber,
                     password = password)
#-------------------------------------------------------------------------------------------------------------------

# Retrieve or Extract Data from the database
def retreive_data(name):
    name = 'register'
    retrieve_dict = r.hgetall(name)
    retrieve_series = pd.Series(retrieve_dict)
    retrieve_series = retrieve_series.apply(lambda x: np.frombuffer(x,dtype=np.float32))
    index = retrieve_series.index
    index = list(map(lambda x: x.decode(),index))
    retrieve_series.index = index
    retrieve_df = retrieve_series.to_frame().reset_index()
    retrieve_df.columns = ['user_info', 'face_embeddings']
    retrieve_df[['Name', 'Course', 'IDnumber', 'SPN','GPN']] = retrieve_df['user_info'].apply(lambda x: x.split('%')).apply(pd.Series)
    return retrieve_df[['Name', 'Course', 'IDnumber', 'SPN', 'GPN', 'face_embeddings']]


#-------------------------------------------------------------------------------------------------------------------

# configure face analysis model from insightface
faceapp = FaceAnalysis(name = 'buffalo_sc',
                     root = 'insightface_model',
                     providers = ['CPUExecutionProvider'])

faceapp.prepare (ctx_id = 0, det_size = (640,640), det_thresh = 0.5)

#-------------------------------------------------------------------------------------------------------------------

# machine learning (cosine similarity algorithm)
def ml_search_algo(dataframe,feature_column,test_vector, user_info = ['Name','Course', 'IDnumber', 'SPN', 'GPN'], thresh = 0.5):
    """
    cosine similarity base search Algorithm
    """
    # step 1 - take the dataframe
    dataframe = dataframe.copy()
    
    # step 2 - Index face embedding from the dataframe and convert into array
    x_list = dataframe[feature_column].tolist()
    x = np.asarray(x_list)

    # step 3 - calculate cosine similarity
    similar = pairwise.cosine_similarity(x,test_vector.reshape(1,-1))  # 1,-1 is equals to 1,512 vector
    similar_arr = np.array(similar).flatten()
    dataframe['cosine'] = similar_arr

    # step 4 - filter the data
    data_filter = dataframe.query(f'cosine >= {thresh}')
    if len(data_filter) > 0:

        # step 5 - get the person name
        data_filter.reset_index(drop = True, inplace = True)
        argmax = data_filter['cosine'].argmax()
        user_name, user_course, user_idnumber, user_spn, user_gpn = data_filter.loc[argmax][user_info]
    else:
        user_name = 'Unknown'
        user_idnumber = 'Unknown'
        user_course = 'Unknown'
        user_spn = 'Unknown'
        user_gpn = 'Unknown'
        
    
    return user_name, user_course, user_idnumber, user_spn, user_gpn

#-------------------------------------------------------------------------------------------------------------------

# saving logs on database every minute 

class RealTimePrediction:
    def __init__(self) -> None:
        self.logs = dict(name =[], course =[], idnumber =[], spn =[], gpn =[], current_time =[])

    def reset_dict(self):
        self.logs = dict(name =[], course =[], idnumber =[], spn =[], gpn =[], current_time =[])

    def save_logs_db(self):
        # 1. creating logs dataframe
        dataframe = pd.DataFrame(self.logs)

        # 2. drop the duplicate information according to the users name or studentID/userID (distinct inforrmation)
        dataframe.drop_duplicates('name',inplace=True)

        # 3. pushing data to database (combine all value into one string because database can only store 1 value as a list name)
        name_list = dataframe['name'].tolist()
        course_list = dataframe['course'].tolist()
        idnumber_list = dataframe['idnumber'].tolist()
        spn_list = dataframe['spn'].tolist()
        gpn_list = dataframe['gpn'].tolist()
        current_time_list = dataframe['current_time'].tolist()
        encoded_data = []

        for name,course,idnumber,spn,gpn,current_time in zip (name_list, course_list, idnumber_list, spn_list, gpn_list, current_time_list):
            if name != 'Unknown':
                link_string = f"{name}%{course}%{idnumber}%{spn}%{gpn}%{current_time}"
                encoded_data.append(link_string)

        if len(encoded_data) > 0:
            r.lpush('attendance:logs', *encoded_data)
            r.expire('attendance:logs', 60)

        self.reset_dict()


    def face_prediction(self,test_image, dataframe,feature_column, user_info = ['Name','Course', 'IDnumber','SPN','GPN'], thresh = 0.5):

        current_time = str(datetime.now())

        # step 1 - take the test image and apply to insightface
        results = faceapp.get(test_image)
        test_copy = test_image.copy()
        
        # step 2 - use for loop and extract each embedding and pass to ml_search_algo
        for res in results:
            x1, y1, x2, y2 = res['bbox'].astype(int)
            embeddings = res ['embedding']
            user_name, user_course, user_spn, user_gpn, user_idnumber = ml_search_algo(dataframe,
                                                                            feature_column, 
                                                                            test_vector = embeddings, 
                                                                            user_info = user_info, 
                                                                            thresh = thresh)
            if user_name == 'Unknown':
                color = (0,0,255)
            else:
                color = (0,255,0)
                
            cv2.rectangle(test_copy,(x1,y1),(x2,y2),color,1)
            #test_gen = user_name
            #cv2.putText(test_copy,test_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.65,color,1)
            #cv2.putText(test_copy, current_time,(x1,y2+10),cv2.FONT_HERSHEY_DUPLEX,0.4,color,1)    

            # saving info in logs dictionary
            self.logs['name'].append(user_name)
            self.logs['course'].append(user_course)
            self.logs['idnumber'].append(user_idnumber)
            self.logs['spn'].append(user_spn)
            self.logs['gpn'].append(user_gpn)
            self.logs['current_time'].append(current_time)

        return test_copy
    
#### Registration form
class RegistrationForm:
    def __init__(self):
        self.sample = 0
    def reset(self):
        self.sample = 0

    def get_embedding(self,frame):
        # get results from insightface model
        results = faceapp.get(frame, max_num = 1)
        embeddings = None
        for res in results:
            self.sample += 1
            x1, y1, x2, y2 = res['bbox'].astype(int)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),1)
            text = f"samples = {self.sample}"
            cv2.putText(frame,text,(x1,y1), cv2.FONT_HERSHEY_DUPLEX,0.6,(255,255,0),2)

            # extracting facial features
            embeddings = res['embedding']

        return frame, embeddings
    
# saving data to the database
    
    def save_data_in_database(self, name, course, idnumber, spn, gpn):

        # info validation' to avoid blank value
        if name is not None:
            if name.strip() != '':
                key = f'{name}%{course}%{idnumber}%{spn}%{gpn}'
            else:
                return 'name_false'
        else:
            return 'name_false'
        
        # if face embedding.txt exist
        if 'face_embedding.txt' not in os.listdir():
            return 'file_false'

        # load face_embeddings.txt
        x_array = np.loadtxt('face_embedding.txt', dtype = np.float32) #float32 - converted to this data type for smaller file size/memory

        # convert it into an array
        received_samples = int(x_array.size/512) #divide to 12 because each face embeddings have 512 values
        x_array = x_array.reshape(received_samples,512)
        x_array = np.asarray(x_array)

        # calculate mean embeddings (for smaller file size)
        x_mean = x_array.mean(axis = 0)
        x_mean = x_mean.astype(np.float32)
        x_mean_bytes = x_mean.tobytes()

        # save it to the database 

        r.hset(name = 'register', key = key,value = x_mean_bytes)

        # 
        os.remove('face_embedding.txt') # to only save person info
        self.reset()

        return True
