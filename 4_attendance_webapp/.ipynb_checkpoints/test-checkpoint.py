import numpy as np
import pandas as pd
import cv2

import redis

# insighface and ML algo importing
from insightface.app import FaceAnalysis
from sklearn.metrics import pairwise

# Redis/Database connection-----------------------------------------------------------
hostname = 'redis-19151.c245.us-east-1-3.ec2.cloud.redislabs.com'
portnumber = 19151
password = 'qGkQaZuXOd1kjWlu9ZXEBoeQQ5q0q1zE'

r = redis.StrictRedis(host = hostname,
                     port = portnumber,
                     password = password)


# face analysis config-----------------------------------------------------------------

faceapp = FaceAnalysis(name = 'buffalo_sc',
                     root = 'insightface_model',
                     providers = ['CPUExecutionProvider'])

faceapp.prepare (ctx_id = 0, det_size = (640,640), det_thresh = 0.5)


# ML algo -----------------------------------------------------------------------------

def ml_search_algo(dataframe,feature_column,test_vector, name_role = ['Name','Course','Email','Role'], thresh = 0.5):
    """
    cosine similarity base search Algorithm
    """
    # step 1 - take the dataframe
    dataframe = dataframe.copy()
    
    # step 2 - Index face embedding from the dataframe and convert into array
    X_list = dataframe[feature_column].tolist()
    x = np.asarray(X_list)

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
        person_name, person_role = data_filter.loc[argmax][name_role]
    else:
        person_name = 'Unknown'
        person_role = 'Unknown'
        
    
    return person_name, person_role


# detecting multiple face-----------------------------------------------------------------------

def face_prediction(test_image, dataframe,feature_column, name_role = ['Name','Course','Email','Role'], thresh = 0.5):
    # step 1 - take the test image and apply to insightface
    results = faceapp.get(test_image)
    test_copy = test_image.copy()
    
    # step 2 - use for loop and extract each embedding and pass to ml_search_algo
    for res in results:
        x1, y1, x2, y2 = res['bbox'].astype(int)
        embeddings = res ['embedding']
        person_name, person_role = ml_search_algo(dataframe,
                                                  'face_embeddings, 
                                                  test_vector = embeddings, 
                                                  name_role = name_role, 
                                                  thresh = thresh)
        if person_name == 'Unknown':
            color = (0,0,255)
        else:
            color = (0,255,0)
            
        cv2.rectangle(test_copy,(x1,y1),(x2,y2),color,1)
        test_gen = person_name
        cv2.putText(test_copy,test_gen,(x1,y1),cv2.FONT_HERSHEY_DUPLEX,0.5,color,1)  

    return test_copy
