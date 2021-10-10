# import the necessary packages
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
#import time
from scipy.spatial import distance as dist
#import os
import cv2


import streamlit as st
st.write("""
         # Mask Detection and Social Distancing
         """
         )
st.write("###########")
file = st.file_uploader("Please upload an image file", type=["jpg", "png","jpeg"])
choose_model = st.selectbox('Select a trained model:', ('EfficientNet','MobileNet'))


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.6:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

print("[INFO] loading face detector model...")
prototxtPath = "Streamlit/deploy.prototxt"
weightsPath = "Streamlit/Res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model('Streamlit/mask_det.hdf5')

thres=100

def find_centroids(locs):
    cent=[]
    for i,box in enumerate(locs):
        # unpack the bounding box and predictions
        startX, startY, endX, endY=box
        centx,centy=(startX+((endX-startX)/2)),(startY-((startY-endY)/2))
        cent.append((centx,centy))
        
    return cent

def violating_points(cent):
    Dist = dist.cdist(cent, cent, metric="euclidean")
    voilate=set()
    #     print(Dist)
    for i in range(0,Dist.shape[0]):
        for j in range(i+1,Dist.shape[1]):
    #       thres = cv2.getTrackbarPos("Threshold1","Parameters")
            if (Dist[i][j]) < int(thres):
                voilate.add(i)
                voilate.add(j)
    #     print(voilate)
    return voilate

from streamlit_webrtc import VideoTransformerBase, webrtc_streamer



class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0

    def transform(self, frame):
             #img = frame.to_ndarray(format="bgr24")
             
             # face mask or not
             #frame = cv2.resize(frame, (600,600))
             (locs, preds) = detect_and_predict_mask(img, faceNet, maskNet)


             #Finding distance if there are more than 1 people
             if(len(locs)>=1):

                 #Finding the Centriods b/w people
                 cent =find_centroids(locs)
                 # loop over the detected face locations and their corresponding
                 # locations

                 #Finding the voilating locations 
                 voilate = violating_points(cent)

                 red=[0,0,255]
                 green =[0,255,0]    
                 distance="Not Near"
                 #For distance
                 for (i,(box,cen)) in enumerate(zip(locs,cent)):
                     # unpack the bounding box and predictions
                     color=green
                     startX, startY, endX, endY=box
                     (cx,cy) = cen
             #         print(i)
             #         print(i in voilate)
                     if(i in voilate):
                         color = red
                         distance="Near"
                     g=6
                     cv2.rectangle(frame, (startX+g, startY-g), (endX-g, endY+g), color, 2)
                     cv2.circle(frame, (int(cx), int(cy)), 4, color, 3)
                     cv2.putText(frame, distance, (endX-30, endY + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)


             #For Mask
             # loop over the detected face locations and their corresponding
             # locations
             for (box, pred) in zip(locs, preds):
                 # unpack the bounding box and predictions
                 (startX, startY, endX, endY) = box
                 (mask, withoutMask) = pred

                 # determine the class label and color we'll use to draw
                 # the bounding box and text
                 label = "Mask" if mask > withoutMask else "No Mask"
                 color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                 # include the probability in the label
                 label = "{} ".format(label)

                 # display the label and bounding box rectangle on the output
                 # frame
                 cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                 cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

             return frame





def run():
    st.title("Face Detection using OpenCV")
    activities = ["Image", "Webcam","video"]
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.markdown("# Choose Input Source")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    link = '[©Developed by Spidy20](http://github.com/spidy20)'
    st.sidebar.markdown(link, unsafe_allow_html=True)
    if choice == 'Image':
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>* Mask Detection"</h4>''',
            unsafe_allow_html=True)
        img_file = st.file_uploader("Choose an Image", type=['jpg', 'jpeg', 'jfif', 'png'])
        if img_file is not None:
            img = np.array(Image.open(img_file))
            img1 = cv2.resize(img, (350, 350))
            place_h = st.beta_columns(2)
            place_h[0].image(img1)
            st.markdown(
                '''<h4 style='text-align: left; color: #d73b5c;'>* Increase & Decrease it to get better accuracy.</h4>''',
                unsafe_allow_html=True)

            # scale_factor = st.slider("Set Scale Factor Value", min_value=1.1, max_value=1.9, step=0.10, value=1.3)
            # min_Neighbors = st.slider("Set Scale Min Neighbors", min_value=1, max_value=9, step=1, value=5)
            # fd, count, orignal_image = face_detect(img, scale_factor, min_Neighbors)
            place_h[1].image(fd)
            
            st.markdown(get_image_download_link(result, img_file.name, 'Download Image'), unsafe_allow_html=True)
    if choice == 'Webcam':
        st.markdown(
            '''<h4 style='text-align: left; color: #d73b5c;'>* It might be not work with Android Camera"</h4>''',
            unsafe_allow_html=True)
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
run()








# print("[INFO] starting video stream...")
# vs = cv2.VideoCapture("test/bs1.mp4") 
# # vs = cv2.VideoCapture(0)# for direct cam
# # time.sleep(1.0)
# # vs = cv2.VideoCapture("")  # For video uncomment it
# pTime =0
# cTime=0
# cent=[]
# # loop over the frames from the video stream
# while True:
#     # grab the frame from the threaded video stream and resize it
#     # to have a maximum width of 400 pixels
#     cent=[]
#     # grab the frame from the threaded video stream and resize it
#     # to have a maximum width of 400 pixels
    
#     (grabbed, frame) = vs.read() #For video uncomment it  or direct cam

    
# #     frame= cv2.imread("test/p7.jpg") # For image uncomment it
#     frame = cv2.resize(frame, (600,600))


#     # detect faces in the frame and determine if they are wearing a
#     # face mask or not
#     (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    
#     #Finding distance if there are more than 1 people
#     if(len(locs)>=1):
        
#         #Finding the Centriods b/w people
#         cent =find_centroids(locs)
#         # loop over the detected face locations and their corresponding
#         # locations

#         #Finding the voilating locations 
#         voilate = violating_points(cent)

#         red=[0,0,255]
#         green =[0,255,0]    
#         distance="Not Near"
#         #For distance
#         for (i,(box,cen)) in enumerate(zip(locs,cent)):
#             # unpack the bounding box and predictions
#             color=green
#             startX, startY, endX, endY=box
#             (cx,cy) = cen
#     #         print(i)
#     #         print(i in voilate)
#             if(i in voilate):
#                 color = red
#                 distance="Near"
#             g=6
#             cv2.rectangle(frame, (startX+g, startY-g), (endX-g, endY+g), color, 2)
#             cv2.circle(frame, (int(cx), int(cy)), 4, color, 3)
#             cv2.putText(frame, distance, (endX-30, endY + 20),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    
#     #For Mask
#     # loop over the detected face locations and their corresponding
#     # locations
#     for (box, pred) in zip(locs, preds):
#         # unpack the bounding box and predictions
#         (startX, startY, endX, endY) = box
#         (mask, withoutMask) = pred

#         # determine the class label and color we'll use to draw
#         # the bounding box and text
#         label = "Mask" if mask > withoutMask else "No Mask"
#         color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            
#         # include the probability in the label
#         label = "{} ".format(label)

#         # display the label and bounding box rectangle on the output
#         # frame
#         cv2.putText(frame, label, (startX, startY - 10),
#             cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
#         cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    
    
#     # Find FPS
#     cTime=time.time()
#     fps=1/(cTime-pTime)
#     pTime=cTime
#     cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_ITALIC,2,(255,0,255),3)
#     # show the output frame
# #     cv2.startWindowThread()
#     cv2.namedWindow("Frame")
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF

#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break
        
# cv2.destroyAllWindows()
