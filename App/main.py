import cv2
import os
import time
import random
from yolo import YOLO
from preprocessing import handDetector, resize, skinDetector
from tensorflow import keras
from collections import defaultdict
import json
from tensorflow.python.keras.preprocessing import image
import numpy as np
import tensorflow as tf
from select_final import selectFinal
import shutil


# TENSORFLOW CODE
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def decode_predictions(preds, top=5, model_json=""):
    CLASS_INDEX = json.load(open(model_json))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        for i in top_indices:
            each_result = []
            each_result.append(CLASS_INDEX[str(i)])
            each_result.append(pred[i])
            results.append(each_result)

    return results
    
def preprocess_input(x):
    # 'RGB'->'BGR'
    x *= (1./255)
    return x

def predictImage(image_input, jsonPath, result_count):
    image_to_predict = image.load_img(image_input, target_size=(224,224))
    image_to_predict = image.img_to_array(image_to_predict, data_format="channels_last")
    image_to_predict = np.expand_dims(image_to_predict, axis=0)
    image_to_predict = preprocess_input(image_to_predict)
    prediction = model.predict(x=image_to_predict, steps=1)
    prediction_results = []
    prediction_probabilities = []
    predictiondata = decode_predictions(prediction, top=int(result_count), model_json=jsonPath)
    for result in predictiondata:
        prediction_results.append(str(result[0]))
        prediction_probabilities.append(str(result[1] * 100))
    return prediction_results, prediction_probabilities

#----------------------------------------------------------------------------#

print("Stage 0: Detect Face Frames for Activation")
'''
Stage 0: Detect Face Frames for Activation
If the user looks straight at the camera for a couple of seconds, their face will
be detected, signifying that they wish to record a message
'''
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                     "haarcascade_frontalface_default.xml")
color_blue = (255, 0, 0)
color_green = (0, 255, 0)

cap = cv2.VideoCapture(0)
ret = cap.set(3, 864)
ret = cap.set(4, 480)

face_detect = False
face_detected_count = 0

while face_detect is False:
    ret, frame = cap.read()
    
    if ret is True:
        frame = cv2.flip(frame, 1)
        faces = face_cascade.detectMultiScale(frame, scaleFactor = 1.05, minNeighbors = 5)
        
        if len(faces) != 0:
            face_detected_count = face_detected_count + 1
            
            text = "Face detected"
            cv2.putText(frame, text, (600, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color_blue, 1)
            
            for(x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_green, 1)
        
        else:
            if face_detected_count > 0:
                face_detected_count = face_detected_count - 1
        
        cv2.imshow("Face Detection", frame)
        cv2.waitKey(1)
        
        #print(face_detected_count)
        
        if face_detected_count >= 35:
            face_detect = True
            face_detected_count = 0
        
    else:
        if face_detected_count > 0:
            face_detected_count = face_detected_count - 1
        print(face_detected_count)
        
        continue

cap.release()
cv2.destroyAllWindows()

video_folder = os.path.join(os.getcwd(),'videos')
CHECK_FOLDER = os.path.isdir(video_folder)
if CHECK_FOLDER:
    shutil.rmtree(video_folder)
    os.mkdir(video_folder)
else :
    os.mkdir(video_folder)
    
frames_folder = os.path.join(os.getcwd(), 'frames')
CHECK_FOLDER = os.path.isdir(frames_folder)
if CHECK_FOLDER:
    shutil.rmtree(frames_folder)
    os.mkdir(frames_folder)
else :
    os.mkdir(frames_folder)

boxed_frames_folder = os.path.join(os.getcwd(), 'frames_boxed')
CHECK_FOLDER = os.path.isdir(boxed_frames_folder)
if CHECK_FOLDER:
    shutil.rmtree(boxed_frames_folder)
    os.mkdir(boxed_frames_folder)
else :
    os.mkdir(boxed_frames_folder)

segmented_frames_folder = os.path.join(os.getcwd(), 'frames_segmented')
CHECK_FOLDER = os.path.isdir(segmented_frames_folder)
if CHECK_FOLDER:
    shutil.rmtree(segmented_frames_folder)
    os.mkdir(segmented_frames_folder)
else :
    os.mkdir(segmented_frames_folder)

#-----------------------------------------------------------------------------#

print("Stage 1: Record Webcam Feed containing the Hand Sign")
'''
Stage 1: Record Webcam Feed containing the Hand Sign
The user will make the appropriate single hand gesture corresponding to the 
message they wish to send 
This video clip is saved and used in further steps
'''
clip_timestamp = str(time.time()).split('.')[0]
clip_path = os.path.join(video_folder, clip_timestamp + ".mp4")

video_codec = cv2.VideoWriter_fourcc(*'MP4V')

fps = 24

cap1 = cv2.VideoCapture(0)
ret = cap1.set(3, 864)
ret = cap1.set(4, 480)

video_writer = cv2.VideoWriter(clip_path, video_codec,  fps, (int(cap1.get(3)), int(cap1.get(4))))

start_time = time.time()
done = 0

while cap1.isOpened():
    ret, frame = cap1.read()
    
    if ret is True:
        if done == 0:
            frame = cv2.flip(frame, 1)
            cv2.imshow("Webcam Feed", frame)
            cv2.waitKey(1)
            
            if time.time() - start_time >= 7:
                #print("Clip created")
                clip_timestamp2 = str(time.time()).split('.')[0]
                clip_path = os.path.join(video_folder, clip_timestamp2 + ".mp4")
                video_writer = cv2.VideoWriter(clip_path, video_codec,  fps, 
                                               (int(cap1.get(3)), int(cap1.get(4))))
                done = 1
                
            video_writer.write(frame)
        
        if done == 1:
            break

cap1.release()
cv2.destroyAllWindows()

#-----------------------------------------------------------------------------#

print("Stage 2: Extract Frames from the Clip")
'''
Stage 2: Extract Frames from the Clip
20% of the frames are extracted from the clip using a random sample selection
method, and are saved 
'''
clip_loc = os.path.join(video_folder, clip_timestamp + ".mp4")
cap2 = cv2.VideoCapture(clip_loc)
num_of_frames = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
num_of_frames_to_select = int(num_of_frames / 5)
#print(clip_timestamp, num_of_frames, num_of_frames_to_select)
    
frames_to_select = []
frame_counter = 0
while frame_counter < num_of_frames_to_select:
    r = random.randint(1, num_of_frames)
    #print(r)
    if r not in frames_to_select:
        #print("Selected")
        frames_to_select.append(r)
        frame_counter = frame_counter + 1
    
frame_index = 1
frame_counter_saved = 0
while(frame_counter_saved < frame_counter):
    ret, frame = cap2.read()
    if ret is True:
        if frame_index in frames_to_select:
            frame_counter_saved = frame_counter_saved + 1
            frame_name = clip_timestamp + '_' + str(frame_counter_saved) + '.jpg'
            frame_loc = os.path.join(frames_folder, frame_name)
            #print(frame_loc)
            cv2.imwrite(frame_loc, frame)
                
        frame_index = frame_index + 1
    
cap2.release()
cv2.destroyAllWindows()

#-----------------------------------------------------------------------------#

print("Stage 3: Preprocess the Frames")
'''
Stage 3: Preprocess the Frames
The frames saved go through the following preprocessing steps:
1. Hand Detection (using pre-trained YOLO-v3 network)
    - if hands are detected in the frame, a bounding box is made over the hand(s)
    - the frame is cropped to contain only the contents inside the bounding box
    - if no hand is detected, the frame is discarded
2. Image Resizing to (224, 224)
3. Skin Segmentation
'''
yolo = YOLO("yolo_models/cross-hands.cfg", "yolo_models/cross-hands.weights", ["hand"])
yolo.confidence = 0.5

_, _, files = next(os.walk(frames_folder))

for f in files:
    filename = str(f)
    
    if filename.split('_')[0] == clip_timestamp:
        input_file_location = os.path.join(frames_folder, filename)
        output_file_location = os.path.join(boxed_frames_folder, filename)

        frame = cv2.imread(input_file_location)
        
        x, y, x1, y1 = handDetector.hand_seg(input_file_location, output_file_location, yolo, 
                                                  color_green)
        
        if x is not None:
            output_file_location = os.path.join(segmented_frames_folder, filename)
            
            x = x - 54
            y = y - 30
            x1 = x1 + 54
            y1 = y1 + 30
            
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x1 > frame.shape[1]:
                x1 = frame.shape[1]
            if y1 > frame.shape[0]:
                y1 = frame.shape[0]
            
            cropped_frame = frame[y:y1, x:x1]
            
            resized_frame = resize.resize_image(cropped_frame, (224, 224))
            
            detector = skinDetector(resized_frame)
            output_frame = detector.find_skin()
            cv2.imwrite(output_file_location, output_frame)

#-----------------------------------------------------------------------------#

print("Stage 4: Use the Trained Model to Detect the Sign Displayed")
'''
Stage 4: Use the Trained Model to Detect the Sign Displayed
The frames are passed through the trained model and their results are 
collectively stored in a list
The output that occurs the most number of times is considered as the final output 
'''
model = keras.models.load_model("final_model")
_, _, files = next(os.walk(segmented_frames_folder))
opcount = defaultdict(int)

for f in files:
    filename = str(f)
    
    if filename.split('_')[0] == clip_timestamp:
        input_file_location = os.path.join(segmented_frames_folder, filename)
        opcount[predictImage(input_file_location, 'model_class.json', 1)[0][0]] += 1
    
print(opcount)
detected_sign = selectFinal.select_max(opcount)
print("Output: ", detected_sign)

#-----------------------------------------------------------------------------#