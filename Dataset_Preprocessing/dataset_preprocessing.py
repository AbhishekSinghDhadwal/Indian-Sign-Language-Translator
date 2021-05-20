import cv2
from cv2 import dnn_superres
import os
from tqdm import tqdm
import sys
from yolo import YOLO
from preprocessing import handDetector, resize, skinDetector

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
path = "ESPCN_x3.pb"
sr.readModel(path)

sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
sr.setModel("espcn", 3)

input_folder = os.path.join(os.getcwd(), 'dataset', 'train')
#input_folder = os.path.join(os.getcwd(), 'dataset', 'val')
output_folder = os.path.join(os.getcwd(), 'dataset_preprocessed', 'train')
#output_folder = os.path.join(os.getcwd(), 'dataset_preprocessed', 'val')

color_green = (0, 255, 0)
yolo = YOLO("yolo_models/cross-hands.cfg", "yolo_models/cross-hands.weights", ["hand"])
yolo.confidence = 0.5

for root, dirs, files in os.walk(input_folder):
    if files != []:
        #print(root)
        slash_loc = root.rfind('\\')
        parent_dir = root[slash_loc + 1:]
        output_parent_location = os.path.join(output_folder, parent_dir)
        count = 0
        
        for f in files:
            filename = str(f)
            input_img_location = os.path.join(root, filename)
            input_img = cv2.imread(input_img_location)
            output_img_location = os.path.join(output_folder, filename)
            #print(input_img_location)
            #print(output_img_location)
            
            #1. Hand Detection
            x, y, x1, y1 = handDetector.hand_seg(input_img_location, output_img_location, yolo, color_green)
            
            if x is not None:  
                x = x - 54
                y = y - 30
                x1 = x1 + 54
                y1 = y1 + 30
                
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0
                if x1 > input_img.shape[1]:
                    x1 = input_img.shape[1]
                if y1 > input_img.shape[0]:
                    y1 = input_img.shape[0]
            
                cropped_img = input_img[y:y1, x:x1]
            
            #2. Resizing
            resized_img = resize.resize_image(cropped_img, (224, 224))
            
            #3. Skin Seg
            detector = skinDetector(resized_img)
            output_img = detector.find_skin()

            #4.Resizing
            
            cv2.imwrite(output_img_location, output_img)