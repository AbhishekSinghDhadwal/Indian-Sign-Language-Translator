import cv2
from PIL import Image
import numpy as np

'''
Uses pre-trained YOLO-v3 hand detection network to find hands in the frame
''' 
class handDetector:
    def hand_seg(input_file_location, output_file_location, yolo, color):
    
        input_img = cv2.imread(input_file_location)
        width, height, inference_time, results = yolo.inference(input_img)
        
        x = None
        y = None
        x_1 = None
        y_1 = None
        
        if len(results) == 1:
            
            id, name, confidence, x, y, w, h = results[0]
            x_1 = x + w
            y_1 = y + h
                    
            cv2.rectangle(input_img, (x, y), (x_1, y_1), color, 1)
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(input_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
    
        elif len(results) == 2:
            id1, name, confidence1, x1, y1, w1, h1 = results[0]
            id2, name, confidence2, x2, y2, w2, h2 = results[1]
            
            x = min(x1, x2)
            y = min(y1, y2)
            x_1 = max(x1 + w1, x2 + w2)
            y_1 = max(y1 + h1, y2 + h2)
            
            cv2.rectangle(input_img, (x, y), (x_1, y_1), color, 1)
            confidence = (confidence1 + confidence2) / 2
            text = "%s (%s)" % (name, round(confidence, 2))
            cv2.putText(input_img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, 1)
            
        cv2.imwrite(output_file_location, input_img)
        return x, y, x_1, y_1

'''
Resizes all frames to (224,224)
''' 
class resize:
    def resize_image(input_img, size):
        img_pil = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_pil)
        resized_img_pil = img_pil.resize(size)
        output_img = np.asarray(resized_img_pil)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
        return output_img

'''
Detects & segments skin of the user in the frame
''' 
class skinDetector(object):
    def __init__(self, imageIP):
        #self.image = cv2.imread(imageName)
        self.image = imageIP
        if self.image is None:
            print("IMAGE NOT FOUND")
            exit(1)
        self.HSV_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.YCbCr_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCR_CB)
        self.binary_mask_image = self.HSV_image
    
    def find_skin(self):
        self.__color_segmentation()
        output = self.__region_based_segmentation()
        return output
    
    def __color_segmentation(self):
        lower_HSV_values = np.array([0, 40, 0], dtype = "uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype = "uint8")
        lower_YCbCr_values = np.array((0, 138, 67), dtype = "uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype = "uint8")
        mask_YCbCr = cv2.inRange(self.YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(self.HSV_image, lower_HSV_values, upper_HSV_values)
        self.binary_mask_image = cv2.add(mask_HSV,mask_YCbCr)
    
    def __region_based_segmentation(self):
        image_foreground = cv2.erode(self.binary_mask_image,None,iterations = 3)     	#remove noise
        dilated_binary_image = cv2.dilate(self.binary_mask_image,None,iterations = 3)   #The background region is reduced a little because of the dilate operation
        ret,image_background = cv2.threshold(dilated_binary_image,1,128,cv2.THRESH_BINARY)  #set all background regions to 128
        image_marker = cv2.add(image_foreground,image_background)   #add both foreground and backgroud, forming markers. The markers are "seeds" of the future image regions.
        image_marker32 = np.int32(image_marker) #convert to 32SC1 format
        cv2.watershed(self.image,image_marker32)
        m = cv2.convertScaleAbs(image_marker32) #convert back to uint8
        ret,image_mask = cv2.threshold(m,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        output = cv2.bitwise_and(self.image,self.image,mask = image_mask)
        
        return output