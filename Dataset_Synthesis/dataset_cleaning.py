import os
import sys
import time
import cv2
import pickle
from tqdm import tqdm
from yolo import YOLO
from collections import defaultdict

def handSeg(rootPath, alphabet, numHands, yolo):
    #Lists that contain files meeting or exceeding confidence intervals for hand detection
    # Eg if a pic input has a hand detected with a confidence of 0.6, it will got in halfCL but not the other two
    # Eg pic confidence = 0.78, it will go in halfCL and threeFourthCL
    halfCL = []
    threeFourthCL = []
    nineTenthCL = []
    inputPath = os.path.join(rootPath,alphabet)
    _, _, files = next(os.walk(inputPath))
    numFiles = len(files)
    # Average confidence for alphabet
    avgconf = 0
    # Counts amount of files with hands detected
    confCtr = 0
    for fileIP in files:
        
        # Getting inference from image
        input_img = cv2.imread(os.path.join(inputPath,fileIP))    
        width, height, inference_time, results = yolo.inference(input_img)
        
        if len(results) == numHands:
            conf = 0 # :(
            for detection in results:
                id, name, confidence, x, y, w, h = detection
                conf += confidence
            conf /= numHands
            avgconf += conf
            confCtr += 1
            # Puts file names in respective lists for pickling and future extraction operations
            if conf >= 0.9 :
                nineTenthCL.append(fileIP)
                threeFourthCL.append(fileIP)
                halfCL.append(fileIP)
            elif conf >= 0.75 :
                threeFourthCL.append(fileIP)
                halfCL.append(fileIP)
            elif conf >= 0.5 :
                halfCL.append(fileIP)
            else :
                # Picture confidence not high enough to warrant storage and further operations, move to next iteration
                # Statement is redundant, kept for continuity purposes
                continue
        else:
            # Inadequate number of hands for alphabet, hence continue
            continue 

    # Total - confidence wrt total files , Selected - conf wrt numfiles with hands detected 
    avgconfTotal = avgconf / numFiles
    avgconfSelected = avgconf / confCtr

    metaDataList = [alphabet,inputPath,halfCL,threeFourthCL,nineTenthCL]    
    
    # Pickles data for future use
    picklePath = os.path.join(os.getcwd(),'pickles',alphabet)
    dbfile = open(picklePath, 'ab')
    pickle.dump(metaDataList, dbfile)                     
    dbfile.close()

    return [len(halfCL),len(threeFourthCL),len(nineTenthCL),avgconfTotal,avgconfSelected,numFiles,confCtr]
            

        
# Main code 
start = time.time()
input_file_path = sys.argv[1]
yolo = YOLO("yolo_models/cross-hands.cfg", "yolo_models/cross-hands.weights", ["hand"])
# Makes pickles folder if not exists
if not os.path.exists(os.path.join(os.getcwd(),'pickles')):
    os.makedirs(os.path.join(os.getcwd(),'pickles'))

root, dirs, files = next(os.walk(input_file_path))
# Taken by checking hands used in isl20c1200isynth dataset
oneHandDict = ['I','L','O','U','V']
numAlphabets = len(dirs)
# Storing counts for metadata analysis
halfCt = defaultdict(float)
threeFourthCt = defaultdict(float)
nineTenthCt = defaultdict(float)
avgCharConfTotal = defaultdict(float)
avgCharConfSelected = defaultdict(float)
totalPics = 0
totalSelected = 0

for alphabet in tqdm(dirs):
    
    # Default case is 2 hands (15/20 alphabets)
    numHands = 2
    # TODO - integrate numHands and handSeg function
    print('Working on',alphabet)
    if alphabet in oneHandDict:
        numHands = 1
    
    opList = handSeg(root, alphabet, numHands, yolo)
    halfCt[alphabet] = opList[0]
    threeFourthCt[alphabet] = opList[1]
    nineTenthCt[alphabet] = opList[2]
    avgCharConfTotal[alphabet] = opList[3]
    avgCharConfSelected[alphabet] = opList[4]
    totalPics += opList[5]
    totalSelected += opList[6]
print('check results.txt for final results!')

sys.stdout = open("results.txt", "w")
print('-------------------------------------------------------------------------------')
print('Number of images exceeding 0.5 confidence alphabet wise :')
print(halfCt)
minHalf = min(halfCt.items(), key=lambda x: x[1]) 
print('Alphabet with minimum values and count for 0.5 confidence',minHalf)
print('Number of images in final dataset if 0.5 conf images selected :',minHalf[1] * numAlphabets)
print('Percentage of images passing in 0.5 case :',(minHalf[1] * numAlphabets * 100) / totalPics)
print('-------------------------------------------------------------------------------')
print('Number of images exceeding 0.75 confidence alphabet wise :')
print(threeFourthCt)
minTf = min(threeFourthCt.items(), key=lambda x: x[1]) 
print('Alphabet with minimum values and count for 0.75 confidence',minTf)
print('Number of images in final dataset if 0.75 conf images selected :',minTf[1] * numAlphabets)
print('Percentage of images passing in 0.75 case :',(minTf[1] * numAlphabets * 100) / totalPics)
print('-------------------------------------------------------------------------------')
print('Number of images exceeding 0.9 confidence alphabet wise :')
print(nineTenthCt)
minNt = min(nineTenthCt.items(), key=lambda x: x[1]) 
print('Alphabet with minimum values and count for 0.9 confidence',minNt)
print('Number of images in final dataset if 0.9 conf images selected :',minNt[1] * numAlphabets)
print('Percentage of images passing in 0.9 case :',(minNt[1] * numAlphabets * 100) / totalPics)
print('-------------------------------------------------------------------------------')
print('Total pictures present : ',totalPics)
print('Total pictures selected :',totalSelected)
print('Percentage of pictures where hand was detected : ',(totalSelected * 100 )/totalPics)
print('-------------------------------------------------------------------------------')
end = time.time()
print("Execution time (hrs) =", (end-start)/3600)
sys.stdout.close()