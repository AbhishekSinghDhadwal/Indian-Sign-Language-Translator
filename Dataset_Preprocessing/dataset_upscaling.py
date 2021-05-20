import cv2
from cv2 import dnn_superres
import os
from tqdm import tqdm
import sys

# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()

# Read the desired model
path = "ESPCN_x3.pb"
sr.readModel(path)

sr.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
sr.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("espcn", 3)
root,dirs,files = next(os.walk(sys.argv[1]))

for alphabet in tqdm(dirs):
    root2,_,files = next(os.walk(os.path.join(sys.argv[1],alphabet)))
    for pic in tqdm(files) :
        image = cv2.imread(os.path.join(root2,pic))
        result = sr.upsample(image)
        cv2.imwrite(os.path.join(root2,pic), result)
