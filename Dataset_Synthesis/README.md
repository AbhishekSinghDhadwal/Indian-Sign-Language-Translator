# Dataset Synthesis

### Usage Instructions :

1. Clone the repository and download **cross-hands.weights** from this [link](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights) and place it in the "yolo-models" folder.
2. Store the input (dataset) and output (modded_dataset) folders within the repository folder. 
3. Run ```python dataset_synthesis.py``` on your terminal, the application will start. After execution, the resultant ouptut will be stored in the corresponding output folder.
4. Run ```python dataset_cleaning.py modded_dataset``` on your terminal, where modded_dataset is the input folder to this script. The application will then start, and after execution, the resultant ouptut will be stored in a newly created pickles folder containing information regarding the images.

<p align="center">
  <img src="https://user-images.githubusercontent.com/39513876/119351798-85d3f200-bcbe-11eb-9f70-699c15256185.png" width="500">
</p>
<div align="center"> <i>Sample modifications</i> </div>
</n>


### How it works :

1. **Dataset Synthesis** :
- **Input Phase** : Each image is read one by one. 
- **Generation Phase** : Each image is then brightened or darkened, and blurred by a random factor within fixed ranges. Each image generates 18 such images with varying factors of brightness and focus. 
- **Output Phase** : The newly generated images are stored in the output folder

2. **Dataset Cleaning**
- **Input Phase** : Each subdirectory corresponding to a class (i.e., letters) are considered as input.
- **Hand Detection Phase** : Each image inside the subdirectory is passed through the YOLO hand detection algorithm, and it outputs the number of hands found in the image as well as the confidence of detection. If the confidence is satisfies the three threshold values (i.e., 0.5, 0.75, 0.9), it is added to the dictionaries corresponding to these values which it has satisfied.
-  **Output Phase** : After the hand detection phase is completed for the entire subfolder, the alphabet, input path, and the three dictionaries are stored as metadata within the output folder. Furthermore, after the whole process has been completed for all the subfolders, a summarization of the whole data is stored in results.txt.  

### File Description :

- ```yolo.py``` : Stores the YOLO class, used to perform hand detection on input images, inputs for which are stored in the yolo_models folder
