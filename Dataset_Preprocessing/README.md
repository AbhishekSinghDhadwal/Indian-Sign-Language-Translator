# Dataset Preprocessing

### Usage Instructions :

1. Clone the repository and download **cross-hands.weights** from this [link](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights) and place it in the "yolo-models" folder.
2. Store the input (dataset) and output (preprocessed_dataset) folders within the repository folder. Note that both contain two subfolders - train and val. Each of these then further contains a number of folders, each corresponding to their class (i.e., letters).
3. Run ```python dataset_preprocessing.py``` on your terminal, the application will start. After execution, the resultant ouptut will be stored in the corresponding output folder.
4. Change the source folders from train to val in the code by modifying the code in dataset_preprocessing.py script lines 20-23.
5. Run ```python dataset_preprocessing.py``` on your terminal, the application will start. After execution, the resultant ouptut will be stored in the corresponding output folder.


<p align="center">
  <img src="https://user-images.githubusercontent.com/39513876/119353680-cf254100-bcc0-11eb-9f24-004adc3e636b.png" width="500">
</p>
<div align="center"> <i>Pipeline diagram of the pre-processing and training phase</i> </div>
</n>

### How it works :

1. **Reading Phase** : The images are read one by one from the dataset/train or dataset/val folder. 
2. **Preprocessing Phase** : These frames go through the following pre-processing steps, i.e., 
- Hand Detection
- Image Resizing
- Skin Segmentation 
5. **Result Phase** : The preprocessed images obtained are then stored in the output folder.

### File Description :

- ```dataset_upscaling.py``` : Script used to perform image upscaling on sample datasets. Run ```python dataset_upscaling.py folder_location``` to perform upscaling on all the images in the folder. Currently performs in-situ upscaling, but can be modified to store results in an external folder.
- ```ESPCN_x3.pb``` : The model utilized for performing image upscaling on sample datasets.
- ```preprocessing.py``` : Contains the functions used to detect hands, resize the input image and to detect and segment skin
- ```yolo.py``` : Stores the YOLO class, used to perform hand detection on input images, inputs for which are stored in the yolo_models folder
