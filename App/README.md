# Main Application

### Usage Instructions :

1. Clone the repository and download **cross-hands.weights** from this [link](https://github.com/cansik/yolo-hand-detection/releases/download/pretrained/cross-hands.weights) and place it in the "yolo-models" folder.
2. Run ```python main.py``` on your terminal, the application will start.

### How it works :

1. **Activation Phase** :The user The first phase involves Face Detection. The intuition behind this is that the user will look at the camera for a couple of seconds straight when they wish to send a message, and doing so will then activate the next phase of the pipeline, Video Recording. 
2. **Recording Phase** : For the next ‘x’ seconds, the user will hold up the symbol that corresponds to their message. This video clip is saved and a random sample of the frames, equivalent to 20% of the total number of frames, are extracted from it. 
3. **Preprocessing Phase** : These frames go through the following pre-processing steps, i.e., 
- Hand Detection
- Image Resizing
- Skin Segmentation 
4. **Result Phase** : The preprocessed images obtained are then passed through the model we had trained in the first implementation, which predicts an output for the given frame. The output which occurs the maximum number of times is selected as the final output.

The respective sections are commented in the main.py code for reference.

### File Description :

- ```final_model``` : Contains the trained model in the SavedModel format
- ```model_class.json``` : Stores the dictionary used to print final results
- ```preprocessing.py``` : Contains the functions used to detect hands, resize the input image and to detect and segment skin
- ```select_final.py``` : Contains the functions used to determine the frame selection criteria
- ```yolo.py``` : Stores the YOLO class, used to perform hand detection on input images, inputs for which are stored in the yolo_models folder
