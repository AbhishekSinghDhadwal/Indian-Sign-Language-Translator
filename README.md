# Indian-Sign-Language-Translator
[![Info](https://img.shields.io/badge/Usage-Instructions-blue?style=flat-square&logo=data:image/svg%2bxml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iaXNvLTg4NTktMSI/Pg0KPCEtLSBHZW5lcmF0b3I6IEFkb2JlIElsbHVzdHJhdG9yIDE5LjAuMCwgU1ZHIEV4cG9ydCBQbHVnLUluIC4gU1ZHIFZlcnNpb246IDYuMDAgQnVpbGQgMCkgIC0tPg0KPHN2ZyB2ZXJzaW9uPSIxLjEiIGlkPSJDYXBhXzEiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIgeG1sbnM6eGxpbms9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkveGxpbmsiIHg9IjBweCIgeT0iMHB4Ig0KCSB2aWV3Qm94PSIwIDAgNTEyIDUxMiIgc3R5bGU9ImVuYWJsZS1iYWNrZ3JvdW5kOm5ldyAwIDAgNTEyIDUxMjsiIHhtbDpzcGFjZT0icHJlc2VydmUiPg0KPHBhdGggc3R5bGU9ImZpbGw6IzBBNEVBRjsiIGQ9Ik0yNTYsNTEyYy02OC4zOCwwLTEzMi42NjctMjYuNjI5LTE4MS4wMi03NC45OEMyNi42MjksMzg4LjY2NywwLDMyNC4zOCwwLDI1Ng0KCVMyNi42MjksMTIzLjMzMyw3NC45OCw3NC45OEMxMjMuMzMzLDI2LjYyOSwxODcuNjIsMCwyNTYsMHMxMzIuNjY3LDI2LjYyOSwxODEuMDIsNzQuOThDNDg1LjM3MSwxMjMuMzMzLDUxMiwxODcuNjIsNTEyLDI1Ng0KCXMtMjYuNjI5LDEzMi42NjctNzQuOTgsMTgxLjAyQzM4OC42NjcsNDg1LjM3MSwzMjQuMzgsNTEyLDI1Niw1MTJ6Ii8+DQo8cGF0aCBzdHlsZT0iZmlsbDojMDYzRThCOyIgZD0iTTQzNy4wMiw3NC45OEMzODguNjY3LDI2LjYyOSwzMjQuMzgsMCwyNTYsMHY1MTJjNjguMzgsMCwxMzIuNjY3LTI2LjYyOSwxODEuMDItNzQuOTgNCglDNDg1LjM3MSwzODguNjY3LDUxMiwzMjQuMzgsNTEyLDI1NlM0ODUuMzcxLDEyMy4zMzMsNDM3LjAyLDc0Ljk4eiIvPg0KPHBhdGggc3R5bGU9ImZpbGw6I0ZGRkZGRjsiIGQ9Ik0yNTYsMTg1Yy0zMC4zMjcsMC01NS0yNC42NzMtNTUtNTVzMjQuNjczLTU1LDU1LTU1czU1LDI0LjY3Myw1NSw1NVMyODYuMzI3LDE4NSwyNTYsMTg1eiBNMzAxLDM5NQ0KCVYyMTVIMTkxdjMwaDMwdjE1MGgtMzB2MzBoMTQwdi0zMEgzMDF6Ii8+DQo8Zz4NCgk8cGF0aCBzdHlsZT0iZmlsbDojQ0NFRkZGOyIgZD0iTTI1NiwxODVjMzAuMzI3LDAsNTUtMjQuNjczLDU1LTU1cy0yNC42NzMtNTUtNTUtNTVWMTg1eiIvPg0KCTxwb2x5Z29uIHN0eWxlPSJmaWxsOiNDQ0VGRkY7IiBwb2ludHM9IjMwMSwzOTUgMzAxLDIxNSAyNTYsMjE1IDI1Niw0MjUgMzMxLDQyNSAzMzEsMzk1IAkiLz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjxnPg0KPC9nPg0KPGc+DQo8L2c+DQo8Zz4NCjwvZz4NCjwvc3ZnPg0K)](https://github.com/AbhishekSinghDhadwal/Indian-Sign-Language-Translator/tree/main/App#usage-instructions-)&nbsp; [![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) 

This repository consists of the code utilized for creation of an Indian Sign Language Translator satisfying the following criteria :
- **Near-Real-Time Application**
- **Achieve background independence**
- **Attain Illumination independence**


<p align="center">
  <img src="https://user-images.githubusercontent.com/39513876/118981340-dcc48900-b997-11eb-98d8-686dd402f544.png" width="650">
</p>

<div align="center"> <i>Flow Diagram of the project implementation</i> </div>
</n>

We achieve these goals by providing the following features :
1. **Face Detection** 
: Used as an activation mechanism. We use Haar cascade models from the OpenCV library to detect faces in an image. When a face is detected, the program checks the next few consecutive frames, and when a threshold value of consecutive frames with a face in it is reached, the sign language system is triggered.

2. **Hand Detection** 
: The first step in preprocessing is preliminary hand detection method, which goes through every frame selected from the clip, and attempts to find a hand in them using a YOLO-v3 pre-trained network. 
If any hands are found in the frame, an extended bounding box is created around the hand(s). These images are then cropped to contain only the contents of the box, and are passed onto the next step of preprocessing which is resizing. If no hands are found, the frame is discarded entirely. 

3. **Skin Segmentation** 
: After cropping and resizing, the images are passed through a combination of HSV (Hue, Saturation, Value) and YCbCr (Luminance, Chrominance) based filters to segment out skin and remove background noise present in the box input.

4. **Sign Recognition** 
: The processed input is passed through a [SqueezeNet](https://arxiv.org/abs/1602.07360) model trained (via Transfer Learning) on a synthesized and cleaned Indian Sign Language dataset consisting of 10 classes, and ~2300+ images per class.

<p align="center">
  <img src="https://user-images.githubusercontent.com/39513876/119128821-c54ad600-ba53-11eb-94b1-25727c70800b.jpg" width="500">
</p>

<div align="center"> <i>Hand Detection performed on the Live Feed Input</i> </div>
</n>

The work performed is divided into the following **folders** :

### Main App
The [App](https://github.com/AbhishekSinghDhadwal/Indian-Sign-Language-Translator/tree/main/App) section consists of the files required to run the standalone webcam implementation of the translator. Contains :
- The trained model
- The hand segmentation network
- Preprocessing scripts
- Main application (main.py)


### Dataset Synthesis
Covers the [scripts](https://github.com/AbhishekSinghDhadwal/Indian-Sign-Language-Translator/tree/main/Dataset_Synthesis) used in :
- Creating new data, via modifications on brightness, clarity and picture quality (Synthesis)
- Cleaning noisy generated data from the previous step, by using the YOLO-v3 Hand Detection Network (Cleaning)

### Dataset Preprocessing
Contains the [scripts](https://github.com/AbhishekSinghDhadwal/Indian-Sign-Language-Translator/tree/main/Dataset_Preprocessing) used in order to perform pre-processing on the input dataset, including image upscaling, skin segmentation and hand centralization. These tasks are performed before entering the image dataset into the neural network.

### Model Training
Consists of the [notebook](https://github.com/AbhishekSinghDhadwal/Indian-Sign-Language-Translator/tree/main/Model_Training) used in order to train and save the SqueezeNet model used for the project. Originally made in Colab.

### Dependencies
- OpenCV
- Tensorflow
- Keras
- Numpy
- Pillow
- ImageAI

The specific versions are mentioned in requirements.txt

### Contributors:

This translator was originally created as part of our Final Year Project, consisting of the following members -

| Name | GithubID |
| ----------- | ----------- |
| Abhishek Singh Dhadwal | [AbhishekSinghDhadwal](https://github.com/AbhishekSinghDhadwal) |
| Saurabh Pujari | [saurabh0719](https://github.com/saurabh0719) | 
| Kopal Bhatnagar | [kopalbhatnagar05](https://github.com/kopalbhatnagar05) | 
| Yash Kumar | [yashKumar2412](https://github.com/yashKumar2412) | 

#### Credits:
1. Jeanvit, for the skin segmentation algorithm [code](https://github.com/Jeanvit/PySkinDetection)
2. Cansik, for the hand detection [NN](https://github.com/cansik/yolo-hand-detection/)

For further details on the implementation, kindly refer to the **Thesis** folder containing both the project report and the final presentation.
