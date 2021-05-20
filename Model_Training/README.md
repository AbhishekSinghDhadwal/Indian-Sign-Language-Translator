# SqueezeNet Model Trainng

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) &nbsp; 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AbhishekSinghDhadwal/Indian-Sign-Language-Translator/blob/main/Model_Training/ModelTraining_Squeezenet.ipynb)

The attached notebook has been used to train the squeezenet model used in our project using Transfer Learning. Originally made for training via Colab, can be run locally by changing the required DIRs.

We utilized the SqueezeNet model for our purposes as it outperformed its peers under various tests in the following categories (described in section 4.5 of the [thesis](https://github.com/AbhishekSinghDhadwal/Indian-Sign-Language-Translator/blob/main/Thesis/FYReport.pdf)) :

1.The accuracy provided

2.Time for training

If required, the code can be modified to transfer train via the following 4 models, you only need to download the pre-trained models and change the dir accordingly :

Model  | Link for downloading weights
------------- | -------------
SqueezeNet  | [weights](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5)
DenseNet  | [DenseNet-BC-121-32 weights](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/DenseNet-BC-121-32.h5)
ResNet 50 | [weights](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/inception_v3_weights_tf_dim_ordering_tf_kernels.h5)
Inceptionv3 | [weights](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/squeezenet_weights_tf_dim_ordering_tf_kernels.h5)

Refer the "Inputs for pre-trained model weights and data" of the notebook for placement of input folders and selection of parameters.
