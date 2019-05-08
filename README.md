# Facial Emotion Recognition with Apache MXNet

This repository demonstrates the implementation and deployoment of a facial expression recognition deep learning model using Apache MXNet, based on the [FER+ paper by Barsoum et. al.](https://arxiv.org/abs/1608.01041).

The repository consists of the following resources:
1. Scripts for data pre-processing as suggested in the paper.
1. Notebook for model building and training, using [Apache MXNet](http://mxnet.io).
1. Model deployment for online inference, using [MXNet Model Server](http://modelserver.io)

Check out a working [FER+ web application demo](http://bit.ly/mxnet-fer), powered by AWS.


# Before you start

## Pre-requisites

```
# Install MXNet

pip install mxnet-mkl # for CPU machines
pip install mxnet-cu92 # for GPU machines with CUDA 9.2
    
# Other Dependencies

pip install Pillow # For image processing
pip install graphviz # For MXNet network visualization
pip install matplotlib # For plotting training graphs

# Install MXNet Model Server and required dependencies for inference model serving

pip install mxnet-model-server
pip install scikit-image
pip install opencv-python
```

*Note: please refer to [MXNet installation guide](http://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU) for more detailed installation instructions.*

## Data preparation

Clone this repository

```
git clone https://github.com/TalkAI/facial-emotion-recognition-gluon
cd facial-emotion-recognition-gluon
```

Download FER dataset `fer2013.tar.gz` from the [FER Kaggle competition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

*Note: You cannot download the dataset with `wget`. You will have to register on Kaggle, and then login to download the dataset.*

Once downloaded:

* Extract the tar file - `fer2013.tar.gz`
* Copy `fer2013.csv` dataset to `facial-emotion-recognition-gluon/data` directory. 

We will now generate `FER+` train/test/validation dataset from the downloaded `FER` data by executing the command below:

```
# In this step, we read the raw FER data, correct the labels using FER+ labels, and save as png images.

# -d : path to "data" folder in this repository. It has folder for Train/Test/Validation data with corrected labels.
# -fer : path to fer dataset that you have extracted.
# -ferplus : path to fer2013new.csv file that comes with this repository in the data folder
    
python utils/prepare_data.py -d ./data -fer ./data/fer2013.csv -ferplus ./data/fer2013new.csv
```

Lastly, we will process the `FER+` train/test/validation dataset

```
# This script reads the FER+ dataset (png images) we prepared in the previous step, applies the transformation suggested in the FER+ paper, and saves the processed images as NumPy binaries (npy files).

# -d : path to data folder. This is where we have created data from the previous step.

python utils/process_data.py -d ./data
```
 
## Deep Learning Basics You Will Need
Go over [this notebook](https://github.com/TalkAI/facial-emotion-recognition-gluon/tree/master/notebooks/Deep_Learning_Basics_Intuitions.ipynb) that provides basic overview and intuitiion for various deep learning concepts and techniaues used in building and training the model.

# Model Building, Training and Deployment
Head over to the [FER+ tutorial](https://github.com/TalkAI/facial-emotion-recognition-gluon/tree/master/notebooks/Gluon_FERPlus.ipynb), to go over the process for building, training and deploying FER+ model. It is best to run as a live Jupyter Notebook, and you would need a GPU machine to complete training in a reasonable time. 

# Advanced Stuff - Left To The Reader

Below are few areas of improvements and next steps for the advanced reader. Contributions back to the repository are welcomed!

* Hyper-parameter optimization - In this implementation, I have not optimized hyper-parameters (learning rate scheduler for SGD) for best possible result.
* Implement multi-gpu version of model training. This script provides single GPU implementation only. Time per epoch on single GPU is around 1 minute => approx 50 minutes for full model training (Model converges at around 50th epoch)

# Contributors

* [Sandeep Krishnamurthy](https://github.com/sandeep-krishnamurthy/) 
* [Saravana Kumar](https://github.com/codewithsk)
* [Hagay Lupesko](https://github.com/lupesko)

# Citation / Credits

* Barsoum, Emad & Zhang, Cha & Canton Ferrer, Cristian & Zhang, Zhengyou (2016). [Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution](https://arxiv.org/abs/1608.01041). In ACM International Conference on Multimodal Interaction (ICMI).
* CNTK implementation of FER+ paper - https://github.com/Microsoft/FERPlus
* FER demo built by https://github.com/codewithsk. (GitHub repo will be public soon)

# Resources

* Apache MXNet (incubating) - http://mxnet.incubator.apache.org/
* Learn Deep Learning with Gluon - https://gluon.mxnet.io/
* Productionizing Deep Learning Models with MXNet Model Server - https://github.com/awslabs/mxnet-model-server
