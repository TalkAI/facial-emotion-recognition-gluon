# Facial Emotion Recognition with Apache MXNet GLUON and MXNet Model Server

This repo is an Apache MXNet GLUON implementation of state of the art [FER+ paper by Barsoum et. al.](https://arxiv.org/abs/1608.01041) for facial emotion recognition.

This repo consists of following resources:
1. Scripts for data pre-processing.
2. Notebook for model training.
3. Large scale productionization of the trained model using MXNet Model Server(MMS) - https://github.com/awslabs/mxnet-model-server

You can see final demo at - http://bit.ly/ferdemo
Note: Please use firefox or safari browser. Chrome is not supported yet.

In this implementation, we use majority voting (MV) technique illustrated in the paper and achieve `81.25%` validation accuracy.(paper achieves 83.85%)

# Prerequisites / Installation

```
# Install MXNet

    pip install mxnet-mkl # for CPU machines
    pip install mxnet-cu92 # For GPU machines with CUDA 9.2
    
# Other Dependencies
    pip install Pillow # For image processing
    pip install graphviz # For MXNet network visualization
    pip install matplotlib # For plotting training graphs
```

Please refer [MXNet installation guide](http://mxnet.incubator.apache.org/install/index.html?platform=Linux&language=Python&processor=CPU) for more details.

# Model Training

## Step 1 - Data preparation

* Clone the repository

```
    git clone https://github.com/sandeep-krishnamurthy/facial-emotion-recognition-gluon
    cd facial-emotion-recognition-gluon
```

* Download FER dataset `fer2013.tar.gz` from - https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Note: You cannot download the dataset with wget. You have register on Kaggle and download the dataset due to licence criteria.

* Extract the tar file - `fer2013.tar.gz`
* Copy `fer2013.csv` dataset to `facial-emotion-recognition-gluon/data` directory. 
* Generate `FER+` train/test/validation dataset from downloaded `FER` data.
```
    python utils/prepare_data.py -d ./data -fer ./data/fer2013.csv -ferplus ./data/fer2013new.csv
```
In this step, we read the raw FER data, correct the labels using FER+ labels, and save as png images.

* Process the `FER+` train/test/validation dataset

```
    python utils/process_data.py -d ./data
```
In this step, we read the FER+ data images we prepared in the previous step, apply the transformation suggested in the [paper by Boursom et. al.](https://arxiv.org/abs/1608.01041) (Crop, Flip, Rotate, Affine Transformation, Scale).

Processed training/test/validation data are saved as numpy binaries (`*.npy`). We use these processed data in the model training notebook.

 
## Step 2 - Model Training and Saving

Open the `notebook/Gluon_FERPlus.ipynb` notebook and follow the instructions to train the model using the processed images saved as *.npy files in previous step. 

# Inference

TODO

# Contributors

* [Sandeep Krishnamurthy](https://github.com/sandeep-krishnamurthy/) 
* [Saravana Kumar](https://github.com/codewithsk)
* [Hagay Lupesko](https://github.com/lupesko/sentiment-analysis-with-sagemaker-mxnet)

# Citation / Credits

* FER+ paper
@inproceedings{BarsoumICMI2016,
    title={Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
    author={Barsoum, Emad and Zhang, Cha and Canton Ferrer, Cristian and Zhang, Zhengyou},
    booktitle={ACM International Conference on Multimodal Interaction (ICMI)},
    year={2016}
}

* CNTK implementation of FER+ paper - https://github.com/Microsoft/FERPlus
* FER demo built by https://github.com/codewithsk. (GitHub repo will be public soon)

# Resources

* Apache MXNet (incubating) - http://mxnet.incubator.apache.org/
* Learn Deep Learning with Gluon - https://gluon.mxnet.io/
* Productionizing Deep Learning Models with MXNet Model Server - https://github.com/awslabs/mxnet-model-server
