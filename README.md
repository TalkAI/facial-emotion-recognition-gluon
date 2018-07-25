# Facial Emotion Recognition with Apache MXNet GLUON and MXNet Model Server

This repo is an Apache MXNet GLUON implementation of state of the art [FER+ paper by Barsoum et. al.](https://arxiv.org/abs/1608.01041) for facial emotion recognition.

This repo consists of following resources:
1. Scripts for data preprocessing.
2. Notebook for model training.
3. Large scale productionization of the trained model using MXNet Model Server(MMS) - https://github.com/awslabs/mxnet-model-server

You can see final demo at - http://bit.ly/ferdemo
Note: Please use firefox or safari browser. Chrome is not supported yet.

# Model Training

## Data preparation

## Model Training and Saving

# Inference

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
