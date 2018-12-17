# camera-model-identification-PyTorch

## Description

Implementation of camera model identification models  in kaggle .

The repository contains the implmentation of various image classification models like ResNet,DPN,DenseNet,etc in PyTorch deep learning framework  .

models in this repo:

- densenet121
- densenet161
- densenet169
- densenet201
- resnet50
- resnet101
- resnet152
- resnext101
- dpn92
- dpn98
- se_resnet50
- se_resnet101
- se_resnext50

## Requirements

To train models and get predictions the following is required:

- Python 3.6

packages:

- torch==1.0.0
- torchvision==0.2.1
- scipy==1.0.0
- tqdm==4.28.1
- tensorboardX==1.4
- matplotlib==2.1.2
- opencv-python==3.4.2.17
- numpy==1.14.0
- pandas==0.20.3

## training

1. Install packages with `pip install -r requirements.txt`
2. Download dataset  from [kaggle](https://www.kaggle.com/c/sp-society-camera-model-identification/data)
3. Place train dataset from Kaggle competition to dataset/train. Place test dataset from Kaggle competition to dataset/test. 
4. run `python data_split.py`
5. run `python runTrain.py --batch_size=64 --pretrained=True --learning_rate=0.0001 --epochs=100`
6. run `python runMakePseudo.py --batch_size=128`
7. run `python runTrain.py --batch_size=64 --learning_rate=0.0001 --epochs=40 --resume=True --use_pseudo=True`
8. run `python runTest.py --batch_size=128`