# [ICCV 2023] MEFLUT: Unsupervised 1D Lookup Tables for Multi-exposure Image Fusion [Paper](https://arxiv.org/abs/2309.11847)

<h4 align="center">Ting Jiang<sup>1,*</sup>, Chuan Wang<sup>1</sup>,  Xinpeng Li<sup>1</sup>,  Ru Li<sup>1</sup>,  Haoqiang Fan<sup>1</sup>,  Shuaicheng Liu<sup>2,1,†</sup></center>
<h4 align="center"> 1. Megvii Research, 2. University of Electronic Science and Technology of China</center>
<h6 align="center"> †Corresponding author</center>


## Abstract

In this paper, we introduce a new approach for high-quality multi-exposure image fusion (MEF). We show that the fusion weights of an exposure can be encoded into a 1D lookup table (LUT), which takes pixel intensity value as input and produces fusion weight as output. We learn one 1D LUT for each exposure, then all the pixels from different exposures can query 1D LUT of that exposure independently for high-quality and efficient fusion. Specifically, to learn these 1D LUTs, we involve attention mechanism in various dimensions including frame, channel and spatial ones into the MEF task so as to bring us significant quality improvement over the state-of-the-art (SOTA). In addition, we collect a new MEF dataset consisting of 960 samples, 155 of which are manually tuned by professionals as ground-truth for evaluation. Our network is trained by this dataset in an unsupervised manner. Extensive experiments are conducted to demonstrate the effectiveness of all the newly proposed components, and results show that our approach outperforms the SOTA in our and another representative dataset SICE, both qualitatively and quantitatively. Moreover, our 1D LUT approach takes less than 4ms to run a 4K image on a PC GPU. Given its high quality, efficiency and robustness, our method has been shipped into millions of Android mobiles across multiple brands world-wide.

## Pipeline

![pipeline](./assets/pipeline.png)

## Our Dataset
The  dataset including training and testing dataset can be download from [Baidu Netdisk](Link: https://pan.baidu.com/s/1F3Re_2CWYibsqejN-pA2tw?pwd=2tih 提取码: 2tih ) or [Huggingface](https://huggingface.co/datasets/peterJing/MEFLUT/tree/main). The dataset is organized as follow:

```
Training set
|--train
|  |--0  // Sample ID, internal file ranging from EV-4 to EV+2. 
|  |  |--0.jpg  
|  |  |--1.jpg
|  |  |--2.jpg
|  |  |--3.jpg
|  |--1
|  |--2
|  |--3
|  |--...
|--train.txt //  Training set index.

Test set
|--test
|  |--0 // Sample ID, internal file ranging from EV-4 to EV+2. 
|  |  |--0.jpg  
|  |  |--1.jpg
|  |  |--2.jpg
|  |  |--3.jpg
|  |--1
|  |--2
|  |--3
|  |--...
|--test.txt //  Test set index.
```
I write a data_selection.py code under the data_utils folder to handle the user can select 2, 3 or 4 frames to operate as needed. In the actual training, our experiments are also randomly selected, so there may be bias in the experimental results.

## Usage
### Requirements

This code is developed under 

* Python 3.7.10
* Pytorch 1.12.1
* torchvision 0.13.1
* CUDA 10.0 on Ubuntu 18.04

We strongly recommend you using anaconda to ensure you can get the same results as us. 

Install the require dependencies:

```bash
conda create -n meflut  python=3.7.10
conda activate meflut
pip install -r requirements.txt
```

## Usage
### Training
```
1. cd MEFLUT
2. python main.py --status 1dluts_train # execute in GPU
```

### Testing
```
1. cd MEFLUT
2. python main.py --status 1dluts_eval # execute in GPU
```
## Citation

If you find this work helpful, please cite our paper:

```
@InProceedings{Jiang_2023_ICCV,
    author    = {Jiang, Ting and Wang, Chuan and Li, Xinpeng and Li, Ru and Fan, Haoqiang and Liu, Shuaicheng},
    title     = {MEFLUT: Unsupervised 1D Lookup Tables for Multi-exposure Image Fusion},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {10542-10551}
}
``` 
