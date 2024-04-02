# Universal-Medical-SSL
This repo is the official implementation of  [Knowledge Prompt Meets Semi-Supervised 3D Medical Image Segmentation ](https//).

<p align="center">
<img src=".//.png" width=60% height=65% class="center">
</p>

Our study highlights the potential of additional knowledge prompts in semi-supervised medical image segmentation, and we have integrated the proposed method and a wide range of existing state-of-the-art methods into a unified platform: [Universal-Medical-SSL](https://github.com/Lry777/Universal-Medical-SSL).

-------

## ![~DA422Z_P `TJ3Z$6HM2}$M](https://github.com/Lry777/Universal-Medical-SSL/assets/102654909/b251ca2f-cbdb-497f-894f-ebfe7fbd5ea3) Motivation

Currently, proposed semi-supervised medical segmentation algorithms tend to design more complex network learning frameworks, overlooking the utilization of additional knowledge. Considering the limited information contained in the image data itself, this study pro-poses a knowledge Prompt semi-supervised Network based on consistency learning for medical image segmentation, named KP-Net.

<p align="center">
<img src=".//.png" width=60% height=65% class="center">
</p>

Specifically, we define multi-scale selective fusion rules and a Token Joint Gate mechanism from the perspectives of edge-enhanced image prompts and text prompts, for the efficient fusion of different modal features. Furthermore, to effectively conduct consistency learning from perturbation estimates, we combine the consistency of different decoding branches and the distinctiveness of the same decoding branch, and enhance the common consistency loss. We validated the effectiveness of the proposed method by comparing it with various advanced methods on four publicly available datasets with multiple modalities.Encouragingly, our method achieves optimal performance with minimal training resources.

-------

## ![4N2C{HOSOWSP$5US{YQ_UAI](https://github.com/Lry777/Universal-Medical-SSL/assets/102654909/bf35b887-c53a-4c7d-acae-086940e3de77) Usage

### 1.Environment

First, clone the repo:

```shell
git 
cd
```

Then, create a new environment and install the requirements:
```shell
conda create -n KP-Net python=3.8
cd KP-Net/




```
### 2.Data Preparation & Pre-trained Weight

#### 2.1 Dataset 2-Class Left Atrium

Download the dataset with :
```shell
```
#### 2.2 Dataset 2-Class Pancreas-CT
#### 2.3 Dataset Multi-class BTCV
#### 2.4 Dataset Multi-class LiTS

Then your file structure will be like:

```
```
Next, download the following [pretrained weights](htttps)

-------

### 3.Training $ Evaluating

```bash

```
-------

### 4.Results

#### 4.1 Results of 2-Class Left Atrium Segmentation Task

<p align="left">
<img src=".//.png" width=60% class="center">
</p>

#### 4.2 Results of 2-Class Pancreas-CT Segmentation Task
#### 4.3 Results of Multi-class BTCV Segmentation Task
#### 4.4 Results of Multi-class LiTS Segmentation Task


## Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{KP-Net,
  title={KP-Net: Knowledge Prompt Meets Semi-Supervised 3D Medical Image Segmentation},
  author={},
  booktitle={},
  year={2024}
}
```

## Acknowlegement
